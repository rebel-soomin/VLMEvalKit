"""PP-OCRv5 recognition — pipeline contract and output formatting.

``RBLNPPOCRv5Rec`` reads text but cannot localise it, so on page-level inputs
it depends on a detector for line crops. Three things in that arrangement fail
*silently* if they regress, so they are pinned here:

* the **frozen box cache** must refuse a miss rather than fall back to live
  detection — a silent fallback makes any NPU-vs-GPU recognition comparison
  unattributable while still producing plausible numbers;
* the **character dictionary** must match the artifact's class count — the
  graph is shared across language variants, so a mismatched dictionary decodes
  to wrong characters instead of raising;
* **CTC decoding** and the **quadrilateral crop** must follow PaddleOCR, whose
  conventions (blank at index 0, collapse repeats, rotate tall crops) have no
  in-band failure signal.

No NPU and no artifact: the backend functions are pure, and the wrapper's
formatting surface is exercised on an unbound instance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vlmeval.vlm.rbln import RBLNPPOCRv5Rec
from vlmeval.vlm.rbln.ppocrv5_rec_backend import (BoxCache, ctc_decode, get_rotate_crop_image,
                                                  image_digest, make_record,
                                                  merge_box_caches, paddle_rec_width,
                                                  pick_width_bucket, preprocess_line,
                                                  save_box_cache, sorted_boxes)

# Verbatim prompts from the categories this wrapper serves.
_CCOCR_Q = ('Please output only the text content from the image without any '
            'additional descriptions or formatting.')
_FULLPAGE_Q = ('Read all the text in the image. Directly output the content and '
               'split the texts with space.')
_REGION_Q = ('Recognize the text within the [603, 370, 980, 558] of the image. '
             'The coordinates have been normalized ranging from 0 to 1000 by the '
             'image width and height. ')


# ---------------------------------------------------------------------------
# CTC decoding
# ---------------------------------------------------------------------------

def _probs(sequence, n_classes=5):
    """One-hot probabilities for a sequence of class indices."""
    out = np.zeros((len(sequence), n_classes), dtype=np.float32)
    for t, i in enumerate(sequence):
        out[t, i] = 1.0
    return out


_CHARSET = ['<blank>', 'a', 'b', 'c', ' ']


def test_ctc_collapses_repeats_and_drops_blanks():
    # a a <blank> a b  ->  'aab' would be wrong; CTC gives 'aab' only if the
    # blank separates the repeat, which it does here.
    text, _ = ctc_decode(_probs([1, 1, 0, 1, 2]), _CHARSET)
    assert text == 'aab'


def test_ctc_blank_index_is_zero():
    """Index 0 is blank, so an all-blank sequence must decode to empty. A
    dictionary that forgot the leading blank would shift every character."""
    text, conf = ctc_decode(_probs([0, 0, 0]), _CHARSET)
    assert text == ''
    assert conf == 0.0


def test_ctc_confidence_averages_kept_timesteps_only():
    probs = np.array([[0.0, 0.9, 0.1, 0.0, 0.0],
                      [0.7, 0.3, 0.0, 0.0, 0.0],     # blank — excluded
                      [0.0, 0.0, 0.5, 0.5, 0.0]], dtype=np.float32)
    text, conf = ctc_decode(probs, _CHARSET)
    assert text == 'ab'
    # (0.9 + 0.5) / 2, not a mean over all three timesteps.
    assert conf == pytest.approx(0.7)


def test_ctc_accepts_batch_dimension():
    batched = _probs([1, 2])[None]
    assert ctc_decode(batched, _CHARSET)[0] == 'ab'


def test_ctc_space_is_the_last_class():
    """PP-OCR appends the space character after the dictionary
    (``use_space_char``), so it is the highest index."""
    text, _ = ctc_decode(_probs([1, 4, 2]), _CHARSET)
    assert text == 'a b'


# ---------------------------------------------------------------------------
# Preprocessing — PaddleOCR's RecResizeImg
# ---------------------------------------------------------------------------

def test_preprocess_preserves_aspect_and_pads():
    """A 2:1 crop at height 48 occupies 96 of the 320 columns; the rest is
    padding. Stretching to fill the width would deform the glyphs."""
    img = np.full((50, 100, 3), 200, dtype=np.uint8)
    x = preprocess_line(img, 48, 320)
    assert x.shape == (1, 3, 48, 320)
    filled = np.abs(x[0, :, :, :96] - (200 / 255.0 - 0.5) / 0.5).max()
    assert filled < 0.05
    # Padding is 0 in normalised space (mid-grey), as PaddleOCR does.
    assert np.all(x[0, :, :, 96:] == 0.0)


def test_preprocess_normalises_to_pm_one():
    black = preprocess_line(np.zeros((48, 48, 3), np.uint8), 48, 48)
    white = preprocess_line(np.full((48, 48, 3), 255, np.uint8), 48, 48)
    assert black.min() == pytest.approx(-1.0)
    assert white.max() == pytest.approx(1.0)


def test_preprocess_channel_order_is_bgr():
    """The checkpoint declares ``DecodeImage: img_mode: BGR``, so an ndarray is
    fed through unswapped and ``bgr=False`` mirrors it. This is invisible on
    grey text (R==G==B), which is why it needs an explicit test."""
    arr = np.zeros((48, 48, 3), dtype=np.uint8)
    arr[:, :, 0] = 255                      # channel 0 hot
    keep = preprocess_line(arr, 48, 48, bgr=True)
    swap = preprocess_line(arr, 48, 48, bgr=False)
    assert keep[0, 0].max() == pytest.approx(1.0)
    assert swap[0, 2].max() == pytest.approx(1.0)
    assert swap[0, 0].max() == pytest.approx(-1.0)


def test_preprocess_rejects_empty_crop():
    with pytest.raises(ValueError):
        preprocess_line(np.zeros((0, 10, 3), np.uint8), 48, 320)


# ---------------------------------------------------------------------------
# Width buckets
# ---------------------------------------------------------------------------

def test_paddle_rec_width_floors_at_base():
    # A square crop's aspect gives 48, well under the 320 floor.
    assert paddle_rec_width(48, 48) == 320
    # A 10:1 crop needs 480.
    assert paddle_rec_width(480, 48) == 480


def test_pick_width_bucket_rounds_up():
    """Rounding down would compress the line and drop spaces, so the smallest
    bucket that is at least the required width wins."""
    buckets = [320, 480, 640]
    w, need, fits = pick_width_bucket(400, 48, buckets)
    assert (w, need, fits) == (480, 400, True)


def test_pick_width_bucket_reports_when_nothing_fits():
    w, need, fits = pick_width_bucket(4800, 48, [320, 480])
    assert w == 480 and need == 4800 and fits is False


# ---------------------------------------------------------------------------
# Quadrilateral crop
# ---------------------------------------------------------------------------

def test_crop_warps_quad_to_upright_rect():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:40, 50:150] = 255
    crop = get_rotate_crop_image(img, [[50, 20], [150, 20], [150, 40], [50, 40]])
    assert crop.shape[:2] == (20, 100)
    assert crop.mean() > 250


def test_crop_rotates_tall_boxes():
    """PaddleOCR rotates a crop at least 1.5x taller than wide — vertical text
    would otherwise be squashed into a 48px height and be unreadable."""
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    tall = get_rotate_crop_image(img, [[10, 10], [40, 10], [40, 190], [10, 190]])
    # 30 wide x 180 high -> rotated to 180 x 30.
    assert tall.shape[0] < tall.shape[1]


def test_crop_rejects_degenerate_quad():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        get_rotate_crop_image(img, [[10, 10], [10, 10], [10, 10], [10, 10]])


def test_sorted_boxes_is_reading_order():
    """Top-to-bottom, then left-to-right within a line (y within 10px)."""
    quad = lambda x, y: np.array([[x, y], [x + 10, y], [x + 10, y + 5], [x, y + 5]])  # noqa: E731
    boxes = [quad(100, 5), quad(10, 5), quad(50, 100)]
    order = [(int(b[0][0]), int(b[0][1])) for b in sorted_boxes(boxes)]
    assert order == [(10, 5), (100, 5), (50, 100)]


# ---------------------------------------------------------------------------
# Frozen box cache — the attributability guarantee
# ---------------------------------------------------------------------------

def _write_image(tmp_path, name='img.png', colour=(10, 20, 30)):
    import cv2
    path = tmp_path / name
    arr = np.zeros((40, 60, 3), dtype=np.uint8)
    arr[:] = colour
    cv2.imwrite(str(path), arr)
    return str(path)


def _cache_with(tmp_path, image_path, boxes):
    rec = make_record(image_path, 60, 40, boxes, [0.9] * len(boxes))
    out = str(tmp_path / 'boxes.json')
    save_box_cache(out, {image_digest(image_path): rec}, {'det_device': 'test'})
    return out


def test_box_cache_roundtrips_quads(tmp_path):
    img = _write_image(tmp_path)
    quad = np.array([[1, 2], [11, 2], [11, 8], [1, 8]])
    cache = BoxCache(_cache_with(tmp_path, img, [quad]))
    boxes, scores = cache.get(img)
    assert len(boxes) == 1
    assert boxes[0].shape == (4, 2)
    assert np.array_equal(boxes[0], quad)
    assert scores == [0.9]


def test_box_cache_miss_raises_and_names_the_cause(tmp_path):
    """A miss must NOT silently fall back to live detection: that would
    reintroduce the very confound the cache exists to remove, invisibly."""
    img = _write_image(tmp_path)
    cache = BoxCache(_cache_with(tmp_path, img, []))
    other = _write_image(tmp_path, 'other.png', colour=(200, 100, 50))
    with pytest.raises(KeyError) as err:
        cache.get(other)
    assert 're-encoded' in str(err.value)


def test_box_cache_is_keyed_by_content_not_filename(tmp_path):
    """Re-encoding an image changes the pixels and therefore the crops, so
    identity is the file's bytes. A filename key would silently pair boxes
    with different pixels."""
    import shutil
    img = _write_image(tmp_path)
    cache = BoxCache(_cache_with(tmp_path, img, []))
    copy = str(tmp_path / 'renamed.png')
    shutil.copyfile(img, copy)
    cache.get(copy)                        # same bytes -> hit despite new name

    edited = np.zeros((40, 60, 3), dtype=np.uint8)
    edited[:] = (11, 20, 30)               # one channel off by one
    import cv2
    cv2.imwrite(copy, edited)
    with pytest.raises(KeyError):
        cache.get(copy)


def test_merging_conflicting_caches_raises(tmp_path):
    """Two shards disagreeing on an image's boxes means they were not produced
    by identical detectors; keeping either silently would hide that."""
    img = _write_image(tmp_path)
    a = _cache_with(tmp_path, img, [np.array([[0, 0], [5, 0], [5, 5], [0, 5]])])
    digest = image_digest(img)
    b = str(tmp_path / 'b.json')
    save_box_cache(b, {digest: make_record(
        img, 60, 40, [np.array([[1, 1], [6, 1], [6, 6], [1, 6]])], [0.5])}, {})
    with pytest.raises(ValueError):
        merge_box_caches([a, b], str(tmp_path / 'merged.json'))


def test_cache_records_its_producer(tmp_path):
    """Provenance is the point: a cache whose origin is unknown cannot support
    a parity claim."""
    img = _write_image(tmp_path)
    cache = BoxCache(_cache_with(tmp_path, img, []))
    assert cache.meta['det_device'] == 'test'


# ---------------------------------------------------------------------------
# Wrapper routing / config validation
# ---------------------------------------------------------------------------

class _Route:
    """Unbound stand-in exposing only the routing surface."""

    _parse_region = staticmethod(RBLNPPOCRv5Rec._parse_region)
    _crop_region = RBLNPPOCRv5Rec._crop_region


def test_region_prompt_is_parsed():
    assert _Route._parse_region(_REGION_Q) == (603, 370, 980, 558)


@pytest.mark.parametrize('question', [_CCOCR_Q, _FULLPAGE_Q])
def test_page_prompts_have_no_region(question):
    assert _Route._parse_region(question) is None


@pytest.mark.parametrize('bad', [
    'Recognize the text within the [900, 370, 100, 558] of the image.',   # x1 >= x2
    'Recognize the text within the [0, 0, 1200, 500] of the image.',      # > 1000
])
def test_invalid_regions_are_rejected(bad):
    """A malformed region must fall through to whole-image reading rather than
    produce an empty or inverted crop."""
    assert _Route._parse_region(bad) is None


def test_region_crop_maps_normalised_to_pixels():
    bgr = np.zeros((200, 100, 3), dtype=np.uint8)
    sub = _Route()._crop_region(bgr, (500, 250, 1000, 750))
    assert sub.shape[:2] == (100, 50)


def test_region_crop_is_never_empty():
    """Clamping must keep at least one pixel; a zero-size crop would raise
    deep inside preprocessing instead of reading something."""
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    sub = _Route()._crop_region(bgr, (0, 0, 1, 1))
    assert sub.shape[0] >= 1 and sub.shape[1] >= 1


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        RBLNPPOCRv5Rec(model_path='unused', mode='sideways')


def test_invalid_orientation_rejected():
    with pytest.raises(ValueError):
        RBLNPPOCRv5Rec(model_path='unused', orientation='upside-down')


# ---------------------------------------------------------------------------
# 180-degree line ambiguity
# ---------------------------------------------------------------------------

class _Orient:
    """Exercises _recognize_crop with a stubbed single-pass recogniser.

    ``get_rotate_crop_image`` rotates tall quads 90 degrees counter-clockwise,
    which lands text upside-down on a page photographed the other way round.
    The stub returns a confident reading only for the flipped crop, standing in
    for that situation.
    """

    _recognize_crop = RBLNPPOCRv5Rec._recognize_crop

    def __init__(self, orientation):
        self.orientation = orientation
        self._n_flipped = 0
        self.calls = 0

    def _rec_once(self, crop):
        self.calls += 1
        # The stub's "upright" marker is a bright top-left pixel.
        upright = crop[0, 0, 0] > 128
        return ('correct', 0.95) if upright else ('gibberish', 0.20)


def _upside_down_crop():
    """A crop whose bright marker sits bottom-right — upright after 180 deg."""
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    crop[-1, -1, 0] = 255
    return crop


def test_orientation_none_keeps_the_upside_down_reading():
    """Baseline PaddleOCR behaviour: one pass, no second chance."""
    o = _Orient('none')
    text, conf = o._recognize_crop(_upside_down_crop())
    assert (text, conf) == ('gibberish', 0.20)
    assert o.calls == 1
    assert o._n_flipped == 0


def test_orientation_confidence_recovers_flipped_lines():
    o = _Orient('confidence')
    text, conf = o._recognize_crop(_upside_down_crop())
    assert (text, conf) == ('correct', 0.95)
    assert o.calls == 2                     # exactly one extra pass
    assert o._n_flipped == 1


def test_orientation_confidence_leaves_upright_lines_alone():
    """An already-upright crop must not be flipped — the tie-break is strict."""
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    crop[0, 0, 0] = 255
    o = _Orient('confidence')
    text, _ = o._recognize_crop(crop)
    assert text == 'correct'
    assert o._n_flipped == 0


def test_orientation_uses_only_model_confidence():
    """The tie-break must never consult ground truth — it is inference-time
    logic, so it only sees the two readings and their confidences."""
    import inspect
    src = inspect.getsource(RBLNPPOCRv5Rec._recognize_crop)
    for forbidden in ('answer', 'gt', 'ground_truth', 'label'):
        assert forbidden not in src


def test_line_join_is_newline():
    """Page-level ground truth is newline-separated, and whitespace-tokenising
    metrics need a separator between lines."""
    from vlmeval.vlm.rbln.ppocrv5_rec import _LINE_JOIN
    assert _LINE_JOIN == '\n'


# ---------------------------------------------------------------------------
# The CC-OCR Korean metric: what the output format has to satisfy
# ---------------------------------------------------------------------------

def test_ccocr_korean_metric_is_character_level():
    """For Korean the real evaluator strips whitespace and compares character
    multisets, so line order and spacing are score-neutral — but a *missing*
    line is not. This is why the wrapper joins every kept line rather than
    picking one, and it is what makes the score attributable to recognition.
    """
    from vlmeval.dataset.utils.ccocr_evaluator import evaluator_map_info

    evaluator = evaluator_map_info['multi_lan_ocr']
    gt = {'a.jpg': '안전보건공단 고용노동부'}
    kwargs = dict(dataset='Korean', op='Korean', group='multi_lan_ocr', num=1)

    exact = evaluator({'a.jpg': '안전보건공단 고용노동부'}, gt, **kwargs)[1]['summary']
    reordered = evaluator({'a.jpg': '고용노동부\n안전보건공단'}, gt, **kwargs)[1]['summary']
    assert exact['mirco_f1_score'] == pytest.approx(1.0)
    assert reordered['mirco_f1_score'] == pytest.approx(1.0)

    dropped = evaluator({'a.jpg': '안전보건공단'}, gt, **kwargs)[1]['summary']
    assert dropped['mirco_f1_score'] < 1.0


def test_empty_prediction_scores_zero_without_raising():
    """A page where nothing is detected must still produce a scorable (zero)
    prediction rather than an exception."""
    from vlmeval.dataset.utils.ccocr_evaluator import evaluator_map_info

    evaluator = evaluator_map_info['multi_lan_ocr']
    out = evaluator({'a.jpg': ''}, {'a.jpg': '안전보건공단'},
                    dataset='Korean', op='Korean', group='multi_lan_ocr', num=1)
    assert out[1]['summary']['mirco_f1_score'] == pytest.approx(0.0, abs=1e-6)


def test_prediction_is_plain_text_not_json():
    """CC-OCR compares the raw prediction string, so any wrapping (a dict, a
    coordinate list) would be scored as characters and tank the result."""
    from vlmeval.dataset.utils.ccocr_evaluator import evaluator_map_info

    evaluator = evaluator_map_info['multi_lan_ocr']
    gt = {'a.jpg': '안전보건공단'}
    kwargs = dict(dataset='Korean', op='Korean', group='multi_lan_ocr', num=1)
    plain = evaluator({'a.jpg': '안전보건공단'}, gt, **kwargs)[1]['summary']
    wrapped = evaluator({'a.jpg': json.dumps({'text': '안전보건공단'}, ensure_ascii=False)},
                        gt, **kwargs)[1]['summary']
    assert plain['mirco_f1_score'] > wrapped['mirco_f1_score']
