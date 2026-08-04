"""US-008 — PP-OCRv5 detection output formatting.

``RBLNPPOCRv5Det`` is a detector, not a VLM: it cannot read text or answer
questions, so the only thing the question influences is the *output
format*. OCRBench_v2's two detection categories parse predictions
differently, and getting either format wrong silently scores 0 — so these
tests pin the formats against the real scorers
(``extract_coordinates`` / ``calculate_iou`` for *text grounding en*,
``convert_str_to_dict`` / ``vqa_with_position_evaluation`` for
*VQA with position en*).

No NPU, no artifact, no model load: the formatting surface is exercised on
an unbound instance.
"""

from __future__ import annotations

import pytest

from vlmeval.vlm.rbln import RBLNPPOCRv5Det

# Verbatim OCRBench_v2 prompt tails.
_GROUNDING_Q = (
    "Where is the region of the text 'SELINCOLN'? Output the normalized "
    'coordinates of the left-top and right-bottom corners of the bounding box. '
    'The coordinates should be normalized ranging from 0 to 1000 by the image '
    'width and height.Your answer should be in the following '
    'format:(x1, y1, x2, y2)'
)
_SPOTTING_Q = (
    'Spotting all the text in the image with word-level. Output the normalized '
    'coordinates of the left-top and right-bottom corners of the bounding box '
    'and the text content. The coordinates should be normalized ranging from 0 '
    'to 1000 by the image width and height.Your answer should be in the '
    'following format:[(x1, y1, x2, y2, text content), '
    '(x1, y1, x2, y2, text content)...]'
)
_VQA_POS_Q = (
    "what is written on the cabin? Output the answer with 'answer' and 'bbox'. "
    "'bbox' refers to the bounding box position of the 'answer' content in the "
    'image. The output format is "answer:gt, bbox:(x1,y1,x2,y2)"'
)


class _Fmt:
    """Unbound stand-in exposing only the formatting surface."""

    _wants_bbox_dict = staticmethod(RBLNPPOCRv5Det._wants_bbox_dict)
    _wants_spotting = staticmethod(RBLNPPOCRv5Det._wants_spotting)
    _format = RBLNPPOCRv5Det._format
    _format_spotting = staticmethod(RBLNPPOCRv5Det._format_spotting)


def test_grounding_question_gets_bare_tuple():
    out = _Fmt()._format(_GROUNDING_Q, (120, 340, 455, 600))
    assert out == '(120, 340, 455, 600)'


def test_vqa_position_question_gets_bbox_dict():
    out = _Fmt()._format(_VQA_POS_Q, (120, 340, 455, 600))
    assert out == '{"bbox": "[120, 340, 455, 600]"}'


def test_formats_are_distinct():
    """The two categories must not collapse onto one format — the grounding
    scorer cannot read a dict and vice versa."""
    box = (10, 20, 30, 40)
    assert _Fmt()._format(_GROUNDING_Q, box) != _Fmt()._format(_VQA_POS_Q, box)


def test_grounding_output_parses_in_real_scorer():
    from vlmeval.dataset.utils.Ocrbench_v2.IoUscore_metric import (
        calculate_iou, extract_coordinates)

    out = _Fmt()._format(_GROUNDING_Q, (120, 340, 455, 600))
    parsed = extract_coordinates(out)
    assert parsed == [120, 340, 455, 600]
    assert calculate_iou(parsed, [100, 300, 460, 610]) > 0.7


def test_vqa_position_output_parses_in_real_scorer():
    from vlmeval.dataset.utils.Ocrbench_v2.IoUscore_metric import \
        vqa_with_position_evaluation
    from vlmeval.dataset.utils.Ocrbench_v2.TEDS_metric import convert_str_to_dict

    out = _Fmt()._format(_VQA_POS_Q, (120, 340, 455, 600))
    parsed = convert_str_to_dict(out)
    assert parsed['bbox'] == '[120, 340, 455, 600]'
    score = vqa_with_position_evaluation(
        parsed, {'answers': ['anything'], 'bbox': [100, 300, 460, 610]})
    # bbox half only: 'answer' is deliberately omitted, so content scores 0
    # and the total cannot exceed 0.5.
    assert 0 < score <= 0.5


def test_answer_key_is_not_faked():
    """The detector cannot read text. Emitting an 'answer' key would claim a
    content answer it never produced — the key must be absent."""
    import json

    parsed = json.loads(_Fmt()._format(_VQA_POS_Q, (1, 2, 3, 4)))
    assert 'answer' not in parsed


@pytest.mark.parametrize('question', [_GROUNDING_Q, _VQA_POS_Q])
def test_no_detection_yields_empty_string(question):
    """No boxes -> empty prediction. Both scorers must return 0 rather than
    raising (an exception would abort the whole eval run)."""
    from vlmeval.dataset.utils.Ocrbench_v2.IoUscore_metric import (
        extract_coordinates, vqa_with_position_evaluation)
    from vlmeval.dataset.utils.Ocrbench_v2.TEDS_metric import convert_str_to_dict

    out = _Fmt()._format(question, None)
    assert out == ''
    assert extract_coordinates(out) is None
    assert vqa_with_position_evaluation(
        convert_str_to_dict(out), {'answers': ['x'], 'bbox': [1, 2, 3, 4]}) == 0


def test_box_select_validated():
    with pytest.raises(ValueError):
        RBLNPPOCRv5Det(model_path='dummy', box_select='nonsense')


# ----------------------------------------------------------------------
# Box selection
# ----------------------------------------------------------------------

class _Pick:
    def __init__(self, box_select):
        self.box_select = box_select

    _pick_box = RBLNPPOCRv5Det._pick_box


_DETECTIONS = [
    ((100, 100, 120, 120), 0.95),   # highest score, tiny
    ((0, 0, 900, 900), 0.70),       # largest area, lower score
]


def test_top_score_selection():
    assert _Pick('top_score')._pick_box(_DETECTIONS) == (100, 100, 120, 120)


def test_largest_selection():
    assert _Pick('largest')._pick_box(_DETECTIONS) == (0, 0, 900, 900)


def test_pick_box_handles_no_detections():
    assert _Pick('top_score')._pick_box([]) is None
    assert _Pick('largest')._pick_box([]) is None


# ----------------------------------------------------------------------
# text spotting en — all boxes, empty transcription
# ----------------------------------------------------------------------

_DETS_2 = [((10, 20, 30, 40), 0.9), ((50, 60, 70, 80), 0.8)]


def test_spotting_routing_is_exclusive():
    """The three categories must route to three different formats. Spotting is
    checked first because its prompt also mentions a bounding box."""
    f = _Fmt()
    assert f._wants_spotting(_SPOTTING_Q)
    assert not f._wants_spotting(_GROUNDING_Q)
    assert not f._wants_spotting(_VQA_POS_Q)


def test_spotting_emits_every_box():
    """Spotting asks for all text, unlike the single-box categories."""
    out = _Fmt()._format_spotting(_DETS_2)
    assert out == "[(10, 20, 30, 40, ''), (50, 60, 70, 80, '')]"


def test_spotting_no_detection_yields_empty_string():
    assert _Fmt()._format_spotting([]) == ''


def test_spotting_output_parses_in_real_scorer():
    """The prediction must be well-formed so the (structural) 0 is
    attributable to the missing transcription, not to a parse failure."""
    from vlmeval.dataset.utils.Ocrbench_v2.spotting_metric import \
        extract_bounding_boxes_robust

    parsed = extract_bounding_boxes_robust(_Fmt()._format_spotting(_DETS_2))
    assert parsed == [[10, 20, 30, 40, ''], [50, 60, 70, 80, '']]


def test_spotting_is_end_to_end_so_a_detector_cannot_score():
    """Locks the reason en_text_spotting is 0 for this model: the scorer
    requires the transcription to match, not just the box (script.py:379).
    Same boxes + correct text scores 1.0; empty text scores 0.
    """
    from vlmeval.dataset.utils.Ocrbench_v2.spotting_metric import (
        extract_bounding_boxes_robust, spotting_evaluation)

    meta = {'bbox': [[100, 100, 200, 100, 200, 140, 100, 140]],
            'content': ['HELLO']}
    boxes_only = extract_bounding_boxes_robust("[(100, 100, 200, 140, '')]")
    with_text = extract_bounding_boxes_robust("[(100, 100, 200, 140, 'HELLO')]")
    assert spotting_evaluation(boxes_only, meta) == 0
    assert spotting_evaluation(with_text, meta) == 1.0
