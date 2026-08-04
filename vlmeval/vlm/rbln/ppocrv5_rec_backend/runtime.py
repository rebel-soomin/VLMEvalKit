"""Recognition preprocessing / NPU execution / CTC decoding / line cropping.

Everything here runs on the **host** except :class:`RecRuntime`; the NPU only
does the forward pass. All of it is shared by the NPU wrapper and its GPU
twin, so a parity comparison isolates the forward pass alone.

**Why width buckets.** PaddleOCR's ``TextRecognizer`` picks the input width
*per image* from the aspect ratio (``imgW = 48 * max(w/h, 320/48)``), but RBLN
bakes the shape into the artifact. So several widths are compiled ahead of
time and the smallest one that fits is chosen at run time
(:func:`pick_width_bucket`). Using a width that is too small compresses the
glyphs horizontally and **spaces disappear** (measured: a 482x29 crop read
``'리에씨가자기는...'`` at width 320 and ``'리에 씨가 자기는...'`` at width 800).

**Channel order is BGR, not RGB.** The checkpoint's ``inference.yml`` declares
``DecodeImage: img_mode: BGR``, i.e. the model was trained on OpenCV-ordered
input. Feeding RGB swaps the R and B planes. Note this is invisible on
black-on-white text (R==G==B there), which is why synthetic validation images
cannot detect it — but coloured scene text is affected. Recognition's
normalisation is ``(x/255 - 0.5) / 0.5`` for every channel, so channel order
changes *which plane each filter sees*, not the scaling.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

# PP-OCR recognition input geometry. 48 is the trained height; 320 is the
# *minimum* width (a floor on the aspect-derived width, not a target).
REC_IMAGE_HEIGHT = 48
REC_BASE_WIDTH = 320

# PaddleOCR pads the canvas with 0 **after** normalisation, i.e. mid-grey
# (pixel 127.5), not white. ``predict_rec.py:resize_norm_img`` builds
# ``np.zeros`` and writes the resized image into its left edge.
PAD_VALUE = 0.0


def load_charset(model_dir: str) -> list[str]:
    """``inference.yml``'s ``character_dict`` -> the CTC label list.

    PP-OCR's ``CTCLabelDecode`` convention: index 0 is blank, 1..N the
    dictionary, and a trailing space (``use_space_char``). The YAML quotes
    characters that would otherwise be special (``- '5'``), so quotes are
    stripped.
    """
    text = Path(model_dir, 'inference.yml').read_text(encoding='utf-8')
    m = re.search(r'character_dict:\s*\n((?:\s*-.*\n)+)', text)
    if m is None:
        raise RuntimeError(f'no character_dict found in {model_dir}/inference.yml')
    chars = []
    for line in m.group(1).splitlines():
        v = line.strip()[2:]
        if len(v) >= 2 and v[0] == v[-1] and v[0] in '\'"':
            v = v[1:-1]
        chars.append(v)
    return ['<blank>'] + chars + [' ']


def paddle_rec_width(img_w: int, img_h: int, height: int = REC_IMAGE_HEIGHT,
                     base_width: int = REC_BASE_WIDTH) -> int:
    """The input width PaddleOCR would use for a crop of this aspect ratio.

    ``imgW = height * max(w/h, base_width/height)`` — so ``base_width`` is a
    floor and wide crops get proportionally more width.
    """
    return int(height * max(img_w / img_h, base_width / height))


def pick_width_bucket(img_w: int, img_h: int, buckets, height: int = REC_IMAGE_HEIGHT,
                      base_width: int = REC_BASE_WIDTH):
    """Choose the **smallest compiled bucket >= the required width**.

    Returns ``(bucket_width, required_width, fits)``. Rounding up matters:
    a narrower bucket squeezes the glyphs and costs accuracy. When even the
    widest bucket is too narrow it is used anyway and ``fits`` is False, so
    the caller can report the compression instead of hiding it.
    """
    need = paddle_rec_width(img_w, img_h, height, base_width)
    fits = [w for w in sorted(buckets) if w >= need]
    if fits:
        return fits[0], need, True
    return max(buckets), need, False


def preprocess_line(img, height: int = REC_IMAGE_HEIGHT, width: int = REC_BASE_WIDTH,
                    bgr: bool = True, pad_value: float = PAD_VALUE) -> np.ndarray:
    """A text-line image -> ``[1, 3, height, width]`` float32 NCHW.

    Reproduces PaddleOCR ``resize_norm_img``: keep the aspect ratio, scale to
    ``height``, then pad the remaining width. Distorting the crop to fill the
    width instead would deform the glyphs and collapse accuracy outright.

    ``img`` may be a PIL image or an ndarray. ndarrays are assumed **BGR**
    (they come from :func:`get_rotate_crop_image`, which works in OpenCV
    order); PIL images are RGB and converted. Set ``bgr=False`` to feed RGB
    instead — only useful for measuring the effect of the channel order.

    ``width`` must be a compiled bucket width — see :func:`pick_width_bucket`.
    """
    import cv2

    arr = _as_bgr_array(img)
    if not bgr:
        arr = arr[:, :, ::-1]

    h, w = arr.shape[:2]
    if h < 1 or w < 1:
        raise ValueError(f'empty crop: shape={arr.shape}')

    # PaddleOCR uses math.ceil here and clamps to the canvas width.
    resized_w = min(width, max(1, int(math.ceil(height * (w / h)))))
    resized = cv2.resize(arr, (resized_w, height), interpolation=cv2.INTER_LINEAR)

    norm = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    norm -= 0.5
    norm /= 0.5

    canvas = np.full((3, height, width), pad_value, dtype=np.float32)
    canvas[:, :, :resized_w] = norm
    return np.ascontiguousarray(canvas[None])


def _as_bgr_array(img) -> np.ndarray:
    """PIL image or ndarray -> 3-channel **BGR** uint8 ndarray."""
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
    # PIL is RGB; the model wants BGR.
    arr = np.asarray(img.convert('RGB'))
    return arr[:, :, ::-1]


def ctc_decode(probs, charset, remove_duplicate: bool = True):
    """``[T, C]`` (or ``[1, T, C]``) probabilities -> ``(text, mean_confidence)``.

    PP-OCR ``CTCLabelDecode``: collapse runs of the same index, drop blanks
    (index 0), and average the confidence over the **kept** timesteps only.
    """
    probs = np.asarray(probs)
    while probs.ndim > 2:
        probs = probs[0]
    idx = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)

    keep = np.ones(len(idx), dtype=bool)
    if remove_duplicate:
        keep[1:] = idx[1:] != idx[:-1]
    keep &= idx != 0                     # 0 is blank

    chars = [charset[i] if i < len(charset) else '' for i in idx[keep]]
    confs = conf[keep]
    return ''.join(chars), (float(confs.mean()) if len(confs) else 0.0)


# ---------------------------------------------------------------------------
# Line cropping — the det -> rec hand-off
# ---------------------------------------------------------------------------

def get_rotate_crop_image(img, points) -> np.ndarray:
    """Crop a quadrilateral into an upright line image (PaddleOCR-faithful).

    Detection returns *quadrilaterals*, not axis-aligned rectangles, so a
    plain bounding-box crop of rotated text drags in neighbouring glyphs and
    background. PaddleOCR perspective-warps the quad to a rectangle, and
    rotates 90 degrees when the result is at least 1.5x taller than wide
    (vertical text, common on Korean signage).

    ``img`` is an ndarray in whatever channel order the caller uses; the
    order is preserved.
    """
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    crop_w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    crop_h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if crop_w < 1 or crop_h < 1:
        raise ValueError(f'degenerate box: {points!r}')

    std = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    matrix = cv2.getPerspectiveTransform(pts, std)
    dst = cv2.warpPerspective(
        img, matrix, (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return np.ascontiguousarray(dst)


def sorted_boxes(boxes):
    """Sort detected quads into reading order (PaddleOCR ``sorted_boxes``).

    Primary sort is top-then-left; the adjacent-swap pass then fixes pairs on
    the same visual line whose y differs by under 10px but which came out
    right-before-left.
    """
    ordered = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    ordered = list(ordered)
    for i in range(len(ordered) - 1):
        for j in range(i, -1, -1):
            if (abs(ordered[j + 1][0][1] - ordered[j][0][1]) < 10
                    and ordered[j + 1][0][0] < ordered[j][0][0]):
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            else:
                break
    return ordered


class RecRuntime:
    """Runs a compiled recognition artifact on the NPU."""

    def __init__(self, rbln_path: str, device: int = 0):
        import rebel
        self.rt = rebel.Runtime(rbln_path, device=device)

    def run(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.rt.run(np.ascontiguousarray(x, dtype=np.float32)))


def make_ort_session(onnx_path, providers=('CPUExecutionProvider',)):
    """onnxruntime session with an **explicit thread count**.

    Where CPUs are restricted by cgroup/taskset (containers), onnxruntime's
    default is one thread per *system* core, each pinned to a specific core.
    Pinning to a disallowed core fails and floods the log::

        [E:onnxruntime:Default, env.cc:226 ThreadMain] pthread_setaffinity_np
        failed for thread: N, index: M, mask: {..}, error code: 22 ...
        Specify the number of threads explicitly so the affinity is not set.

    Results are unaffected (measured max|delta|=0 against the default), but
    threads pile onto fewer cores and real errors get buried.
    ``OMP_NUM_THREADS`` does not help — this is onnxruntime's own pool.
    """
    import os

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = len(os.sched_getaffinity(0))   # actually-allowed cores
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=list(providers))
