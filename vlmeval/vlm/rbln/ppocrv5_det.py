from __future__ import annotations
import glob
import json
import os
import re

from .base import RBLNVLMBase

# OCRBench_v2's *VQA with position en* prompt asks for a dict:
#   "Output the answer with 'answer' and 'bbox'. 'bbox' refers to the
#    bounding box position of the 'answer' content in the image."
# and is scored by ``vqa_with_position_evaluation``, which reads
# ``predict["bbox"]`` with ``ast.literal_eval``. *text grounding en* instead
# says "bounding box" and wants a bare ``(x1, y1, x2, y2)``, parsed by
# ``extract_coordinates``. The literal token ``bbox`` is what separates them.
_BBOX_DICT_MARKER = 'bbox'

# OCRBench_v2's *text spotting en* prompt: "Spotting all the text in the image
# with word-level. Output the normalized coordinates ... and the text content
# ... format:[(x1, y1, x2, y2, text content), ...]". Scored by
# ``spotting_evaluation``, which is **end-to-end**: a detection counts only if
# box IoU >= 0.5 AND the transcription matches (script.py:379). A detector has
# no transcription, so this category is a structural 0 — see the class
# docstring. It is still emitted in the correct shape so the 0 is attributable
# to missing recognition rather than to an unparseable prediction.
_SPOTTING_MARKER = 'spotting'

# ``extract_coordinates`` only accepts values in 0..1000 and returns the
# **last** match in the string, so a prediction must contain exactly one box
# in normalized space.
_COORD_SCALE = 1000

_BUCKET_RE = re.compile(r'det_(\d+)x(\d+)\.rbln$')


class RBLNPPOCRv5Det(RBLNVLMBase):
    """RBLN backend for PP-OCRv5 text detection (``PaddlePaddle/PP-OCRv5_server_det``).

    A DBNet-family detector: the NPU runs the forward pass, the host does
    PaddleOCR's ``DetResizeForTest`` preprocessing and ``DBPostProcess``
    (binarize -> contours -> min-area rect -> unclip). Model classes come
    from the vendored :mod:`~vlmeval.vlm.rbln.ppocrv5_det_backend`.

    **Not an optimum-rbln model.** PP-OCRv5 ships as a PaddlePaddle
    inference model compiled through ONNX with ``rebel`` directly, so the
    artifact is a set of bare ``det_<h>x<w>.rbln`` files with no
    ``save_pretrained`` and no ``rbln_config.json``. The optimum-specific
    hooks inherited from :class:`RBLNVLMBase` are therefore disabled
    (:meth:`_maybe_save_compiled_artifact`, :meth:`_check_compiled_max_seq_len`).

    **This is a detector, not a VLM: it localises text but cannot read it
    and cannot answer questions.** The question is used only to choose an
    output *format* — never interpreted:

    * question mentions ``bbox`` (*VQA with position en*) -> emit
      ``{"bbox": "[x1, y1, x2, y2]"}``. The ``answer`` key is deliberately
      omitted rather than faked: ``vqa_with_position_evaluation`` scores
      ``0.5 * content + 0.5 * IoU`` and skips a missing key, so the box
      half is earned honestly and the content half scores 0.
    * otherwise (*text grounding en*) -> emit a single ``(x1, y1, x2, y2)``
      in 0-1000 normalized coordinates.

    ``box_select`` decides *which* detected box is emitted. This matters
    because both categories are **query-conditioned** — grounding asks
    where one specific string is, and a detector cannot tell its boxes
    apart without recognition. ``'top_score'`` (default) takes the highest
    DB confidence; ``'largest'`` takes the largest area. Neither reads the
    query, so the resulting score is a detector-only baseline, not what a
    det+rec pipeline would achieve.

    Resolution is baked into each artifact, so the compiled
    ``det_<h>x<w>.rbln`` files act as buckets: the PaddleOCR-mandated
    resolution is computed per image and the closest bucket is used (see
    :func:`~.ppocrv5_det_backend.runtime.pick_bucket`). Put a **single**
    artifact in the directory and every image is resized to that one shape.

    **Reproducing PaddleOCR's resolution is not the accuracy-optimal
    choice, and one fixed shape beat it here.** Measured over OCRBench_v2's
    500 detection items::

        single 768x1024   14.48   <- best, one 83MB artifact
        single 960x960    13.40
        single 1024x1024  12.95
        12 exact buckets  12.86   <- "PaddleOCR-exact", 935MB
        single 1024x768   11.47
        single 1920x1088  10.44

    The reason is that ``limit_type='min', limit_side_len=64`` is a *floor*,
    not a target: every real image already clears a 64px min side, so
    ``ratio == 1.0`` and the "PaddleOCR resolution" is just native size
    rounded to a multiple of 32. Matching it exactly therefore means running
    small images (320x512, 416x640) at their native size, starving the
    detector of pixels for small text. A fixed larger shape upsamples them,
    and that outweighs the aspect distortion. Note the effect is driven by
    matching the corpus's dominant *aspect*, not raw pixels — 1024x768 (the
    winner transposed) is near the bottom, and the two largest shapes are
    worst of all.

    So pick the shape empirically for the corpus. ``det_960x960`` is a
    reasonable default for mixed aspect ratios.
    """

    INTERLEAVE = False

    _BOX_SELECT = ('top_score', 'largest')

    def __init__(
        self,
        model_path: str = 'PP-OCRv5_server_det-rbln',
        # Model-card defaults.
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        limit_side_len: int = 64,
        limit_type: str = 'min',
        box_select: str = 'top_score',
        device: int = 0,
        **kwargs,
    ) -> None:
        if box_select not in self._BOX_SELECT:
            raise ValueError(
                f'box_select must be one of {self._BOX_SELECT}, got {box_select!r}')
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.box_select = box_select
        self.device = device
        self._runtimes: dict[tuple[int, int], object] = {}
        super().__init__(model_path=model_path, **kwargs)

    # ------------------------------------------------------------------
    # Loading. There is no optimum-rbln model and no processor here: the
    # artifact is a directory of per-resolution .rbln files, opened lazily
    # so a run only pays for the buckets it actually uses.
    # ------------------------------------------------------------------

    def _load_rbln_model_and_processor(self):
        from . import ppocrv5_det_backend as backend

        self._backend = backend
        self.buckets = self._discover_buckets()
        if not self.buckets:
            raise ValueError(
                f'No det_<h>x<w>.rbln artifacts found in {self.model_path!r}. '
                'Compile them first (see ppocrv5_det_backend.compile_backend).'
            )
        return None, None

    def _discover_buckets(self) -> list[tuple[int, int]]:
        out = []
        for path in glob.glob(os.path.join(self.model_path, 'det_*x*.rbln')):
            m = _BUCKET_RE.search(os.path.basename(path))
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
        return sorted(out)

    def _runtime(self, bucket: tuple[int, int]):
        if bucket not in self._runtimes:
            path = os.path.join(self.model_path, f'det_{bucket[0]}x{bucket[1]}.rbln')
            self._runtimes[bucket] = self._backend.DetRuntime(path, device=self.device)
        return self._runtimes[bucket]

    # The artifact is not an optimum-rbln model — it has no
    # ``save_pretrained`` and no ``rbln_config.json``, so neither inherited
    # hook applies.
    def _maybe_save_compiled_artifact(self) -> None:
        return

    def _check_compiled_max_seq_len(self) -> None:
        return

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self, image) -> list[tuple[tuple[int, int, int, int], float]]:
        """Return ``[((x1,y1,x2,y2), score), ...]`` in 0-1000 normalized
        coordinates, ordered by descending detection confidence."""
        backend = self._backend
        width, height = image.size
        target = backend.paddle_resize_shape(
            height, width, self.limit_side_len, self.limit_type)
        bucket, exact = backend.pick_bucket(target, self.buckets)
        if not exact:
            # Informational, NOT a defect: a fixed shape can beat the
            # PaddleOCR-exact one. See the class docstring — measured on
            # OCRBench_v2 detection, a single 768x1024 artifact scored 14.48
            # against 12.86 for exact-resolution buckets.
            self._warn_once(
                'bucket',
                f'{type(self).__name__}: no exact resolution bucket for '
                f'{target[1]}x{target[0]}; using {bucket[1]}x{bucket[0]}. '
                'This deviates from PaddleOCR, which runs at native '
                'resolution — often for the better on small images.',
            )

        x, (orig_h, orig_w) = backend.preprocess_image(image, bucket)
        prob = self._runtime(bucket).run(x)
        boxes, scores = backend.db_postprocess(
            prob, orig_w, orig_h, self.thresh, self.box_thresh, self.unclip_ratio)
        rects = backend.boxes_to_rects(boxes)

        out = []
        for (x1, y1, x2, y2), score in zip(rects, scores):
            norm = (
                int(round(x1 / orig_w * _COORD_SCALE)),
                int(round(y1 / orig_h * _COORD_SCALE)),
                int(round(x2 / orig_w * _COORD_SCALE)),
                int(round(y2 / orig_h * _COORD_SCALE)),
            )
            # extract_coordinates rejects anything outside 0..1000, and a
            # degenerate box scores 0 IoU anyway.
            if norm[0] >= norm[2] or norm[1] >= norm[3]:
                continue
            if any(v < 0 or v > _COORD_SCALE for v in norm):
                continue
            out.append((norm, score))
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    def _pick_box(self, detections):
        if not detections:
            return None
        if self.box_select == 'largest':
            return max(
                detections,
                key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]),
            )[0]
        return detections[0][0]  # top_score — already sorted

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _wants_bbox_dict(prompt: str) -> bool:
        return _BBOX_DICT_MARKER in prompt.lower()

    @staticmethod
    def _wants_spotting(prompt: str) -> bool:
        return _SPOTTING_MARKER in prompt.lower()

    @staticmethod
    def _format_spotting(detections) -> str:
        """``[(x1, y1, x2, y2, text), ...]`` for **every** detected box.

        The transcription is empty: this model cannot read. Emitting all
        boxes (rather than one) is what the category asks for, and keeps the
        resulting 0 attributable to the missing text rather than to a
        malformed prediction — verified against ``spotting_evaluation``,
        which returns 0 for empty transcriptions and 1.0 for correct ones on
        the same boxes.
        """
        if not detections:
            return ''
        items = ', '.join(f"({b[0]}, {b[1]}, {b[2]}, {b[3]}, '')"
                          for b, _ in detections)
        return f'[{items}]'

    def _format(self, prompt: str, box) -> str:
        if box is None:
            # No detection. Emitting nothing is the honest answer; both
            # scorers award 0 for an unparseable prediction.
            return ''
        x1, y1, x2, y2 = box
        if self._wants_bbox_dict(prompt):
            # 'answer' intentionally absent — see the class docstring.
            return json.dumps({'bbox': f'[{x1}, {y1}, {x2}, {y2}]'})
        return f'({x1}, {y1}, {x2}, {y2})'

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        if image_path is None:
            raise ValueError(f'{type(self).__name__} requires an image input.')

        image = self._load_image(image_path)
        detections = self._detect(image)
        prompt = prompt or ''

        # Spotting wants every box; the other categories want exactly one
        # (their scorers read a single box, and grounding takes the *last*
        # match in the string).
        if self._wants_spotting(prompt):
            self._debug_log(f'{len(detections)} boxes, spotting format', 'blue')
            return self._format_spotting(detections)

        box = self._pick_box(detections)
        self._debug_log(
            f'{len(detections)} boxes, picked {box} ({self.box_select})', 'blue')
        return self._format(prompt, box)
