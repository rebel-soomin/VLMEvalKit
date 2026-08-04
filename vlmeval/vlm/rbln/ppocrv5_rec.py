"""RBLN backend for PP-OCRv5 text recognition, as a det -> rec pipeline."""

from __future__ import annotations

import glob
import os
import re

from .base import RBLNVLMBase

# ``fine-grained text recognition`` prompts name the region to read:
#   "Recognize the text within the [603, 370, 980, 558] of the image. The
#    coordinates have been normalized ranging from 0 to 1000 by the image
#    width and height."
# Four integers in 0..1000 after "within the".
_REGION_RE = re.compile(
    r'within\s+the\s*[\[(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\])]',
    re.IGNORECASE,
)
_REGION_SCALE = 1000

_REC_BUCKET_RE = re.compile(r'rec_(\d+)x(\d+)\.rbln$')

# Lines are joined with a newline: it matches how the ground truth of the
# page-level categories is written, and the metrics that tokenise on
# whitespace treat it as a separator. (Note the CC-OCR Korean metric strips
# whitespace entirely and compares character multisets, so there the join is
# score-neutral.)
_LINE_JOIN = '\n'


class RBLNPPOCRv5Rec(RBLNVLMBase):
    """PP-OCRv5 recognition (``PaddlePaddle/*_PP-OCRv5_mobile_rec``) on RBLN.

    A CNN + SVTR-transformer head with a CTC decoder. The NPU runs the
    forward pass; the host does PaddleOCR's ``RecResizeImg`` preprocessing,
    the quadrilateral crop, and ``CTCLabelDecode``. Model code comes from the
    vendored :mod:`~vlmeval.vlm.rbln.ppocrv5_rec_backend`, which documents
    the two numerical workarounds this model needs.

    **Not an optimum-rbln model.** The artifact is a set of bare
    ``rec_48x<w>.rbln`` files with no ``save_pretrained`` and no
    ``rbln_config.json``, so the optimum-specific hooks inherited from
    :class:`RBLNVLMBase` are disabled (:meth:`_maybe_save_compiled_artifact`,
    :meth:`_check_compiled_max_seq_len`).

    **This model reads a single text line — it does not localise text and it
    does not answer questions.** Benchmark images are usually whole pages or
    photos, so a detector must supply the lines first. That is what
    ``det_model_path`` / ``boxes_cache`` are for; see :meth:`_boxes_for`.
    Recognition confidence below ``drop_score`` is discarded, as PaddleOCR
    does.

    Three input modes, selected by ``mode``:

    * ``'page'`` — detect lines, crop each with a perspective warp, recognise,
      join in reading order. What full-page OCR categories need.
    * ``'line'`` — the image *is* one text line; feed it straight in. Correct
      for word-crop categories, and the only option with no detector
      configured.
    * ``'auto'`` (default) — a region-naming prompt (*fine-grained text
      recognition*) crops that region first and then reads it as a page;
      otherwise ``'page'`` when lines can be supplied, else ``'line'``.

    Width is baked into each artifact, so ``rec_48x<w>.rbln`` files act as
    buckets: PaddleOCR's aspect-derived width is computed per crop and the
    smallest bucket that fits is used
    (:func:`~.ppocrv5_rec_backend.pick_width_bucket`). Too narrow a bucket
    compresses glyphs and drops spaces, so the choice rounds **up**.

    **Reproducibility of the detection stage.** Detection is a sharp cutoff
    (``box_thresh=0.6``), so tiny numerical differences flip boxes in and
    out — which changes how many crops exist and where. Any comparison of
    two recognition runs must therefore hold detection fixed: pass
    ``boxes_cache`` (built by ``scripts/ppocr_boxes_precompute.py``) and both
    runs consume the identical box set, leaving the recognition forward pass
    as the only variable. A cache miss raises rather than falling back to
    live detection, which would silently reintroduce the confound.
    """

    INTERLEAVE = False

    _MODES = ('auto', 'page', 'line')

    #: How to resolve a line crop's 180-degree ambiguity.
    #:
    #: ``get_rotate_crop_image`` rotates a taller-than-wide quad by 90 degrees
    #: counter-clockwise, exactly as PaddleOCR does. For a photographed page
    #: that is *itself* rotated, text lines are vertical in image space, so
    #: every line takes that path — and the rotation lands the text upright
    #: only if the page was turned the one way. Turned the other way, every
    #: crop comes out upside-down and recognition returns confident-looking
    #: gibberish. PaddleOCR resolves this with a separate textline-orientation
    #: classifier (``PP-LCNet_x1_0_textline_ori``), which is not part of this
    #: port.
    #:
    #: ``'confidence'`` substitutes for that classifier without another model:
    #: read the crop both ways and keep the more confident result. It uses only
    #: the model's own output — never the ground truth — and costs one extra
    #: recognition pass per line, which is cheap here (recognition is a ~19MB
    #: graph; the full 150-image run takes 16s across 8 NPUs).
    #:
    #: ``'none'`` is the plain PaddleOCR det+rec behaviour.
    _ORIENTATIONS = ('none', 'confidence')

    def __init__(
        self,
        model_path: str = 'korean_PP-OCRv5_mobile_rec-rbln',
        charset_dir: str | None = None,
        det_model_path: str | None = None,
        boxes_cache: str | None = None,
        mode: str = 'auto',
        orientation: str = 'none',
        drop_score: float = 0.5,
        device: int = 0,
        # Detection postprocess knobs — model-card defaults, mirroring
        # RBLNPPOCRv5Det so a live detector here behaves like that wrapper.
        thresh: float = 0.3,
        box_thresh: float = 0.6,
        unclip_ratio: float = 1.5,
        limit_side_len: int = 64,
        limit_type: str = 'min',
        bgr: bool = True,
        **kwargs,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f'mode must be one of {self._MODES}, got {mode!r}')
        if orientation not in self._ORIENTATIONS:
            raise ValueError(
                f'orientation must be one of {self._ORIENTATIONS}, got {orientation!r}')
        self.mode = mode
        self.orientation = orientation
        self._n_flipped = 0
        self.charset_dir = charset_dir
        self.det_model_path = det_model_path
        self.boxes_cache_path = boxes_cache
        self.drop_score = drop_score
        self.device = device
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.bgr = bgr
        self._runtimes: dict[int, object] = {}
        self._det_runtimes: dict[tuple[int, int], object] = {}
        self._n_classes_checked = False
        super().__init__(model_path=model_path, **kwargs)

    # ------------------------------------------------------------------
    # Loading. No optimum-rbln model and no processor: the artifact is a
    # directory of per-width .rbln files, opened lazily so a run only pays
    # for the buckets it actually uses.
    # ------------------------------------------------------------------

    def _load_rbln_model_and_processor(self):
        from . import ppocrv5_rec_backend as backend

        self._backend = backend
        self.buckets = self._discover_buckets()
        if not self.buckets:
            raise ValueError(
                f'No rec_48x<w>.rbln artifacts found in {self.model_path!r}. '
                'Compile them first with scripts/ppocr_rec_compile.py.'
            )
        self.charset = backend.load_charset(self._resolve_charset_dir())

        self._box_cache = None
        if self.boxes_cache_path:
            self._box_cache = backend.BoxCache(self.boxes_cache_path)
        self._det_backend = None
        self._det_buckets: list[tuple[int, int]] = []
        if self.det_model_path:
            if self._box_cache is not None:
                # Both configured would be ambiguous about which boxes were
                # used, and the whole point of the cache is attributability.
                raise ValueError(
                    'Pass either det_model_path or boxes_cache, not both: with '
                    'a frozen cache the live detector is never consulted, so '
                    'configuring one would misrepresent what produced the boxes.'
                )
            from . import ppocrv5_det_backend as det_backend
            self._det_backend = det_backend
            self._det_buckets = self._discover_det_buckets()
            if not self._det_buckets:
                raise ValueError(
                    f'No det_<h>x<w>.rbln artifacts found in {self.det_model_path!r}.')
        return None, None

    def _resolve_charset_dir(self) -> str:
        """Where to read ``inference.yml`` (the character dictionary) from.

        The compiled artifact carries no dictionary, so it comes from the
        paddle checkpoint. ``charset_dir`` wins; otherwise the artifact
        directory itself is tried, then the conventional sibling with the
        ``-rbln`` suffix dropped.
        """
        candidates = []
        if self.charset_dir:
            candidates.append(self.charset_dir)
        candidates.append(self.model_path)
        base = self.model_path.rstrip('/')
        if base.endswith('-rbln'):
            candidates.append(base[: -len('-rbln')])
        for cand in candidates:
            if cand and os.path.isfile(os.path.join(cand, 'inference.yml')):
                return cand
        raise ValueError(
            'Could not find inference.yml (the character dictionary) in any of '
            f'{candidates}. Pass charset_dir=<paddle checkpoint dir>. The '
            'dictionary is not part of the compiled artifact, and a wrong one '
            'silently maps class indices to the wrong characters.'
        )

    def _discover_buckets(self) -> list[int]:
        """Compiled recognition widths. All artifacts must share one height."""
        widths, heights = [], set()
        for path in glob.glob(os.path.join(self.model_path, 'rec_*x*.rbln')):
            m = _REC_BUCKET_RE.search(os.path.basename(path))
            if m:
                heights.add(int(m.group(1)))
                widths.append(int(m.group(2)))
        if len(heights) > 1:
            raise ValueError(
                f'Mixed recognition heights {sorted(heights)} in {self.model_path!r}; '
                'the height is baked in and must be identical across width buckets.')
        self.rec_height = heights.pop() if heights else self._backend.REC_IMAGE_HEIGHT
        return sorted(widths)

    def _discover_det_buckets(self) -> list[tuple[int, int]]:
        out = []
        for path in glob.glob(os.path.join(self.det_model_path, 'det_*x*.rbln')):
            m = re.search(r'det_(\d+)x(\d+)\.rbln$', os.path.basename(path))
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
        return sorted(out)

    def _runtime(self, width: int):
        if width not in self._runtimes:
            path = os.path.join(self.model_path, f'rec_{self.rec_height}x{width}.rbln')
            self._runtimes[width] = self._backend.RecRuntime(path, device=self.device)
        return self._runtimes[width]

    def _det_runtime(self, bucket: tuple[int, int]):
        if bucket not in self._det_runtimes:
            path = os.path.join(self.det_model_path, f'det_{bucket[0]}x{bucket[1]}.rbln')
            self._det_runtimes[bucket] = self._det_backend.DetRuntime(
                path, device=self.device)
        return self._det_runtimes[bucket]

    # The artifact is not an optimum-rbln model — it has no
    # ``save_pretrained`` and no ``rbln_config.json``, so neither inherited
    # hook applies.
    def _maybe_save_compiled_artifact(self) -> None:
        return

    def _check_compiled_max_seq_len(self) -> None:
        return

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def _recognize_crop(self, crop) -> tuple[str, float]:
        """One line crop (BGR ndarray) -> ``(text, confidence)``.

        With ``orientation='confidence'`` the crop is also read upside-down and
        the more confident reading wins — see :attr:`_ORIENTATIONS`.
        """
        import numpy as np

        text, conf = self._rec_once(crop)
        if self.orientation == 'confidence':
            flipped, fconf = self._rec_once(np.ascontiguousarray(np.rot90(crop, 2)))
            if fconf > conf:
                self._n_flipped += 1
                return flipped, fconf
        return text, conf

    def _rec_once(self, crop) -> tuple[str, float]:
        backend = self._backend
        h, w = crop.shape[:2]
        width, need, fits = backend.pick_width_bucket(
            w, h, self.buckets, self.rec_height)
        if not fits:
            self._warn_once(
                'rec_width',
                f'{type(self).__name__}: crop needs width {need} but the widest '
                f'compiled bucket is {width}; the line is compressed '
                'horizontally, which tends to drop spaces. Compile a wider '
                'bucket to avoid this.',
            )
        x = backend.preprocess_line(crop, self.rec_height, width, bgr=self.bgr)
        probs = self._runtime(width).run(x)
        self._check_n_classes(probs)
        return backend.ctc_decode(probs, self.charset)

    def _check_n_classes(self, probs) -> None:
        """Assert the artifact's class count matches the character dictionary.

        A mismatch means the dictionary belongs to a different language
        variant of the model. The graph is identical across variants, so
        nothing fails — the indices just decode to the wrong characters.
        """
        if self._n_classes_checked:
            return
        self._n_classes_checked = True
        n_out = int(probs.shape[-1])
        if n_out != len(self.charset):
            raise ValueError(
                f'Artifact emits {n_out} classes but the character dictionary in '
                f'{self._resolve_charset_dir()!r} has {len(self.charset)} entries '
                '(blank + dict + space). These are different model variants; the '
                'graph is shared, so decoding would silently produce wrong '
                'characters rather than fail. Point charset_dir at the checkpoint '
                'this artifact was compiled from.'
            )

    # ------------------------------------------------------------------
    # Detection — live, or frozen from a cache
    # ------------------------------------------------------------------

    def _boxes_for(self, image, image_path: str):
        """Return the line quadrilaterals for an image, in reading order."""
        if self._box_cache is not None:
            boxes, _scores = self._box_cache.get(image_path)
        elif self._det_backend is not None:
            boxes, _scores = self._detect(image)
        else:
            return []
        return self._backend.sorted_boxes(boxes) if len(boxes) else []

    def _detect(self, image):
        """Run the live detector. Mirrors ``RBLNPPOCRv5Det._detect``'s setup."""
        det = self._det_backend
        width, height = image.size
        target = det.paddle_resize_shape(
            height, width, self.limit_side_len, self.limit_type)
        bucket, exact = det.pick_bucket(target, self._det_buckets)
        if not exact:
            # Informational, not a defect: a single fixed shape measured
            # better than PaddleOCR-exact resolution on this detector.
            self._warn_once(
                'det_bucket',
                f'{type(self).__name__}: no exact detection bucket for '
                f'{target[1]}x{target[0]}; using {bucket[1]}x{bucket[0]}.',
            )
        x, (orig_h, orig_w) = det.preprocess_image(image, bucket)
        prob = self._det_runtime(bucket).run(x)
        return det.db_postprocess(
            prob, orig_w, orig_h, self.thresh, self.box_thresh, self.unclip_ratio)

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------

    def _read_page(self, bgr, image, image_path: str) -> str:
        """Detect lines, recognise each, join in reading order."""
        backend = self._backend
        boxes = self._boxes_for(image, image_path)
        if not boxes:
            # Nothing detected. Reading the whole image as one line is a
            # genuine fallback (it cannot do worse than an empty answer) and
            # is logged so the page is not mistaken for a clean read.
            self._debug_log('no boxes detected; falling back to line mode', 'yellow')
            return self._read_line(bgr)

        texts = []
        for box in boxes:
            try:
                crop = backend.get_rotate_crop_image(bgr, box)
            except ValueError:
                continue          # degenerate quad — nothing to read
            text, conf = self._recognize_crop(crop)
            if text and conf >= self.drop_score:
                texts.append(text)
        self._debug_log(f'{len(boxes)} boxes -> {len(texts)} lines kept', 'blue')
        return _LINE_JOIN.join(texts)

    def _read_line(self, bgr) -> str:
        text, conf = self._recognize_crop(bgr)
        return text if conf >= self.drop_score else ''

    @staticmethod
    def _parse_region(prompt: str):
        """``(x1, y1, x2, y2)`` in 0..1000 from a region-naming prompt, or None."""
        m = _REGION_RE.search(prompt or '')
        if not m:
            return None
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        if x1 >= x2 or y1 >= y2:
            return None
        if any(v < 0 or v > _REGION_SCALE for v in (x1, y1, x2, y2)):
            return None
        return x1, y1, x2, y2

    def _crop_region(self, bgr, region):
        """Crop a 0..1000-normalised region, clamped to at least one pixel."""
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = region
        px1 = max(0, min(w - 1, int(round(x1 / _REGION_SCALE * w))))
        px2 = max(px1 + 1, min(w, int(round(x2 / _REGION_SCALE * w))))
        py1 = max(0, min(h - 1, int(round(y1 / _REGION_SCALE * h))))
        py2 = max(py1 + 1, min(h, int(round(y2 / _REGION_SCALE * h))))
        return bgr[py1:py2, px1:px2]

    def _can_supply_lines(self) -> bool:
        return self._box_cache is not None or self._det_backend is not None

    def generate_inner(self, message, dataset=None):
        import numpy as np

        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        if image_path is None:
            raise ValueError(f'{type(self).__name__} requires an image input.')

        image = self._load_image(image_path)
        # Recognition was trained on OpenCV-ordered input (the checkpoint's
        # inference.yml declares ``img_mode: BGR``), so crops are taken in BGR.
        bgr = np.asarray(image.convert('RGB'))[:, :, ::-1]
        prompt = prompt or ''

        if self.mode == 'line':
            return self._read_line(bgr)
        if self.mode == 'page':
            return self._read_page(bgr, image, image_path)

        # auto
        region = self._parse_region(prompt)
        if region is not None:
            sub = np.ascontiguousarray(self._crop_region(bgr, region))
            if self._can_supply_lines():
                # The region usually spans several lines, so it still needs
                # segmenting; but the frozen/live boxes belong to the *whole*
                # image, so detect within the crop is not available from a
                # cache. Recognise the region as one line when boxes cannot
                # be recomputed for it.
                if self._det_backend is not None:
                    from PIL import Image
                    sub_img = Image.fromarray(sub[:, :, ::-1])
                    return self._read_page(sub, sub_img, image_path)
                self._warn_once(
                    'region_cache',
                    f'{type(self).__name__}: a frozen box cache covers whole '
                    'images, so a named sub-region cannot be re-segmented; '
                    'reading it as a single line. Use a live det_model_path '
                    'for region-level categories.',
                )
            return self._read_line(sub)

        if self._can_supply_lines():
            return self._read_page(bgr, image, image_path)
        self._warn_once(
            'no_detector',
            f'{type(self).__name__}: no det_model_path or boxes_cache configured, '
            'so the whole image is read as a single text line. That is correct '
            'for word-crop inputs but not for pages.',
        )
        return self._read_line(bgr)
