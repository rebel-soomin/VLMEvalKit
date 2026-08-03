from __future__ import annotations
import re

from .base import RBLNVLMBase

# Question-text markers that select GOT's "OCR with format" query
# (markdown / HTML / LaTeX output) instead of plain "OCR".
#
# Routing is driven by the *question text* — the only thing the model is
# actually given — not by the dataset's ground-truth ``category`` column.
# These cover the OCRBench_v2 parsing families:
#   document parsing en/cn  -> "markdown format" / "Parse the document image in Markdown format"
#   table parsing en/cn     -> "HTML-format" / "markdown-style"
#   formula recognition en  -> "What is the Latex tag for ..."
#   formula recognition cn  -> "将图中的数学公式转换为LaTex表达式"
_FORMAT_MARKERS = (
    'markdown',
    'html',
    'latex',
    'tex表达式',
    'formula',
    '公式',
)

# ``Recognize the text within the [423, 617, 524, 700] of the image. The
# coordinates have been normalized ranging from 0 to 1000 ...``
#
# Four bare integers in brackets. The text-grounding / text-spotting
# prompts also talk about normalized coordinates but only ever contain the
# literal placeholders ``(x1, y1, x2, y2)``, so requiring digits keeps them
# out of the fine-grained path.
_BOX_RE = re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')

# GOT-OCR2.0 terminates its answer with ``<|im_end|>``, but the shipped
# generation_config.json carries no eos_token_id, so generate falls back to
# the tokenizer's ``<|endoftext|>`` (151643) and never stops on the token
# the model actually emits. Measured on the first OCRBench_v2 items: the
# transcription finishes after ~45 tokens, then the model rambles all the
# way to max_new_tokens ("4 4 4 4 ...") — 2048 tokens and 12.6s instead of
# 45 tokens and 1.2s, with the garbage landing in the prediction.
#
# stepfun-ai/GOT-OCR-2.0-hf's model card fixes this with
# ``stop_strings="<|im_end|>"`` + ``tokenizer=...``. We add the token id to
# ``eos_token_id`` instead: byte-identical output on the samples checked,
# without StopStringCriteria's per-step string matching (~0.3s/sample).
_GOT_EOS_TOKEN = '<|im_end|>'


class RBLNGotOcr2(RBLNVLMBase):
    """optimum-rbln backend for GOT-OCR2.0 (``stepfun-ai/GOT-OCR-2.0-hf``).

    optimum-rbln ships no GOT-OCR2.0 support, so the model classes come
    from the vendored :mod:`~vlmeval.vlm.rbln.got_ocr2_backend` package,
    which registers itself into optimum-rbln's in-memory registries on
    import. All three stages run on the NPU: vision tower (SAM/ViTDet) +
    multi_modal_projector + Qwen2 language model.

    **GOT-OCR2.0 is an OCR model, not a VQA model.** Its processor does
    not accept a free-form question: when ``text=None`` it synthesises one
    of a small set of fixed queries (``"OCR: "``, ``"OCR with format: "``,
    ``"[box] OCR: "``, ``"[color] OCR: "``) and the model was trained on
    those only. A dataset question is therefore used to *select a query
    mode*, never passed through verbatim:

    * ``format`` — question asks for markdown / HTML / LaTeX (see
      :data:`_FORMAT_MARKERS`) -> ``"OCR with format: "``.
    * ``box`` — question carries an explicit ``[x1, y1, x2, y2]`` region
      (OCRBench_v2's *fine-grained text recognition*) -> ``"[box] OCR: "``
      restricted to that region.
    * ``plain`` — everything else -> ``"OCR: "``.

    The two axes compose: a region-scoped parsing question yields
    ``"[box] OCR with format: "``.

    Consequence for mixed benchmarks such as OCRBench_v2: the OCR-shaped
    categories (full-page OCR, text recognition, fine-grained recognition,
    document/table/formula parsing) are answered on the model's own terms,
    while the VQA / grounding / counting categories receive transcribed
    page text rather than an answer and score accordingly. That is a model
    capability limit, not a wrapper bug.

    ``ocr_mode`` pins the query mode for the whole run (``'auto'`` — the
    default — applies the routing above; ``'plain'`` / ``'format'`` force
    one query).

    ``crop_to_patches`` is intentionally *not* exposed: the vision tower is
    compiled for exactly one 1024x1024 input (``vision_tower.batch_size=1``,
    ``image_size=1024``), so multi-patch inputs cannot be executed by the
    compiled artifact.
    """

    INTERLEAVE = False
    # model.generate returns the full sequence including the prompt tokens.
    _DECODE_TRIM = True
    _DECODE_STRIP = True

    _OCR_MODES = ('auto', 'plain', 'format')

    def __init__(
        self,
        model_path: str = 'stepfun-ai/GOT-OCR-2.0-hf',
        ocr_mode: str = 'auto',
        # The reference compile bakes in max_seq_len=4096 and a 1024x1024
        # image is 256 image tokens, so ~3.7k tokens are available for the
        # answer. 2048 covers dense full-page documents while keeping the
        # decode cost of a 10k-sample benchmark bounded.
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        **kwargs,
    ) -> None:
        if ocr_mode not in self._OCR_MODES:
            raise ValueError(
                f'ocr_mode must be one of {self._OCR_MODES}, got {ocr_mode!r}')
        self.ocr_mode = ocr_mode
        super().__init__(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
        self._install_got_eos()

    def _install_got_eos(self) -> None:
        """Add GOT's real end-of-answer token to ``eos_token_id``.

        See :data:`_GOT_EOS_TOKEN`. Keeps the tokenizer's own eos as a
        secondary stop, and no-ops if the checkpoint's tokenizer lacks the
        token (nothing to add, and the runaway-generation warning is worth
        surfacing).
        """
        tokenizer = getattr(self.processor, 'tokenizer', None)
        if tokenizer is None:
            return
        got_eos = tokenizer.convert_tokens_to_ids(_GOT_EOS_TOKEN)
        if got_eos is None or got_eos == tokenizer.unk_token_id:
            self._warn_once(
                'got_eos',
                f'{type(self).__name__}: tokenizer has no {_GOT_EOS_TOKEN} token; '
                'generation will run to max_new_tokens and append garbage to '
                'every prediction.',
            )
            return
        eos_ids = [got_eos]
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != got_eos:
            eos_ids.append(tokenizer.eos_token_id)
        self.generate_kwargs['eos_token_id'] = eos_ids

    def _load_rbln_model_and_processor(self):
        # Importing the backend registers RBLNGotOcr2ForConditionalGeneration
        # into optimum.rbln. Kept inside this method so that importing
        # ``vlmeval.vlm.rbln`` does not pull in the RBLN runtime.
        from . import got_ocr2_backend  # noqa: F401
        from optimum.rbln import RBLNGotOcr2ForConditionalGeneration
        from transformers import AutoProcessor

        model = self._from_pretrained(RBLNGotOcr2ForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    # ------------------------------------------------------------------
    # Query-mode routing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_box(prompt: str) -> list[int] | None:
        """Return the ``[x1, y1, x2, y2]`` region a prompt asks about.

        Coordinates are returned in the prompt's own 0-1000 normalized
        space; :meth:`_processor_kwargs` rescales them to pixels because
        ``GotOcr2Processor`` normalizes by the image size itself.
        """
        lowered = prompt.lower()
        if 'normalized' not in lowered and 'coordinate' not in lowered:
            return None
        m = _BOX_RE.search(prompt)
        if m is None:
            return None
        box = [int(v) for v in m.groups()]
        if box[0] >= box[2] or box[1] >= box[3]:
            return None
        if max(box) > 1000:
            return None
        return box

    def _resolve_mode(self, prompt: str) -> tuple[bool, list[int] | None]:
        """``(format_output, normalized_box)`` for a dataset question."""
        if self.ocr_mode == 'plain':
            return False, None
        if self.ocr_mode == 'format':
            return True, None
        lowered = prompt.lower()
        fmt = any(marker in lowered for marker in _FORMAT_MARKERS)
        # The two axes are independent: GotOcr2Processor composes them into
        # ``"[box] OCR with format: "``, which OCRBench_v2's region-scoped
        # "table parsing cn" items ("Parse the HTML-formatted table
        # structure within the region [136, 750, 863, 870] ...") need.
        return fmt, self._extract_box(prompt)

    def _processor_kwargs(self, prompt: str, image) -> dict:
        fmt, box = self._resolve_mode(prompt)
        kwargs: dict = {'format': fmt}
        if box is not None:
            # GotOcr2Processor.preprocess_box_annotation divides by the
            # image size, so it wants *pixel* coordinates. Convert the
            # prompt's 0-1000 values back to pixels; the processor then maps
            # them to the same 0-1000 values in the query string.
            width, height = image.size
            kwargs['box'] = [
                box[0] * width / 1000,
                box[1] * height / 1000,
                box[2] * width / 1000,
                box[3] * height / 1000,
            ]
        self._debug_log(f'ocr query: format={fmt} box={kwargs.get("box")}', 'blue')
        return kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        if image_path is None:
            raise ValueError(f'{type(self).__name__} requires an image input.')

        image = self._load_image(image_path)
        inputs = self.processor(
            images=image,
            return_tensors='pt',
            **self._processor_kwargs(prompt or '', image),
        )
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        return self._finalize_response(inputs, generated_ids)
