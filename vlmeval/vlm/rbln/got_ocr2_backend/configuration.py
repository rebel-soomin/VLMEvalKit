"""RBLN configs for GOT-OCR2.0.

* :class:`RBLNGotOcr2VisionEncoderConfig` — the SAM/ViTDet vision tower.
  ``image_size`` is the load-bearing value: the input resolution is baked
  into the compiled artifact.
* :class:`RBLNGotOcr2ForConditionalGenerationConfig` — the end-to-end
  model. Holds the two submodule configs (``vision_tower``,
  ``language_model``); ``language_model`` is a plain optimum-rbln
  decoder-only config because GOT's ``text_config`` is Qwen2.

Baked into the artifact: ``image_size`` (vision) and ``max_seq_len`` /
``batch_size`` / ``tensor_parallel_size`` / ``attn_impl`` (language).
Runtime-only keys (``device``, ``create_runtimes``) are not persisted.
"""

from __future__ import annotations

from typing import Optional

from optimum.rbln.configuration_utils import RBLNModelConfig


class RBLNGotOcr2VisionEncoderConfig(RBLNModelConfig):
    """Compile settings for the GOT-OCR2.0 vision tower."""

    def __init__(self, batch_size: Optional[int] = None,
                 image_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        # RBLNModelConfig does not provide batch_size — it has to be defined
        # here so _update_rbln_config can build input_info from it.
        self.batch_size = batch_size or 1
        # When None, modeling fills it from model_config.vision_config.image_size.
        self.image_size = image_size


class RBLNGotOcr2ForConditionalGenerationConfig(RBLNModelConfig):
    """End-to-end settings; composes the vision_tower / language_model configs."""

    submodules = ['vision_tower', 'language_model']

    def __init__(
        self,
        batch_size: Optional[int] = None,
        vision_tower: Optional[RBLNModelConfig] = None,
        language_model: Optional[RBLNModelConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f'batch_size must be a positive integer: {self.batch_size}')

        # The vision tower is compiled for a single image — GOT processes
        # crops sequentially.
        self.vision_tower = self.initialize_submodule_config(
            submodule_config=vision_tower, batch_size=1,
        )
        # GOT splices image embeddings into the text embeddings before
        # prefill, so the language model **must** accept inputs_embeds.
        # With use_inputs_embeds=False, embed_tokens is not saved as an
        # artifact, ``get_input_embeddings()`` returns None and generate
        # breaks (observed). Forced here rather than left to the caller.
        self.language_model = self.initialize_submodule_config(
            submodule_config=language_model, batch_size=self.batch_size,
            use_inputs_embeds=True,
        )
