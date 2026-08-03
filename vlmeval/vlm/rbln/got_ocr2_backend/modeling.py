"""User-facing RBLN classes for GOT-OCR2.0.

Structure (follows optimum-rbln's ``llava`` pattern)
----------------------------------------------------
GOT-OCR2.0 = vision tower (SAM/ViTDet, 89.7M) + multi_modal_projector +
language_model (Qwen2, 463.9M). Its module tree matches LLaVA's
(``model.model.language_model`` is a ``Qwen2Model``, ``model.lm_head`` is
separate), so the same three-way split applies:

* ``vision_tower``   -> submodule, compiled by :class:`RBLNGotOcr2VisionEncoder`.
* ``language_model`` -> submodule, **zero new code** — ``text_config`` is
  qwen2, so optimum-rbln's ``RBLNQwen2ForCausalLM`` machinery is reused
  verbatim.
* ``multi_modal_projector`` -> compiled by the end-to-end class itself
  (2x conv_upsampler + projector).

Neither transformers nor optimum-rbln is patched on disk: registration
happens at import time against optimum-rbln's in-memory registries (see
``__init__.py``).
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Optional

import torch
from optimum.rbln.configuration_utils import RBLNCompileConfig, RBLNModelConfig
from optimum.rbln.modeling import RBLNModel
from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
    RBLNDecoderOnlyGenerationMixin,
)
from transformers import AutoModelForImageTextToText, PreTrainedModel

# transformers 5.x moved no_init_weights to transformers.initialization.
try:
    from transformers.initialization import no_init_weights
except ImportError:  # transformers 4.x
    from transformers.modeling_utils import no_init_weights

from .architecture import GotOcr2VisionEncoderWrapper
from .configuration import (RBLNGotOcr2ForConditionalGenerationConfig,
                            RBLNGotOcr2VisionEncoderConfig)


class RBLNGotOcr2VisionEncoder(RBLNModel):
    """Runs the GOT-OCR2.0 vision tower on the NPU.

    Works around the missing ``aten::upsample_linear1d`` (rel_pos
    interpolation) by precomputing it on the host — see
    :mod:`.architecture`.
    """

    _rbln_config_class = RBLNGotOcr2VisionEncoderConfig

    @classmethod
    def get_hf_class(cls):
        """``GotOcr2VisionEncoder`` is not exported at the transformers top level.

        The default implementation strips "RBLN" from the class name and
        looks it up in the ``transformers`` namespace, which returns None
        here and then fails with ``'NoneType' object has no attribute
        'from_pretrained'``.
        """
        from transformers.models.got_ocr2.modeling_got_ocr2 import GotOcr2VisionEncoder

        return GotOcr2VisionEncoder

    @classmethod
    def _wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: RBLNModelConfig):
        return GotOcr2VisionEncoderWrapper(model, rbln_config=rbln_config).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors=None,
        model: Optional[PreTrainedModel] = None,
        model_config=None,
        rbln_config: Optional[RBLNGotOcr2VisionEncoderConfig] = None,
    ) -> RBLNGotOcr2VisionEncoderConfig:
        image_size = rbln_config.image_size
        if image_size is None:
            vc = getattr(model_config, 'vision_config', model_config)
            image_size = vc.image_size
            rbln_config.image_size = image_size

        input_info = [
            ('pixel_values', [rbln_config.batch_size, 3, image_size, image_size], 'float32'),
        ]
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        """Re-wrap the raw tensor the way the caller expects.

        Returning a bare Tensor makes ``get_image_features`` fail with
        ``AttributeError: 'Tensor' object has no attribute
        'last_hidden_state'`` (observed).
        """
        from transformers.modeling_outputs import BaseModelOutput

        out = super().forward(pixel_values.contiguous().to(torch.float32))
        tensor = out if isinstance(out, torch.Tensor) else out[0]
        return BaseModelOutput(last_hidden_state=tensor)


class RBLNGotOcr2ForConditionalGeneration(RBLNModel, RBLNDecoderOnlyGenerationMixin):
    """End-to-end GOT-OCR2.0. vision_tower / language_model are submodules;
    only the projector is compiled by this class."""

    auto_model_class = AutoModelForImageTextToText
    _rbln_config_class = RBLNGotOcr2ForConditionalGenerationConfig
    _rbln_submodules = [
        {'name': 'vision_tower'},
        {'name': 'language_model'},
    ]

    def __getattr__(self, name: str) -> Any:
        """Borrow the HF implementation (multimodal merge helpers etc.),
        same trick optimum-rbln's llava class uses."""
        from transformers import GotOcr2ForConditionalGeneration

        def redirect(func):
            return lambda *args, **kwargs: func(self, *args, **kwargs)

        val = getattr(GotOcr2ForConditionalGeneration, name)
        if isinstance(val, Callable) and 'self' in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def can_generate(self):
        return True

    @classmethod
    def _reconstruct_model_if_needed(cls, model: PreTrainedModel):
        """Reassemble ``language_model`` (Qwen2Model) into a ``Qwen2ForCausalLM``.

        The decoder-only machinery expects a CausalLM that owns lm_head,
        while GOT splits it into ``model.model.language_model`` (Model) +
        ``model.lm_head``. Same situation as LLaVA, same fix.
        """
        with no_init_weights():
            model_cls_name = model.model.language_model.__class__.__name__
            causal_cls_name = model_cls_name.replace('Model', 'ForCausalLM')
            causal_cls = getattr(importlib.import_module('transformers'), causal_cls_name)
            new_lm = causal_cls(model.model.language_model.config)

        new_lm.lm_head = model.lm_head
        new_lm.model = model.model.language_model
        model.model.language_model = new_lm
        model.lm_head = None
        del model.lm_head
        return model

    @classmethod
    def _wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: RBLNModelConfig):
        # The only thing this class compiles directly is the projector.
        return model.model.multi_modal_projector

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors=None,
        model: Optional[PreTrainedModel] = None,
        model_config=None,
        rbln_config: Optional[RBLNGotOcr2ForConditionalGenerationConfig] = None,
    ) -> RBLNGotOcr2ForConditionalGenerationConfig:
        vc = model_config.vision_config
        # The vision tower emits [1, output_channels, H/patch, W/patch] (4D).
        grid = vc.image_size // vc.patch_size
        input_info = [
            ('vision_embeddings', [1, vc.output_channels, grid, grid], 'float32'),
        ]
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def __post_init__(self, **kwargs):
        self.vision_tower = self.rbln_submodules[0]
        self.language_model = self.rbln_submodules[1]
        self.multi_modal_projector = self.model[0]
        return super().__post_init__(**kwargs)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_attn_impl(self) -> str:
        return self.rbln_config.language_model.attn_impl

    def get_kvcache_num_blocks(self) -> int:
        return self.rbln_config.language_model.kvcache_num_blocks

    # ------------------------------------------------------------------
    # Multimodal generate loop
    #
    # The decoder-only machinery splits into prefill / decode runtimes.
    # Image embeddings have to be spliced into the text embeddings during
    # prefill, which is what the three methods below are for.
    # ------------------------------------------------------------------

    def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs):
        """vision tower (NPU) -> projector (NPU)."""
        vision_out = self.vision_tower(pixel_values)
        last_hidden = (vision_out.last_hidden_state
                       if hasattr(vision_out, 'last_hidden_state') else vision_out)
        return self.multi_modal_projector(last_hidden.contiguous().to(torch.float32))

    def _preprocess_prefill(self, input_ids=None, inputs_embeds=None, pixel_values=None):
        """Splice image features into the text embeddings at each image token."""
        if inputs_embeds is not None:
            return inputs_embeds
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return inputs_embeds

        image_features = self.get_image_features(pixel_values)
        image_features = image_features.reshape(-1, inputs_embeds.shape[-1]).to(
            inputs_embeds.dtype)

        image_token_id = getattr(self.config, 'image_token_id',
                                 getattr(self.config, 'image_token_index', None))
        mask = (input_ids == image_token_id)
        n_slots, n_feats = int(mask.sum()), image_features.shape[0]
        if n_slots != n_feats:
            raise ValueError(
                f'image token count ({n_slots}) != image feature count ({n_feats})')
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = image_features
        return inputs_embeds

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, pixel_values=None,
                                      attention_mask=None, cache_position=None,
                                      generate_idx=None, **kwargs):
        is_prefill = generate_idx is None
        model_inputs = {}
        if is_prefill:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            cache_position = None
        else:
            if inputs_embeds is not None:
                raise NotImplementedError(
                    'inputs_embeds is not supported during the decode step.')
            pixel_values = None
            input_ids = input_ids[:, -1:]
            cache_position = generate_idx
            generate_idx = generate_idx + 1
        model_inputs.update({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'cache_position': cache_position,
            'generate_idx': generate_idx,
        })
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs,
                                            is_encoder_decoder, **kwargs):
        model_kwargs['generate_idx'] = outputs.generate_idx
        return model_kwargs

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
                inputs_embeds=None, return_dict=None, cache_position=None,
                generate_idx=None, **kwargs):
        from optimum.rbln.transformers.models.decoderonly.modeling_decoderonly import (
            RBLNDecoderOnlyOutput,
        )

        if cache_position is None:      # prefill
            inputs_embeds = self._preprocess_prefill(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                pixel_values=pixel_values)
            logits = []
            for b in range(inputs_embeds.shape[0]):
                pos = torch.arange(0, generate_idx[b].item(),
                                   dtype=torch.int32).unsqueeze(0)
                out = self.language_model.prefill_decoder(
                    inputs_embeds=inputs_embeds[b:b + 1],
                    attention_mask=attention_mask[b] if attention_mask is not None else None,
                    cache_position=pos,
                    batch_idx=b,
                )
                logits.append(out.logits)
            logits = torch.cat(logits, dim=0)
        else:                           # decode
            logits = self.language_model.decoder(
                input_ids=input_ids, cache_position=cache_position).logits

        if not return_dict:
            return logits, generate_idx
        return RBLNDecoderOnlyOutput(logits=logits, generate_idx=generate_idx)
