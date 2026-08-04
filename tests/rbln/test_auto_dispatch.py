"""US-002 — auto.py architecture dispatch.

``auto_select_wrapper`` maps a model path to ``(RBLNWrapperCls,
default_kwargs)`` using ``config.json``'s ``architectures`` field, with a
``"cosmos"`` substring override. These tests build throwaway local
directories with a synthetic ``config.json`` so no HF download or NPU is
involved, and assert:

* each ``_ARCH_TABLE`` token resolves to the documented wrapper class,
* ordering hazards are respected (paligemma2 before paligemma, the qwen
  variants, blip2),
* the ``"cosmos"`` path override wins without reading config,
* compile-time ``rbln_config`` defaults are seeded into the kwargs,
* an unknown architecture raises ``ValueError``.
"""

from __future__ import annotations
import json

import pytest

from vlmeval.vlm.rbln import (RBLNBlip2, RBLNCosmosReason1, RBLNGemma3, RBLNGotOcr2,
                              RBLNIdefics3, RBLNLlava, RBLNLlavaNext, RBLNPaliGemma,
                              RBLNPaliGemma2, RBLNPixtral, RBLNPPOCRv5Det,
                              RBLNPPOCRv5Rec, RBLNQwen2VL,
                              RBLNQwen3VL)
from vlmeval.vlm.rbln.auto import auto_select_wrapper


def _make_model_dir(tmp_path, name: str, architectures):
    d = tmp_path / name
    d.mkdir()
    (d / 'config.json').write_text(
        json.dumps({'architectures': architectures}), encoding='utf-8'
    )
    return str(d)


# (subdir name, architectures list, expected wrapper class)
_CASES = [
    ('qwen3', ['Qwen3VLForConditionalGeneration'], RBLNQwen3VL),
    ('qwen25', ['Qwen2_5_VLForConditionalGeneration'], RBLNQwen2VL),
    ('qwen2', ['Qwen2VLForConditionalGeneration'], RBLNQwen2VL),
    ('llavanext', ['LlavaNextForConditionalGeneration'], RBLNLlavaNext),
    ('llava', ['LlavaForConditionalGeneration'], RBLNLlava),
    ('idefics3', ['Idefics3ForConditionalGeneration'], RBLNIdefics3),
    ('gemma3', ['Gemma3ForConditionalGeneration'], RBLNGemma3),
    # NOTE: pixtral is NOT here — it ships as LlavaForConditionalGeneration
    # (model_type llava), indistinguishable from LLaVA-1.5 by architectures,
    # so it is routed by a path marker. See test_pixtral_* below.
    ('paligemma2', ['PaliGemma2ForConditionalGeneration'], RBLNPaliGemma2),
    ('paligemma', ['PaliGemmaForConditionalGeneration'], RBLNPaliGemma),
    ('blip2', ['Blip2ForConditionalGeneration'], RBLNBlip2),
    ('gotocr2', ['GotOcr2ForConditionalGeneration'], RBLNGotOcr2),
]


@pytest.mark.parametrize('name,archs,expected', _CASES,
                         ids=[c[0] for c in _CASES])
def test_arch_token_resolves_to_wrapper(tmp_path, name, archs, expected):
    path = _make_model_dir(tmp_path, name, archs)
    cls, defaults = auto_select_wrapper(path)
    assert cls is expected
    # Compile defaults are seeded into the returned kwargs.
    assert 'rbln_config' in defaults
    assert isinstance(defaults['rbln_config'], dict)


def test_paligemma2_wins_over_paligemma(tmp_path):
    """paligemma2 token precedes paligemma in _ARCH_TABLE — the more
    specific token must win even though 'paligemma2' contains 'paligemma'.
    """
    path = _make_model_dir(tmp_path, 'pg2', ['PaliGemma2ForConditionalGeneration'])
    cls, _ = auto_select_wrapper(path)
    assert cls is RBLNPaliGemma2
    assert cls is not RBLNPaliGemma


def test_qwen3_does_not_collapse_to_qwen2(tmp_path):
    path = _make_model_dir(tmp_path, 'q3', ['Qwen3VLForConditionalGeneration'])
    cls, _ = auto_select_wrapper(path)
    assert cls is RBLNQwen3VL


def test_cosmos_path_override_skips_config(tmp_path):
    """Anything with 'cosmos' in the path resolves to RBLNCosmosReason1
    even though it shares the Qwen2.5-VL architecture — and even without a
    config.json present (the override returns before reading it).
    """
    d = tmp_path / 'Cosmos-Reason1-7B'
    d.mkdir()  # intentionally no config.json
    cls, defaults = auto_select_wrapper(str(d))
    assert cls is RBLNCosmosReason1
    # Cosmos seeds its own visual.max_seq_lens budget (8192, vs Qwen 6400).
    assert defaults['rbln_config']['visual']['max_seq_lens'] == 8192


def test_pixtral_routed_by_path_despite_llava_arch(tmp_path):
    """Real Pixtral ships as architectures=['LlavaForConditionalGeneration']
    (model_type llava) — identical to LLaVA-1.5. It must still resolve to
    RBLNPixtral via the 'pixtral' path marker, with the Pixtral compile
    defaults (max_seq_len + kvcache_partition_len, which the LLaVA defaults
    lack and whose absence caused the 1M-context eager-attention failure).
    """
    path = _make_model_dir(tmp_path, 'pixtral-12b', ['LlavaForConditionalGeneration'])
    cls, defaults = auto_select_wrapper(path)
    assert cls is RBLNPixtral
    lm = defaults['rbln_config']['language_model']
    assert lm['max_seq_len'] == 131072
    assert lm['kvcache_partition_len'] == 16384


def test_llava_not_misrouted_to_pixtral(tmp_path):
    """Same architectures as Pixtral, but no 'pixtral' path marker -> stays
    RBLNLlava (guards against the path override over-matching)."""
    path = _make_model_dir(tmp_path, 'llava-1.5-7b-hf', ['LlavaForConditionalGeneration'])
    cls, _ = auto_select_wrapper(path)
    assert cls is RBLNLlava


def test_qwen_seeds_pixel_kwargs(tmp_path):
    path = _make_model_dir(tmp_path, 'q2', ['Qwen2VLForConditionalGeneration'])
    _, defaults = auto_select_wrapper(path)
    # _WRAPPER_KWARG_DEFAULTS seeds min/max pixels for Qwen families.
    assert 'min_pixels' in defaults and 'max_pixels' in defaults


def test_gotocr2_forces_inputs_embeds_via_config_not_table(tmp_path):
    """The GOT compile defaults deliberately omit
    ``language_model.use_inputs_embeds``: the vendored
    ``RBLNGotOcr2ForConditionalGenerationConfig`` forces it True, because a
    False value silently drops embed_tokens from the artifact and breaks
    generate. Guards against someone "helpfully" adding it here (where a
    caller override could then turn it off).
    """
    path = _make_model_dir(tmp_path, 'GOT-OCR-2.0-hf', ['GotOcr2ForConditionalGeneration'])
    cls, defaults = auto_select_wrapper(path)
    assert cls is RBLNGotOcr2
    lm = defaults['rbln_config']['language_model']
    assert 'use_inputs_embeds' not in lm
    assert lm['max_seq_len'] == 4096
    # vision_tower must be present (empty) so its submodule config is built.
    assert defaults['rbln_config']['vision_tower'] == {}


@pytest.mark.parametrize('name', [
    'PP-OCRv5_server_det',
    'PP-OCRv5_server_det-rbln',
    'PaddlePaddle/PP-OCRv5_server_det',
    'ppocrv5_mobile_det',
])
def test_ppocrv5_det_routed_by_path_marker(tmp_path, name):
    """PP-OCRv5 detection is a PaddlePaddle model with no HF config.json, so
    it must route on the path marker alone — before _fetch_architectures is
    reached (which would raise). The marker is punctuation-insensitive so
    'PP-OCRv5' matches 'ppocrv5'.
    """
    cls, defaults = auto_select_wrapper(name)
    assert cls is RBLNPPOCRv5Det
    # Compiled with rebel directly, not optimum-rbln: nothing to seed.
    assert defaults == {}


@pytest.mark.parametrize('name', [
    'korean_PP-OCRv5_mobile_rec-rbln',
    'PP-OCRv5_mobile_rec',
    'PP-OCRv5_server_rec-rbln',
    'ppocrv5_mobile_rec',
])
def test_ppocrv5_rec_routed_by_path_marker(name):
    """Recognition shares the ``ppocrv5`` marker with detection, so the stage
    token is what separates them. Routing rec to the detector would be silent
    nonsense — a detector emits coordinates, never text."""
    cls, defaults = auto_select_wrapper(name)
    assert cls is RBLNPPOCRv5Rec
    # Compiled with rebel directly, not optimum-rbln: nothing to seed.
    assert defaults == {}


def test_ppocrv5_det_and_rec_do_not_cross_route():
    """The two stages must never resolve to each other's wrapper."""
    assert auto_select_wrapper('PP-OCRv5_server_det-rbln')[0] is RBLNPPOCRv5Det
    assert auto_select_wrapper('korean_PP-OCRv5_mobile_rec-rbln')[0] is RBLNPPOCRv5Rec


def test_ppocrv5_without_stage_token_raises(tmp_path):
    """A ``ppocrv5`` path naming neither stage is ambiguous, and guessing would
    pick a model that cannot do the task. It must fail instead."""
    d = tmp_path / 'PP-OCRv5_something-rbln'
    d.mkdir()  # no config.json, as with the real artifact
    with pytest.raises(ValueError):
        auto_select_wrapper(str(d))


def test_unknown_architecture_raises(tmp_path):
    path = _make_model_dir(tmp_path, 'mystery', ['TotallyUnknownArch'])
    with pytest.raises(ValueError):
        auto_select_wrapper(path)


def test_missing_config_json_raises(tmp_path):
    d = tmp_path / 'nocfg'
    d.mkdir()
    with pytest.raises(ValueError):
        auto_select_wrapper(str(d))
