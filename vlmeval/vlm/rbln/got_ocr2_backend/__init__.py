"""Self-contained GOT-OCR2.0 backend for optimum-rbln.

optimum-rbln (as of 0.11.x) ships no GOT-OCR2.0 support, so this package
adds it. Importing it registers the classes into optimum-rbln's
**in-memory** registries — nothing under ``site-packages`` is modified and
there is no install step.

⚠️ Import this lazily. ``vlmeval/vlm/rbln`` guarantees that importing the
package never pulls in ``optimum.rbln`` (tests/rbln/test_imports.py), and
this module imports it at module scope. ``RBLNGotOcr2._load_rbln_model_and_processor``
is the only place that imports it.

Registration is needed in three places, each read by a different lookup path:

1. ``configuration_utils.CONFIG_MAPPING`` — an ``export=False`` load
   resolves ``rbln_config.json``'s ``cls_name`` by name.
2. ``utils.model_utils.MODEL_MAPPING`` — submodule export resolves
   ``rbln_model_cls_name``.
3. the ``optimum.rbln`` module namespace — ``RBLNModelConfig.rbln_model_cls``
   only looks at the namespace, with no MODEL_MAPPING fallback.
"""

from __future__ import annotations

import importlib

from .configuration import (RBLNGotOcr2ForConditionalGenerationConfig,
                            RBLNGotOcr2VisionEncoderConfig)
from .modeling import RBLNGotOcr2ForConditionalGeneration, RBLNGotOcr2VisionEncoder

_CONFIGS = [RBLNGotOcr2VisionEncoderConfig, RBLNGotOcr2ForConditionalGenerationConfig]
_MODELS = [RBLNGotOcr2VisionEncoder, RBLNGotOcr2ForConditionalGeneration]


def register() -> None:
    """Register the GOT-OCR2.0 classes with optimum-rbln. Idempotent."""
    from optimum.rbln.configuration_utils import CONFIG_MAPPING
    from optimum.rbln.utils.model_utils import MODEL_MAPPING

    ns = importlib.import_module('optimum.rbln')
    for cfg in _CONFIGS:
        CONFIG_MAPPING.setdefault(cfg.__name__, cfg)
        if not hasattr(ns, cfg.__name__):
            setattr(ns, cfg.__name__, cfg)
    for mdl in _MODELS:
        MODEL_MAPPING.setdefault(mdl.__name__, mdl)
        if not hasattr(ns, mdl.__name__):
            setattr(ns, mdl.__name__, mdl)


register()

__all__ = [
    'RBLNGotOcr2VisionEncoder',
    'RBLNGotOcr2VisionEncoderConfig',
    'RBLNGotOcr2ForConditionalGeneration',
    'RBLNGotOcr2ForConditionalGenerationConfig',
    'register',
]
