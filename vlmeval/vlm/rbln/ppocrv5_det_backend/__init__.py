"""Self-contained PP-OCRv5 text-detection backend for the RBLN NPU.

This is **not** an optimum-rbln model. PP-OCRv5_server_det ships as a
PaddlePaddle inference model, and paddle is not an RBLN frontend, so it
goes through ONNX and is compiled with ``rebel`` directly:

    inference.json + inference.pdiparams  --paddle2onnx-->  ONNX
                                          --fix shapes-->   ONNX(static)
                                          --compile_from_onnx--> .rbln

No numerical workarounds were needed — it is a standard CNN with no
unsupported ops; only the **dynamic shapes** have to be fixed. But fixing
them bakes the resolution into the artifact, hence the **resolution
buckets** (see :func:`~.runtime.pick_bucket`).

Because the runtime is raw ``rebel.Runtime`` rather than optimum-rbln,
:class:`~vlmeval.vlm.rbln.RBLNPPOCRv5Det` disables the optimum-specific
hooks it would otherwise inherit (there is no ``save_pretrained`` and no
``rbln_config.json``).

⚠️ Import this lazily — importing it pulls ``cv2`` / ``rebel``, and
``vlmeval/vlm/rbln`` guarantees that importing the package never loads the
RBLN runtime (tests/rbln/test_imports.py).
"""

from __future__ import annotations

from .compile_backend import compile_det, export_onnx, fix_static_shape
from .postprocess import boxes_to_rects, db_postprocess
from .runtime import (DetRuntime, make_ort_session, paddle_resize_shape, pick_bucket,
                      preprocess_image)

__all__ = [
    'export_onnx',
    'fix_static_shape',
    'compile_det',
    'db_postprocess',
    'boxes_to_rects',
    'DetRuntime',
    'preprocess_image',
    'paddle_resize_shape',
    'pick_bucket',
    'make_ort_session',
]
