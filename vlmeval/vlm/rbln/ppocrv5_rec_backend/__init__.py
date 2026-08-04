"""Self-contained PP-OCRv5 text-**recognition** backend for the RBLN NPU.

Not an optimum-rbln model: PP-OCRv5 ships as a PaddlePaddle inference model,
and paddle is not an RBLN frontend. Unlike the *detection* sibling
(:mod:`~vlmeval.vlm.rbln.ppocrv5_det_backend`, a plain CNN that
``compile_from_onnx`` accepts as-is), recognition needs a longer path and
**two workarounds**::

    inference.json + inference.pdiparams
      --paddle2onnx (opset 17)--->  ONNX          <- workaround (1)
      --onnxsim (fix input + fold shapes)-->  ONNX(static)
      --onnx2torch--------------->  torch.nn.Module
      --make_static()------------>  static graph  <- workaround (2)
      --rebel.compile_from_torch->  .rbln

**Workaround (1) — decomposed LayerNorm on a Conv output is rejected.**
Measured across all four combinations:

Columns are ``compile_from_onnx`` / ``compile_from_torch``:

=====================================================  =======  ========
graph                                                  onnx     torch
=====================================================  =======  ========
Conv -> reshape -> **fused** LayerNorm                 FAIL     **OK**
Conv -> reshape -> **decomposed** LayerNorm            FAIL     FAIL
Conv -> reshape (no LayerNorm)                         OK       OK
**decomposed** LayerNorm (no Conv)                     OK       OK
=====================================================  =======  ========

So only the *conjunction* "tensor derived from Conv" + "decomposed" fails;
LayerNorm itself is fine. Exporting at opset 17 keeps a fused
``LayerNormalization`` node that ``onnx2torch`` maps to ``nn.LayerNorm``.
``compile_from_onnx`` re-decomposes it internally even at opset 17, which is
why the **torch** frontend is mandatory rather than merely preferred.

**Workaround (2) — ``GlobalAveragePool`` divides by ``W`` instead of
``H*W``.** ⚠️ ``torch.mean(dim=(2,3), keepdim=True)`` immediately after an
``Add`` is miscomputed, and **the compile succeeds** — so the model silently
returns garbage (measured: every timestep predicts blank with 0.99
confidence, i.e. every string comes out empty). The error factor equals the
input height exactly:

==================  ==============
input shape         NPU/CPU ratio
==================  ==============
``[1,240,6,80]``    **6.01x**
``[1,240,6,40]``    **6.01x**
``[1,240,2,80]``    **2.00x**
``[1,240,1,80]``    1.00 (correct — H=1, so W == H*W)
==================  ==============

The trigger is the preceding ``Add``: ``mean`` alone and ``Relu -> mean`` are
both correct. :func:`~.onnx2torch_static.make_static` substitutes
``adaptive_avg_pool2d(x, 1)``, which is mathematically identical and correct
on the NPU.

Three defences keep workaround (2) from silently lapsing on an upgrade:
``onnx2torch`` is version-pinned (the patch targets its internal class
names); :func:`~.onnx2torch_static.make_static` raises if a patch target or
the GlobalAveragePool converter is missing; and the compile asserts CPU
equivalence (``max|delta| == 0``) *before* emitting an artifact.

⚠️ Import this lazily. It pulls ``cv2``/``rebel``, and
``vlmeval/vlm/rbln`` guarantees that importing the package never loads the
RBLN runtime (``tests/rbln/test_imports.py``).
"""

from __future__ import annotations

from .box_cache import (BoxCache, image_digest, load_box_cache, make_record,
                        merge_box_caches, save_box_cache)
from .compile_backend import (INPUT_NAME, compile_rec, export_onnx_op17,
                              export_torch_onnx, simplify_static)
from .onnx2torch_static import assert_equivalent, make_static
from .runtime import (PAD_VALUE, REC_BASE_WIDTH, REC_IMAGE_HEIGHT, RecRuntime, ctc_decode,
                      get_rotate_crop_image, load_charset, make_ort_session,
                      paddle_rec_width, pick_width_bucket, preprocess_line, sorted_boxes)

__all__ = [
    # compile
    'export_onnx_op17',
    'simplify_static',
    'compile_rec',
    'export_torch_onnx',
    'make_static',
    'assert_equivalent',
    'INPUT_NAME',
    # runtime
    'RecRuntime',
    'preprocess_line',
    'ctc_decode',
    'load_charset',
    'paddle_rec_width',
    'pick_width_bucket',
    'get_rotate_crop_image',
    'sorted_boxes',
    'make_ort_session',
    'REC_IMAGE_HEIGHT',
    'REC_BASE_WIDTH',
    'PAD_VALUE',
    # frozen detection
    'BoxCache',
    'image_digest',
    'load_box_cache',
    'save_box_cache',
    'merge_box_caches',
    'make_record',
]
