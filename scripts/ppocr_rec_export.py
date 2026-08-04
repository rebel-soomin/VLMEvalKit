"""Export a PP-OCRv5 recognition checkpoint to static-shape ONNX width buckets.

The GPU twin of ``scripts/ppocr_rec_compile.py``: the same compile pipeline —
paddle -> ONNX (opset 17) -> ``onnxsim`` static -> ``onnx2torch`` ->
``make_static()`` — stopped one step before ``rebel.compile_from_torch``,
re-exporting the **patched torch module** to ONNX instead. Re-exporting after
the patches is what makes the GPU run the same graph the NPU compiles: the
``adaptive_avg_pool2d`` substitution and the frozen Reshape/Slice/Squeeze
constants are applied at the torch level (see
``vlmeval/vlm/rbln/ppocrv5_rec_backend/onnx2torch_static.py``). On the GPU the
substitution is mathematically equivalent to the original, which
``_prepare_torch_module`` asserts (``max|delta| == 0``) before anything is
written.

Needs no ``rebel``/``optimum-rbln`` — this runs on a plain GPU box. Requires
``paddle2onnx`` (CPU paddle runtime suffices), ``onnxsim`` and
``onnx2torch==1.5.15`` (the version the static patches are validated against).

Width is baked into each artifact; the runtime picks the smallest bucket that
fits the crop's aspect ratio. Export the SAME width set the NPU run used, or
the two runs will not be comparable.

    python scripts/ppocr_rec_export.py
    MODEL_ID=PaddlePaddle/PP-OCRv5_mobile_rec python scripts/ppocr_rec_export.py
    WIDTHS=320,480,640 python scripts/ppocr_rec_export.py

The character dictionary is NOT in the artifact — pass the downloaded
checkpoint directory as ``charset_dir`` at run time.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = os.environ.get('MODEL_ID', 'PaddlePaddle/korean_PP-OCRv5_mobile_rec')
_NAME = MODEL_ID.split('/')[-1]
LOCAL_DIR = os.environ.get('CHECKPOINT_DIR', f'./{_NAME}')
ONNX_DIR = os.environ.get('ONNX_DIR', f'./{_NAME}-onnx')

# Baked into every artifact. 48 is the trained height.
IMAGE_HEIGHT = int(os.environ.get('IMAGE_HEIGHT', '48'))
WIDTHS = [int(w) for w in os.environ.get('WIDTHS', '320,480,640,800,960').split(',')]


def download():
    if Path(LOCAL_DIR, 'inference.json').exists():
        return LOCAL_DIR
    from huggingface_hub import snapshot_download
    print(f'[download] {MODEL_ID} -> {LOCAL_DIR}', flush=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
    return LOCAL_DIR


def main():
    from vlmeval.vlm.rbln.ppocrv5_rec_backend import (export_onnx_op17,
                                                      export_torch_onnx, simplify_static)

    src = download()
    Path(ONNX_DIR).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print('[1/2] paddle -> ONNX (opset 17 — keeps LayerNorm fused)', flush=True)
    onnx_path = export_onnx_op17(src, f'{ONNX_DIR}/work/rec_op17.onnx')

    print(f'[2/2] exporting {len(WIDTHS)} width buckets {WIDTHS}', flush=True)
    for w in WIDTHS:
        shape = [1, 3, IMAGE_HEIGHT, w]
        static = f'{ONNX_DIR}/work/rec_op17_simp_w{w}.onnx'
        simplify_static(onnx_path, static, shape)

        out = f'{ONNX_DIR}/rec_{IMAGE_HEIGHT}x{w}.onnx'
        export_torch_onnx(static, out, shape, verbose=(w == WIDTHS[0]))
        print(f'   {IMAGE_HEIGHT}x{w} -> {out} '
              f'({Path(out).stat().st_size / 1e6:.1f} MB)', flush=True)

    print(f'[export] done in {time.time() - t0:.0f}s')
    print(f'The character dictionary is NOT in the artifact — pass '
          f'charset_dir={LOCAL_DIR} at run time.')


if __name__ == '__main__':
    main()
