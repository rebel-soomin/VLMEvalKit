"""Compile a PP-OCRv5 recognition checkpoint into RBLN width buckets.

Recognition needs **two workarounds** that detection does not; both are
documented with their measurements in
``vlmeval/vlm/rbln/ppocrv5_rec_backend/__init__.py``:

1. export at **opset 17** so LayerNorm stays fused, and enter through the
   **torch** frontend (``compile_from_onnx`` rejects the graph either way);
2. substitute ``adaptive_avg_pool2d`` for ``GlobalAveragePool`` — otherwise
   the compile *succeeds* and every string comes out empty.

Compiling needs no NPU.

Width is baked into each artifact, so several widths are compiled and the
run-time picks the smallest that fits the crop's aspect ratio. Too narrow a
bucket compresses glyphs and drops spaces.

    python scripts/ppocr_rec_compile.py
    MODEL_ID=PaddlePaddle/PP-OCRv5_mobile_rec python scripts/ppocr_rec_compile.py
    WIDTHS=320,480,640 python scripts/ppocr_rec_compile.py
    ONNX=1 python scripts/ppocr_rec_compile.py      # also export the GPU twin's ONNX

``ONNX=1`` additionally writes ``rec_48x<w>.onnx`` next to the ``.rbln``
files, re-exported from the **patched** torch module so the GPU twin runs the
same graph the NPU compiles (see ``vlmeval/vlm/ppocr_rec.py``).
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = os.environ.get('MODEL_ID', 'PaddlePaddle/korean_PP-OCRv5_mobile_rec')
_NAME = MODEL_ID.split('/')[-1]
LOCAL_DIR = os.environ.get('CHECKPOINT_DIR', f'./{_NAME}')
SAVE_DIR = os.environ.get('ARTIFACT_DIR', f'./{_NAME}-rbln')
ONNX_DIR = os.environ.get('ONNX_DIR', f'./{_NAME}-onnx')
WANT_ONNX = os.environ.get('ONNX', '') not in ('', '0', 'false', 'False')

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
    from vlmeval.vlm.rbln.ppocrv5_rec_backend import (compile_rec, export_onnx_op17,
                                                      export_torch_onnx, simplify_static)

    src = download()
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print('[1/2] paddle -> ONNX (opset 17 — keeps LayerNorm fused)', flush=True)
    onnx_path = export_onnx_op17(src, f'{SAVE_DIR}/work/rec_op17.onnx')

    print(f'[2/2] compiling {len(WIDTHS)} width buckets {WIDTHS}', flush=True)
    for w in WIDTHS:
        shape = [1, 3, IMAGE_HEIGHT, w]
        static = f'{SAVE_DIR}/work/rec_op17_simp_w{w}.onnx'
        simplify_static(onnx_path, static, shape)

        out = f'{SAVE_DIR}/rec_{IMAGE_HEIGHT}x{w}.rbln'
        compile_rec(static, out, shape, verbose=(w == WIDTHS[0]))
        print(f'   {IMAGE_HEIGHT}x{w} -> {out} '
              f'({Path(out).stat().st_size / 1e6:.1f} MB)', flush=True)

        if WANT_ONNX:
            Path(ONNX_DIR).mkdir(parents=True, exist_ok=True)
            oout = f'{ONNX_DIR}/rec_{IMAGE_HEIGHT}x{w}.onnx'
            export_torch_onnx(static, oout, shape, verbose=False)
            print(f'   {IMAGE_HEIGHT}x{w} -> {oout} '
                  f'({Path(oout).stat().st_size / 1e6:.1f} MB)', flush=True)

    print(f'[compile] done in {time.time() - t0:.0f}s')
    print(f'The character dictionary is NOT in the artifact — pass '
          f'charset_dir={LOCAL_DIR} at run time.')


if __name__ == '__main__':
    main()
