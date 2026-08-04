"""Export PaddlePaddle/PP-OCRv5_server_det to static-shape ONNX buckets.

The RBLN compile pipeline stopped one step short: it reuses the vendored
backend's ``export_onnx`` + ``fix_static_shape`` (the exact functions
``ppocrv5_det_backend.compile_backend`` feeds into ``rebel.compile_from_onnx``)
and simply keeps the static ONNX. Running those graphs on onnxruntime's CUDA
provider is the GPU analog of the NPU's static-graph execution, which is what
score-parity comparisons need (see ``vlmeval/vlm/ppocr_det.py``).

Resolution is baked into each artifact; ``det_<h>x<w>.onnx`` files act as
buckets exactly like the ``.rbln`` ones. The default exports ONE shape —
768x1024 (h x w), the empirically best single shape on OCRBench_v2 detection
(see tests/rbln/PPOCRV5_DET_OCRBENCH_V2_RESULTS.md) — so every image is
squashed into it and the GPU run reproduces the RBLN benchmark configuration.

    python scripts/ppocr_det_export.py
    SHAPES=960x960,768x1024 python scripts/ppocr_det_export.py
    ARTIFACT_DIR=./elsewhere python scripts/ppocr_det_export.py

Requires ``paddle2onnx`` (which needs a paddle runtime — the CPU build
suffices) and ``onnx`` / ``onnxruntime``.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = 'PaddlePaddle/PP-OCRv5_server_det'
LOCAL_DIR = os.environ.get('CHECKPOINT_DIR', './PP-OCRv5_server_det')
ARTIFACT_DIR = os.environ.get('ARTIFACT_DIR', './PP-OCRv5_server_det-onnx')
INPUT_NAME = 'x'

# (h, w), multiples of 32 — see module docstring for why a single shape.
SHAPES = [
    tuple(int(v) for v in s.split('x'))
    for s in os.environ.get('SHAPES', '768x1024').split(',')
]


def download():
    if Path(LOCAL_DIR, 'inference.json').exists():
        return LOCAL_DIR
    from huggingface_hub import snapshot_download
    print(f'[download] {MODEL_ID} -> {LOCAL_DIR}', flush=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
    return LOCAL_DIR


def main():
    from vlmeval.vlm.rbln.ppocrv5_det_backend import export_onnx, fix_static_shape

    src = download()
    Path(ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print('[1/2] paddle -> ONNX (dynamic shape)', flush=True)
    dyn = export_onnx(src, f'{ARTIFACT_DIR}/work/det.onnx', opset=13)

    print(f'[2/2] fixing {len(SHAPES)} static buckets', flush=True)
    for h, w in SHAPES:
        out = f'{ARTIFACT_DIR}/det_{h}x{w}.onnx'
        fix_static_shape(dyn, out, INPUT_NAME, [1, 3, h, w])
        print(f'   {h}x{w} -> {out} ({Path(out).stat().st_size / 1e6:.1f} MB)', flush=True)

    print(f'[export] done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
