"""GPU wrapper for PP-OCRv5 text detection (``PaddlePaddle/PP-OCRv5_server_det``).

The GPU twin of :class:`~vlmeval.vlm.rbln.RBLNPPOCRv5Det`: it **subclasses**
the RBLN wrapper and swaps only the executor, so query routing, box
selection and every output format (grounding tuple / bbox-dict / spotting
list — all locked by ``tests/rbln/test_ppocrv5_det_output.py``) are
inherited byte-identical. Pre/postprocess (squash resize, DBPostProcess)
come from the shared vendored :mod:`~vlmeval.vlm.rbln.ppocrv5_det_backend`,
which never imports the RBLN runtime at module level, so this class runs on
machines without ``rebel``/``optimum-rbln`` installed.

Static-graph parity: the artifact is the same paddle -> ONNX ->
static-shape graph the NPU compiles (``scripts/ppocr_det_export.py`` is the
compile pipeline stopped one step before ``rebel.compile_from_onnx``),
executed on onnxruntime's CUDA provider. ``det_<h>x<w>.onnx`` files act as
resolution buckets exactly like the ``.rbln`` artifacts do — put a single
``det_768x1024.onnx`` in the directory to reproduce the RBLN benchmark
configuration (one fixed shape, every image squashed into it).
"""

from __future__ import annotations

import glob
import os
import re

from .rbln.ppocrv5_det import RBLNPPOCRv5Det

_ONNX_BUCKET_RE = re.compile(r'det_(\d+)x(\d+)\.onnx$')

_INPUT_NAME = 'x'


class _OrtDetRuntime:
    """Drop-in for ``ppocrv5_det_backend.DetRuntime`` backed by onnxruntime."""

    def __init__(self, onnx_path: str, providers):
        from .rbln.ppocrv5_det_backend import make_ort_session
        self.session = make_ort_session(onnx_path, providers)

    def run(self, x):
        return self.session.run(None, {_INPUT_NAME: x})[0]


class PPOCRv5Det(RBLNPPOCRv5Det):
    """PP-OCRv5_server_det on onnxruntime CUDA. See the module docstring.

    ``device=None`` (default) resolves the GPU ordinal per rank: under
    ``run.py --gpus N`` each torchrun rank is pinned to its own slice of
    GPUs via ``CUDA_VISIBLE_DEVICES``, so ``LOCAL_RANK % visible`` lands on
    device 0 of that slice; single-process runs also get device 0.
    """

    def __init__(
        self,
        model_path: str = './PP-OCRv5_server_det-onnx',
        use_gpu: bool = True,
        device: int | None = None,
        **kwargs,
    ) -> None:
        self.use_gpu = use_gpu
        super().__init__(model_path=model_path, device=device, **kwargs)

    # The artifact dir holds local ONNX exports — never fall through to the
    # RBLN compiled-dir / HF-snapshot resolution of the base class.
    def _resolve_model_path(self, model_path: str) -> str:
        if os.path.isdir(model_path):
            return os.path.abspath(model_path)
        raise FileNotFoundError(
            f'{model_path!r} is not a directory. Run '
            '`python scripts/ppocr_det_export.py` first to produce '
            'det_<h>x<w>.onnx artifacts.'
        )

    def _load_rbln_model_and_processor(self):
        from .rbln import ppocrv5_det_backend as backend

        self._backend = backend
        self.buckets = self._discover_buckets()
        if not self.buckets:
            raise ValueError(
                f'No det_<h>x<w>.onnx artifacts found in {self.model_path!r}. '
                'Run `python scripts/ppocr_det_export.py` first.'
            )
        self._providers = self._pick_providers()
        return None, None

    def _discover_buckets(self):
        out = []
        for path in glob.glob(os.path.join(self.model_path, 'det_*x*.onnx')):
            m = _ONNX_BUCKET_RE.search(os.path.basename(path))
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
        return sorted(out)

    def _pick_providers(self):
        if not self.use_gpu:
            return ['CPUExecutionProvider']
        # onnxruntime-gpu resolves cuDNN/cuBLAS from already-loaded shared
        # libraries; importing torch first loads the pip-shipped NVIDIA libs
        # so the CUDA provider initializes without a system CUDA install.
        import torch
        if not torch.cuda.is_available():
            self._warn_once(
                'nogpu',
                f'{type(self).__name__}: use_gpu=True but CUDA is not '
                'available; falling back to CPUExecutionProvider.',
            )
            return ['CPUExecutionProvider']
        if self.device is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device_id = local_rank % max(torch.cuda.device_count(), 1)
        else:
            device_id = self.device
        return [
            ('CUDAExecutionProvider', {'device_id': device_id}),
            'CPUExecutionProvider',
        ]

    def _runtime(self, bucket):
        if bucket not in self._runtimes:
            path = os.path.join(self.model_path, f'det_{bucket[0]}x{bucket[1]}.onnx')
            rt = _OrtDetRuntime(path, self._providers)
            active = rt.session.get_providers()[0]
            if self.use_gpu and active != 'CUDAExecutionProvider':
                self._warn_once(
                    'cpu_fallback',
                    f'{type(self).__name__}: requested CUDA but onnxruntime '
                    f'activated {active}; inference will run on CPU.',
                )
            self._runtimes[bucket] = rt
        return self._runtimes[bucket]
