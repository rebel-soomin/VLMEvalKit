"""GPU wrapper for PP-OCRv5 text recognition (``PaddlePaddle/*_PP-OCRv5_mobile_rec``).

The GPU twin of :class:`~vlmeval.vlm.rbln.RBLNPPOCRv5Rec`: it **subclasses**
the RBLN wrapper and swaps only the executor, so mode routing, quadrilateral
cropping, ``RecResizeImg`` preprocessing, CTC decoding and the ``drop_score``
filter are inherited byte-identical. All of that lives in the shared vendored
:mod:`~vlmeval.vlm.rbln.ppocrv5_rec_backend`, which never imports the RBLN
runtime at module level, so this class runs on machines without
``rebel``/``optimum-rbln`` installed.

Static-graph parity: the artifact is the same graph the NPU compiles.
``scripts/ppocr_rec_export.py`` runs the compile pipeline — paddle -> ONNX
(opset 17) -> ``onnxsim`` -> ``onnx2torch`` -> ``make_static()`` — and stops
one step before ``rebel.compile_from_torch``, re-exporting the patched torch
module to ONNX instead. Re-exporting *after* the patches is what makes it the
same graph: the ``adaptive_avg_pool2d`` substitution and the frozen
Reshape/Slice/Squeeze constants are applied at the torch level. On the NPU
that substitution works around a miscomputed ``GlobalAveragePool``; on the
GPU it is mathematically equivalent to the original, which the compile
pipeline asserts (``max|delta| == 0``) before writing anything.

``rec_48x<w>.onnx`` files act as width buckets exactly like the ``.rbln``
artifacts do.

**Detection must be frozen for a comparison to mean anything.** Pass the same
``boxes_cache`` to both wrappers: detection is a sharp cutoff, so running it
live on each side changes the number and geometry of crops and the two
recognition runs never see the same inputs. See
:mod:`~vlmeval.vlm.rbln.ppocrv5_rec_backend.box_cache`.
"""

from __future__ import annotations

import glob
import os
import re

from .rbln.ppocrv5_rec import RBLNPPOCRv5Rec

_ONNX_BUCKET_RE = re.compile(r'rec_(\d+)x(\d+)\.onnx$')


class _OrtRecRuntime:
    """Drop-in for ``ppocrv5_rec_backend.RecRuntime`` backed by onnxruntime."""

    def __init__(self, onnx_path: str, providers):
        from .rbln.ppocrv5_rec_backend import INPUT_NAME, make_ort_session
        self.session = make_ort_session(onnx_path, providers)
        self._input_name = INPUT_NAME
        names = [i.name for i in self.session.get_inputs()]
        if self._input_name not in names:
            # The re-export names it explicitly, so a mismatch means the
            # artifact came from somewhere else.
            if len(names) != 1:
                raise ValueError(
                    f'{onnx_path} has inputs {names}; expected a single '
                    f'{INPUT_NAME!r}.')
            self._input_name = names[0]

    def run(self, x):
        return self.session.run(None, {self._input_name: x})[0]


class PPOCRv5Rec(RBLNPPOCRv5Rec):
    """PP-OCRv5 recognition on onnxruntime CUDA. See the module docstring.

    ``device=None`` (default) resolves the GPU ordinal per rank: under
    ``run.py --gpus N`` each torchrun rank is pinned to its own slice of GPUs
    via ``CUDA_VISIBLE_DEVICES``, so ``LOCAL_RANK % visible`` lands on device
    0 of that slice; single-process runs also get device 0.
    """

    def __init__(
        self,
        model_path: str = './korean_PP-OCRv5_mobile_rec-onnx',
        use_gpu: bool = True,
        device: int | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get('det_model_path'):
            # det_model_path points at .rbln detection artifacts, so honouring
            # it here would try to run detection on the NPU from the GPU
            # wrapper. Refuse rather than half-work.
            raise ValueError(
                f'{type(self).__name__} does not run detection; det_model_path is '
                'for the RBLN wrapper. Pass boxes_cache=<path from '
                'scripts/ppocr_boxes_precompute.py> instead, which is also what '
                'makes this run comparable to the NPU run.'
            )
        self.use_gpu = use_gpu
        super().__init__(model_path=model_path, device=device, **kwargs)

    # The artifact dir holds local ONNX exports — never fall through to the
    # RBLN compiled-dir / HF-snapshot resolution of the base class.
    def _resolve_model_path(self, model_path: str) -> str:
        if os.path.isdir(model_path):
            return os.path.abspath(model_path)
        raise FileNotFoundError(
            f'{model_path!r} is not a directory. Run '
            '`python scripts/ppocr_rec_export.py` first to produce '
            'rec_48x<w>.onnx artifacts.'
        )

    def _load_rbln_model_and_processor(self):
        out = super()._load_rbln_model_and_processor()
        self._providers = self._pick_providers()
        return out

    def _discover_buckets(self) -> list[int]:
        widths, heights = [], set()
        for path in glob.glob(os.path.join(self.model_path, 'rec_*x*.onnx')):
            m = _ONNX_BUCKET_RE.search(os.path.basename(path))
            if m:
                heights.add(int(m.group(1)))
                widths.append(int(m.group(2)))
        if len(heights) > 1:
            raise ValueError(
                f'Mixed recognition heights {sorted(heights)} in {self.model_path!r}; '
                'the height is baked in and must be identical across width buckets.')
        self.rec_height = heights.pop() if heights else 48
        return sorted(widths)

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

    def _runtime(self, width: int):
        if width not in self._runtimes:
            path = os.path.join(self.model_path, f'rec_{self.rec_height}x{width}.onnx')
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f'{path} missing. Export the same width buckets the NPU run '
                    'used, or the two runs will not be comparable.')
            rt = _OrtRecRuntime(path, self._providers)
            active = rt.session.get_providers()[0]
            if self.use_gpu and active != 'CUDAExecutionProvider':
                self._warn_once(
                    'cpu_fallback',
                    f'{type(self).__name__}: requested CUDA but onnxruntime '
                    f'activated {active}; inference will run on CPU.',
                )
            self._runtimes[width] = rt
        return self._runtimes[width]

    # A live detector here would be the RBLN detector; the GPU twin uses the
    # ONNX detector instead. Recognition comparisons should use a frozen box
    # cache anyway, so live detection is refused rather than silently wired
    # to the wrong device.
    def _detect(self, image):
        raise NotImplementedError(
            f'{type(self).__name__} does not run detection. Pass '
            'boxes_cache=<path from scripts/ppocr_boxes_precompute.py> so this '
            'run consumes the same frozen boxes as the NPU run — that is what '
            'makes the two recognition results comparable.'
        )
