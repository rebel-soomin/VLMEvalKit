"""Detection preprocessing / NPU execution / resolution-bucket selection.

Pre- and post-processing run on the CPU (outside the compiled graph); the
NPU handles only the forward pass.

**Why resolution buckets are needed.** PaddleOCR's ``DetResizeForTest``
picks a per-image inference resolution: it preserves aspect ratio and
rounds h/w to a multiple of 32. So the resolution depends on the input
image size — but RBLN bakes shapes in at compile time, so each resolution
needs its own artifact. A few buckets are compiled ahead of time and the
closest one is chosen at run time.
"""

from __future__ import annotations

import numpy as np

# PP-OCR detection standard normalisation.
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def paddle_resize_shape(h, w, limit_side_len=64, limit_type='min'):
    """The inference resolution PaddleOCR's ``DetResizeForTest`` would pick.

    The model card's defaults are ``limit_type='min', limit_side_len=64``.
    (Older PaddleOCR used ``'max', 960`` — both are supported.)
    """
    if limit_type == 'min':
        ratio = 1.0 if min(h, w) >= limit_side_len else limit_side_len / min(h, w)
    elif limit_type == 'max':
        ratio = 1.0 if max(h, w) <= limit_side_len else limit_side_len / max(h, w)
    else:
        raise ValueError(f"limit_type must be 'min' or 'max': {limit_type}")
    rh, rw = int(h * ratio), int(w * ratio)
    rh = max(int(round(rh / 32) * 32), 32)
    rw = max(int(round(rw / 32) * 32), 32)
    return rh, rw


def pick_bucket(target_hw, buckets):
    """Pick the best compiled bucket for a target resolution.

    An exact match is used when available (identical result to PaddleOCR).
    Otherwise the bucket with the **closest area** is used — that adds one
    more resize, so results can differ slightly from PaddleOCR.
    """
    th, tw = target_hw
    if (th, tw) in buckets:
        return (th, tw), True
    ta = th * tw
    best = min(buckets, key=lambda b: (abs(b[0] * b[1] - ta), abs(b[0] / b[1] - th / tw)))
    return best, False


def preprocess_image(img, shape_hw):
    """PIL/ndarray image -> ``[1,3,h,w]`` float32 NCHW + the original size.

    ``shape_hw`` is the compiled artifact's ``(h, w)``.
    """
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    oh, ow = arr.shape[:2]
    rh, rw = shape_hw
    resized = cv2.resize(arr, (rw, rh))
    rgb = resized.astype(np.float32) / 255.0
    x = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x), (oh, ow)


class DetRuntime:
    """Runs a compiled detection artifact on the NPU."""

    def __init__(self, rbln_path: str, device: int = 0):
        import rebel
        self.rt = rebel.Runtime(rbln_path, device=device)

    def run(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.rt.run(np.ascontiguousarray(x, dtype=np.float32)))


def make_ort_session(onnx_path, providers=('CPUExecutionProvider',)):
    """onnxruntime session for CPU reference comparisons, with an **explicit
    thread count** to avoid affinity errors.

    Where CPUs are restricted by cgroup/taskset (containers), onnxruntime's
    default is to create one thread per *system* core and pin each to a
    specific core. Pinning to a disallowed core fails and floods the log:

        [E:onnxruntime:Default, env.cc:226 ThreadMain] pthread_setaffinity_np
        failed for thread: N, index: M, mask: {..}, error code: 22 ...
        Specify the number of threads explicitly so the affinity is not set.

    It **does not affect results** (measured: max|Δ|=0 vs the default), but
    threads pile onto fewer cores and real errors get buried.
    ``OMP_NUM_THREADS`` does not help — this is onnxruntime's own thread
    pool. Setting the counts explicitly makes onnxruntime skip the
    automatic affinity assignment.
    """
    import os

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = len(os.sched_getaffinity(0))   # actually-allowed cores
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=list(providers))
