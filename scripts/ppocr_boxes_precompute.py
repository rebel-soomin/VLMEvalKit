"""Freeze PP-OCRv5 detection output into a box cache for det -> rec runs.

Why freeze it
-------------
Recognition needs line crops, and on full-page images a detector supplies
them — so an end-to-end score depends on two models. Comparing two
recognition runs (NPU vs GPU, or two recognition configs) with detection live
on each side isolates nothing: detection is a **sharp** cutoff
(``box_thresh=0.6``), and small numerical differences flip boxes in and out.
That does not just jitter the score, it changes **how many crops exist and
where they are**, so the two runs never see the same inputs. Measured on the
detection port: re-encoding the source images JPEG -> PNG moved the detection
score 1.5 points.

Running detection once and freezing it here makes the recognition forward pass
the only variable.

Which device should produce the cache
-------------------------------------
``--det-device onnx-cpu`` (default) is the neutral choice for a parity
artifact: neither accelerator gets an advantage and it reproduces on any
machine. Use ``rbln`` when the cache should represent the full RBLN pipeline
(the product path), or ``onnx-cuda`` for the GPU pipeline. The choice is
recorded in the cache's ``meta`` so a reader can tell what produced it.

Usage
-----
    # neutral CPU boxes for a parity comparison
    python scripts/ppocr_boxes_precompute.py \
        --data CCOCR_MultiLanOcr_Korean \
        --det-model ./PP-OCRv5_server_det-onnx \
        --out ./ppocr_boxes/boxes.json

    # the RBLN pipeline's own boxes
    python scripts/ppocr_boxes_precompute.py \
        --data CCOCR_MultiLanOcr_Korean --det-device rbln \
        --det-model ./PP-OCRv5_server_det-rbln \
        --out ./ppocr_boxes/boxes_rbln.json

Detection is run over the dataset's **own dumped image files**, the same ones
inference reads, because the cache is keyed by image content — see
``vlmeval/vlm/rbln/ppocrv5_rec_backend/box_cache.py``.
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys
import time

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_DET_DEVICES = ('onnx-cpu', 'onnx-cuda', 'rbln')


def _build_runtime_factory(det_model: str, det_device: str, device_id: int):
    """Return ``(factory(bucket) -> runtime, buckets, artifact_suffix)``."""
    import glob
    import re

    from vlmeval.vlm.rbln import ppocrv5_det_backend as backend

    suffix = 'rbln' if det_device == 'rbln' else 'onnx'
    pattern = osp.join(det_model, f'det_*x*.{suffix}')
    buckets = []
    for path in glob.glob(pattern):
        m = re.search(rf'det_(\d+)x(\d+)\.{suffix}$', osp.basename(path))
        if m:
            buckets.append((int(m.group(1)), int(m.group(2))))
    buckets = sorted(buckets)
    if not buckets:
        raise SystemExit(
            f'no det_<h>x<w>.{suffix} artifacts in {det_model!r}. '
            + ('Compile them first.' if suffix == 'rbln'
               else 'Run scripts/ppocr_det_export.py first.'))

    cache: dict = {}

    def factory(bucket):
        if bucket in cache:
            return cache[bucket]
        path = osp.join(det_model, f'det_{bucket[0]}x{bucket[1]}.{suffix}')
        if det_device == 'rbln':
            rt = backend.DetRuntime(path, device=device_id)
        else:
            providers = (['CPUExecutionProvider'] if det_device == 'onnx-cpu' else
                         [('CUDAExecutionProvider', {'device_id': device_id}),
                          'CPUExecutionProvider'])
            session = backend.make_ort_session(path, providers)
            active = session.get_providers()[0]
            if det_device == 'onnx-cuda' and active != 'CUDAExecutionProvider':
                print(f'[warn] requested CUDA but onnxruntime activated {active}; '
                      'boxes will be produced on CPU', flush=True)

            class _Ort:
                def run(self, x):
                    return session.run(None, {'x': x})[0]

            rt = _Ort()
        cache[bucket] = rt
        return rt

    return factory, buckets, suffix


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='dataset name, e.g. CCOCR_MultiLanOcr_Korean')
    ap.add_argument('--det-model', required=True, help='detection artifact directory')
    ap.add_argument('--det-device', default='onnx-cpu', choices=_DET_DEVICES)
    ap.add_argument('--out', required=True, help='output cache path (.json)')
    ap.add_argument('--device-id', type=int, default=0)
    ap.add_argument('--categories', default=None,
                    help='comma-separated category filter (datasets with a category column)')
    # Detection postprocess — model-card defaults, same as the det wrapper.
    ap.add_argument('--thresh', type=float, default=0.3)
    ap.add_argument('--box-thresh', type=float, default=0.6)
    ap.add_argument('--unclip-ratio', type=float, default=1.5)
    ap.add_argument('--limit-side-len', type=int, default=64)
    ap.add_argument('--limit-type', default='min', choices=('min', 'max'))
    args = ap.parse_args()

    from vlmeval.dataset import build_dataset
    from vlmeval.vlm.rbln import ppocrv5_det_backend as det_backend
    from vlmeval.vlm.rbln.ppocrv5_rec_backend import (image_digest, make_record,
                                                      save_box_cache)

    dataset = build_dataset(args.data)
    if dataset is None:
        raise SystemExit(f'unknown dataset {args.data!r}')
    if args.categories:
        wanted = [c.strip() for c in args.categories.split(',') if c.strip()]
        if 'category' not in dataset.data:
            raise SystemExit(f'{args.data} has no "category" column')
        dataset.data = dataset.data[dataset.data['category'].isin(wanted)].reset_index(drop=True)
        if not len(dataset.data):
            raise SystemExit(f'--categories {wanted} matched no samples')

    factory, buckets, suffix = _build_runtime_factory(
        args.det_model, args.det_device, args.device_id)
    print(f'[det] {args.det_device} · buckets {buckets}', flush=True)

    from PIL import Image

    records: dict = {}
    n_boxes = 0
    t0 = time.time()
    total = len(dataset.data)
    for i in range(total):
        line = dataset.data.iloc[i]
        paths = dataset.dump_image(line)
        image_path = paths[0] if isinstance(paths, list) else paths
        image = Image.open(image_path).convert('RGB')

        target = det_backend.paddle_resize_shape(
            image.height, image.width, args.limit_side_len, args.limit_type)
        bucket, _exact = det_backend.pick_bucket(target, buckets)
        x, (orig_h, orig_w) = det_backend.preprocess_image(image, bucket)
        prob = factory(bucket).run(x)
        boxes, scores = det_backend.db_postprocess(
            prob, orig_w, orig_h, args.thresh, args.box_thresh, args.unclip_ratio)

        digest = image_digest(image_path)
        records[digest] = make_record(image_path, orig_w, orig_h, boxes, scores)
        n_boxes += len(boxes)
        if (i + 1) % 25 == 0 or i + 1 == total:
            print(f'  {i + 1}/{total} images · {n_boxes} boxes', flush=True)

    meta = {
        'dataset': args.data,
        'categories': args.categories,
        'det_model': osp.abspath(args.det_model),
        'det_device': args.det_device,
        'det_artifact_kind': suffix,
        'buckets': [list(b) for b in buckets],
        'postprocess': {
            'thresh': args.thresh,
            'box_thresh': args.box_thresh,
            'unclip_ratio': args.unclip_ratio,
            'limit_side_len': args.limit_side_len,
            'limit_type': args.limit_type,
        },
        'n_images': len(records),
        'n_boxes': n_boxes,
    }
    save_box_cache(args.out, records, meta)
    print(f'[done] {len(records)} images · {n_boxes} boxes · {time.time() - t0:.0f}s '
          f'-> {args.out}')


if __name__ == '__main__':
    main()
