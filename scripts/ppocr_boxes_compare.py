"""Compare two frozen detection box caches — the detection-stage parity check.

Recognition parity (``scripts/ppocr_rec_parity.py``) deliberately holds
detection fixed so the recognition forward pass is the only variable. This
script measures the stage that was held fixed: given two caches built over the
same images by different producers (NPU vs CPU vs GPU, or two input shapes),
how much do the box sets actually agree?

Reported per image and pooled:

* ``matched``      — greedy one-to-one pairs at IoU >= threshold (axis-aligned)
* ``precision`` / ``recall`` / ``f1`` — treating cache A as reference. Neither
  cache is ground truth, so these are *agreement* rates, not accuracy: swapping
  the arguments swaps precision and recall.
* ``mean_iou``     — over matched pairs, i.e. how tightly the agreed boxes align
* ``only_a`` / ``only_b`` — boxes with no counterpart

Comparing caches is only meaningful when they cover the same images; images
present in one and not the other are reported and excluded.

Usage
-----
    python scripts/ppocr_boxes_compare.py a.json b.json \
        [--label-a rbln] [--label-b cpu] [--iou 0.5] [--out cmp.json]
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _rect(quad):
    xs = [float(quad[i]) for i in range(0, len(quad), 2)]
    ys = [float(quad[i]) for i in range(1, len(quad), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _match(ra, rb, thr):
    """Greedy one-to-one matching, best pair first. Returns (pairs, ious)."""
    cands = sorted(
        ((_iou(x, y), i, j) for i, x in enumerate(ra) for j, y in enumerate(rb)),
        key=lambda t: -t[0])
    used_a, used_b, ious = set(), set(), []
    for iou, i, j in cands:
        if iou < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        ious.append(iou)
    return ious


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cache_a')
    ap.add_argument('cache_b')
    ap.add_argument('--label-a', default='a')
    ap.add_argument('--label-b', default='b')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from vlmeval.vlm.rbln.ppocrv5_rec_backend import load_box_cache

    meta_a, rec_a = load_box_cache(args.cache_a)
    meta_b, rec_b = load_box_cache(args.cache_b)
    keys = sorted(set(rec_a) & set(rec_b))
    if not keys:
        raise SystemExit('the two caches share no images (different image files?)')
    only = (len(set(rec_a) - set(rec_b)), len(set(rec_b) - set(rec_a)))
    if any(only):
        print(f'[warn] {only[0]} images only in {args.label_a}, {only[1]} only in '
              f'{args.label_b}; comparing the {len(keys)} in common', flush=True)

    print(f'{args.label_a}: det_device={meta_a.get("det_device")!r} '
          f'buckets={meta_a.get("buckets")} n_boxes={meta_a.get("n_boxes")}')
    print(f'{args.label_b}: det_device={meta_b.get("det_device")!r} '
          f'buckets={meta_b.get("buckets")} n_boxes={meta_b.get("n_boxes")}')
    if meta_a.get('buckets') != meta_b.get('buckets'):
        print('[note] the caches use DIFFERENT input shapes, so this measures '
              'shape sensitivity, not device parity.', flush=True)
    if meta_a.get('postprocess') != meta_b.get('postprocess'):
        print('[warn] postprocess settings differ; disagreement is not '
              'attributable to the device.', flush=True)

    tot_a = tot_b = tot_m = 0
    all_ious: list[float] = []
    identical = 0
    per_image = []
    for k in keys:
        ba = [_rect(q) for q in rec_a[k]['boxes']]
        bb = [_rect(q) for q in rec_b[k]['boxes']]
        ious = _match(ba, bb, args.iou)
        tot_a += len(ba)
        tot_b += len(bb)
        tot_m += len(ious)
        all_ious += ious
        if len(ba) == len(bb) == len(ious):
            identical += 1
        per_image.append({'image_name': rec_a[k].get('image_name'),
                          'n_a': len(ba), 'n_b': len(bb), 'matched': len(ious)})

    recall = tot_m / (tot_a + 1e-9)
    precision = tot_m / (tot_b + 1e-9)
    f1 = 2 * recall * precision / (recall + precision + 1e-9)
    summary = {
        'label_a': args.label_a, 'label_b': args.label_b,
        'iou_threshold': args.iou, 'images': len(keys),
        'boxes_a': tot_a, 'boxes_b': tot_b, 'matched': tot_m,
        f'only_in_{args.label_a}': tot_a - tot_m,
        f'only_in_{args.label_b}': tot_b - tot_m,
        'agreement_recall': round(100 * recall, 4),
        'agreement_precision': round(100 * precision, 4),
        'agreement_f1': round(100 * f1, 4),
        'mean_iou_of_matched': round(100 * sum(all_ious) / max(len(all_ious), 1), 4),
        'images_fully_identical': identical,
    }
    print('\n' + json.dumps(summary, indent=2))

    worst = sorted(per_image, key=lambda r: r['matched'] - max(r['n_a'], r['n_b']))[:8]
    print('\nlargest disagreements (image, n_a, n_b, matched):')
    for r in worst:
        print(f'  {r["image_name"]:>12}  {r["n_a"]:4d} {r["n_b"]:4d} {r["matched"]:4d}')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary, 'meta': {args.label_a: meta_a,
                                                    args.label_b: meta_b},
                       'per_image': per_image}, f, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
