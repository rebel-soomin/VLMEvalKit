"""Detection-only P/R/F1 for PP-OCRv5_server_det predictions on OCRBench_v2.

OCRBench_v2's official "text spotting en" hmean is end-to-end: a detection
counts only when IoU >= 0.5 AND the transcription matches. A pure detector
(empty transcription) therefore scores 0 there by construction. This script
scores the SAME predictions with the same matching protocol minus the
transcription equality — the detection-quality number that GPU-vs-NPU
parity comparisons use.

Protocol (mirrors the RRC script used by the official metric):
  * axis-aligned IoU >= 0.5, greedy one-to-one matching in GT order;
  * GT entries whose content is '###' are don't-care: they never count as
    recall, and any unmatched detection overlapping one (IoU >= 0.5) is
    dropped from the precision denominator.

Usage: python scripts/ppocr_det_score.py <prediction_xlsx>
"""

import ast
import sys


def rect_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def poly_to_rect(poly):
    xs, ys = poly[0::2], poly[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


def score_sample(pred_boxes, gt_boxes, gt_care, iou_thr=0.5):
    """Return (matched, n_gt_care, n_det_counted) for one image."""
    matched_det = set()
    matched = 0
    for gi, (g, care) in enumerate(zip(gt_boxes, gt_care)):
        if not care:
            continue
        for di, d in enumerate(pred_boxes):
            if di in matched_det:
                continue
            if rect_iou(g, d) >= iou_thr:
                matched_det.add(di)
                matched += 1
                break
    # Precision denominator: drop unmatched detections overlapping don't-care GT.
    dontcare = [g for g, care in zip(gt_boxes, gt_care) if not care]
    n_det = 0
    for di, d in enumerate(pred_boxes):
        if di in matched_det:
            n_det += 1
            continue
        if any(rect_iou(g, d) >= iou_thr for g in dontcare):
            continue
        n_det += 1
    return matched, sum(gt_care), n_det


def main(pred_file):
    from vlmeval.smp import load

    data = load(pred_file)
    spotting = data[data['category'] == 'text spotting en']
    total_m = total_gt = total_det = 0
    for _, row in spotting.iterrows():
        # Empty predictions round-trip through the xlsx as NaN.
        raw = str(row['prediction']).strip()
        pred = ast.literal_eval(raw) if raw and raw != 'nan' else []
        pred_boxes = [tuple(p[:4]) for p in pred if len(p) >= 4]
        gt_polys = ast.literal_eval(row['bbox'])
        contents = ast.literal_eval(row['content'])
        gt_boxes = [poly_to_rect(p) for p in gt_polys]
        gt_care = [str(c).strip() != '###' for c in contents]
        m, g, d = score_sample(pred_boxes, gt_boxes, gt_care)
        total_m += m
        total_gt += g
        total_det += d

    recall = total_m / total_gt if total_gt else 0.0
    precision = total_m / total_det if total_det else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f'text spotting en (detection-only, IoU>=0.5, {len(spotting)} images):')
    print(f'  matched={total_m} gt={total_gt} det={total_det}')
    print(f'  precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}')
    return precision, recall, f1


if __name__ == '__main__':
    main(sys.argv[1])
