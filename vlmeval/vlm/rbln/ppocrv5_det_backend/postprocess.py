"""DBPostProcess — reproduces PaddleOCR's implementation.

probability map -> binarize -> contours -> min-area rect -> **unclip
expansion** -> rescale to original image coordinates.

Defaults are the model card's
(https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det):
``thresh=0.3, box_thresh=0.6, unclip_ratio=1.5``. With these, this
implementation reproduced **98 detected lines** on the card's test image
(matching the card's ``rec_texts`` count) and its ``dt_scores`` to within
1e-3.

``unclip`` needs ``pyclipper`` and ``shapely``.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_SIZE = 3
MAX_CANDIDATES = 1000


def _get_mini_boxes(contour):
    rect = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    i1, i4 = (0, 1) if points[1][1] > points[0][1] else (1, 0)
    i2, i3 = (2, 3) if points[3][1] > points[2][1] else (3, 2)
    return np.array([points[i1], points[i2], points[i3], points[i4]]), min(rect[1])


def _box_score_fast(bitmap, box_):
    """Mean probability inside the polygon — this is the ``dt_score``."""
    h, w = bitmap.shape[:2]
    box = box_.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] -= xmin
    box[:, 1] -= ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box, ratio):
    import pyclipper
    from shapely.geometry import Polygon

    poly = Polygon(box)
    distance = poly.area * ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return np.array(offset.Execute(distance))


def db_postprocess(prob_map, dest_w, dest_h, thresh=0.3, box_thresh=0.6,
                   unclip_ratio=1.5):
    """probability map -> ``(boxes, scores)``.

    ``boxes`` are 4-point polygons in the *original* image coordinate space
    (pass the original width/height as ``dest_w`` / ``dest_h``).
    """
    pred = np.asarray(prob_map)
    while pred.ndim > 2:
        pred = pred[0]
    bitmap = pred > thresh
    height, width = bitmap.shape

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[0] if len(outs) == 2 else outs[1]

    boxes, scores = [], []
    for contour in contours[:MAX_CANDIDATES]:
        points, sside = _get_mini_boxes(contour)
        if sside < MIN_SIZE:
            continue
        score = _box_score_fast(pred, points.reshape(-1, 2))
        if score < box_thresh:
            continue
        expanded = _unclip(points, unclip_ratio)
        if len(expanded) == 0:
            continue
        box, sside = _get_mini_boxes(expanded.reshape(-1, 1, 2))
        if sside < MIN_SIZE + 2:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_w), 0, dest_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_h), 0, dest_h)
        boxes.append(box.astype('int32'))
        scores.append(float(score))
    return boxes, scores


def boxes_to_rects(boxes):
    """4-point polygons -> axis-aligned ``(x0, y0, x1, y1)`` rects."""
    out = []
    for b in boxes:
        b = np.asarray(b)
        out.append((int(b[:, 0].min()), int(b[:, 1].min()),
                    int(b[:, 0].max()), int(b[:, 1].max())))
    return out
