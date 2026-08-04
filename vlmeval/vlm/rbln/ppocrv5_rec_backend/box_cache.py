"""A frozen detection result, so det+rec comparisons are attributable.

Why this exists
---------------
Recognition needs line crops, and on full-page images those come from the
detector — so an end-to-end score depends on **two** models. Comparing NPU
against GPU with the detector live on each side does not isolate anything:
detection is a *sharp* cutoff (``box_thresh=0.6``), and small numerical
differences flip boxes in and out. That does not merely perturb the score, it
changes **how many crops exist and where they are**, so the two recognition
runs never see the same inputs. (Measured previously on the detection port:
re-encoding the source images JPEG -> PNG moved the detection score by 1.5
points for exactly this reason.)

The fix is to run detection **once**, freeze its output here, and have every
downstream run consume the identical box set. Then the recognition forward
pass is the only variable, and the recognition comparison means something.

Identity is the **sha256 of the image file's bytes**, not the dataset index.
The same lesson as above applies to the crops themselves: a re-encoded copy
of an image produces different crops, so a run must read the very files the
cache was built from. Keying on content makes a mismatch *loud* — an index
key would silently pair boxes with the wrong pixels.

A lookup miss or a hash mismatch **raises**. It deliberately does not fall
back to running detection live: a silent fallback would reintroduce precisely
the confound this module removes, and would do so invisibly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_VERSION = 1


def image_digest(path: str) -> str:
    """sha256 of the image file's bytes — the cache key."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save_box_cache(path: str, records: dict, meta: dict | None = None) -> str:
    """Write ``{digest: record}`` plus provenance metadata as JSON.

    ``meta`` should record *how* the boxes were produced (detector artifact,
    device, thresholds). Without it the cache is an unattributable pile of
    numbers — the whole point is that a reader can tell what produced it.
    """
    payload = {
        'version': CACHE_VERSION,
        'meta': dict(meta or {}),
        'boxes': records,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def merge_box_caches(paths, out_path: str, meta: dict | None = None) -> str:
    """Merge shard caches into one. Conflicting entries for a digest raise.

    Two shards producing *different* boxes for the same image means the
    producers were not identical — silently keeping one would hide that.
    """
    merged: dict = {}
    metas = []
    for p in paths:
        m, records = load_box_cache(p)
        metas.append(m)
        for digest, rec in records.items():
            prev = merged.get(digest)
            if prev is not None and prev.get('boxes') != rec.get('boxes'):
                raise ValueError(
                    f'conflicting boxes for {digest[:12]} '
                    f'({rec.get("image_name")}) while merging {p}: shards were '
                    'not produced by identical detectors'
                )
            merged[digest] = rec
    combined = dict(meta or {})
    combined.setdefault('merged_from', metas)
    return save_box_cache(out_path, merged, combined)


def load_box_cache(path: str):
    """Read a cache file -> ``(meta, {digest: record})``."""
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    version = payload.get('version')
    if version != CACHE_VERSION:
        raise ValueError(
            f'box cache {path} has version {version!r}, expected {CACHE_VERSION}')
    return payload.get('meta', {}), payload['boxes']


def make_record(image_path: str, width: int, height: int, boxes, scores) -> dict:
    """Build one cache record. ``boxes`` are 4x2 quads in original-image pixels."""
    return {
        'image_name': os.path.basename(image_path),
        'width': int(width),
        'height': int(height),
        'boxes': [[int(v) for v in np.asarray(b).reshape(-1)] for b in boxes],
        'scores': [float(s) for s in scores],
    }


class BoxCache:
    """Read-only lookup of frozen detection boxes, keyed by image content."""

    def __init__(self, path: str):
        self.path = path
        self.meta, self._records = load_box_cache(path)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, image_path: str):
        """Return ``(boxes, scores)`` for ``image_path``.

        ``boxes`` is a list of ``(4, 2)`` int arrays in original-image pixel
        coordinates. Raises :class:`KeyError` when the image is absent —
        never falls back to live detection (see the module docstring).
        """
        digest = image_digest(image_path)
        rec = self._records.get(digest)
        if rec is None:
            raise KeyError(
                f'{os.path.basename(image_path)} (sha256 {digest[:12]}) is not in '
                f'the frozen box cache {self.path} ({len(self._records)} entries). '
                'Either the cache was built from a different image set, or these '
                'image files were re-encoded after it was built — re-encoding '
                'changes the pixels and therefore the crops. Rebuild the cache '
                'with scripts/ppocr_boxes_precompute.py over these exact files.'
            )
        boxes = [np.asarray(b, dtype=np.int64).reshape(4, 2) for b in rec['boxes']]
        return boxes, list(rec.get('scores', []))
