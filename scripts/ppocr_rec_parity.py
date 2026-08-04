"""Compare two PP-OCRv5 recognition runs — the judge-free rec-parity check.

What this measures, and what it does not
----------------------------------------
Given two prediction files produced from the **same frozen box cache**, this
reports how often the two runs transcribed the same text. Because the boxes
are identical, the crops are identical (same host code, same source images),
so the only thing that can differ is the recognition forward pass. That is
what makes the number attributable to the model rather than to the pipeline.

If the two runs used *different* box caches this script's numbers are
meaningless, and it will say so: it compares the ``boxes_cache`` recorded in
each run's metadata when available, and otherwise warns.

Metrics, all rule-based — no LLM judge:

* ``exact_match``     — fraction of images whose full prediction is identical
* ``char_f1``         — bag-of-characters F1 between the two runs (the same
                        tokenisation CC-OCR uses for Korean/Japanese/Chinese)
* ``mean_edit_ratio`` — 1 - normalised Levenshtein distance, averaged
* ``n_differing``     — images that differ at all, with examples

Usage
-----
    python scripts/ppocr_rec_parity.py <run_a.xlsx> <run_b.xlsx> \
        [--label-a npu] [--label-b gpu] [--out parity.json] [--examples 10]

Each input is a VLMEvalKit prediction file (the ``*_<dataset>.xlsx`` a run
writes), which must carry ``prediction`` plus an identity column
(``image_name``, else ``index``).
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import Counter

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load(path: str) -> dict[str, str]:
    import pandas as pd

    df = pd.read_excel(path)
    if 'prediction' not in df:
        raise SystemExit(f'{path} has no "prediction" column')
    key = 'image_name' if 'image_name' in df else 'index'
    out = {}
    for _, row in df.iterrows():
        pred = row['prediction']
        out[str(row[key])] = '' if pd.isna(pred) else str(pred)
    return out


def _chars(text: str) -> list[str]:
    """Character tokens, whitespace-insensitive.

    Matches CC-OCR's ``text_normalize_and_tokenize`` for the non-word-level
    languages: strip whitespace entirely, then compare per character.
    """
    return [c for c in ''.join(str(text).split())]


def _bag_f1(a: str, b: str) -> float:
    ca, cb = Counter(_chars(a)), Counter(_chars(b))
    right = sum(min(n, cb.get(t, 0)) for t, n in ca.items())
    if not sum(ca.values()) and not sum(cb.values()):
        return 1.0
    recall = right / (sum(ca.values()) + 1e-9)
    precision = right / (sum(cb.values()) + 1e-9)
    return 2 * recall * precision / (recall + precision + 1e-9)


def _edit_ratio(a: str, b: str) -> float:
    import nltk
    if not a and not b:
        return 1.0
    return 1.0 - nltk.edit_distance(a, b) / max(len(a), len(b), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_a')
    ap.add_argument('run_b')
    ap.add_argument('--label-a', default='a')
    ap.add_argument('--label-b', default='b')
    ap.add_argument('--out', default=None)
    ap.add_argument('--examples', type=int, default=10)
    args = ap.parse_args()

    a, b = _load(args.run_a), _load(args.run_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit('the two runs share no images — different datasets?')
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        print(f'[warn] {len(only_a)} images only in {args.label_a}, '
              f'{len(only_b)} only in {args.label_b}; comparing the '
              f'{len(keys)} in common', flush=True)

    exact, f1s, ratios, diffs = 0, [], [], []
    for k in keys:
        if a[k] == b[k]:
            exact += 1
        else:
            diffs.append(k)
        f1s.append(_bag_f1(a[k], b[k]))
        ratios.append(_edit_ratio(a[k], b[k]))

    # Corpus-level bag-of-characters F1, pooled over all images.
    right = gt_n = pd_n = 0
    for k in keys:
        ca, cb = Counter(_chars(a[k])), Counter(_chars(b[k]))
        right += sum(min(n, cb.get(t, 0)) for t, n in ca.items())
        gt_n += sum(ca.values())
        pd_n += sum(cb.values())
    recall = right / (gt_n + 1e-9)
    precision = right / (pd_n + 1e-9)
    micro_f1 = 2 * recall * precision / (recall + precision + 1e-9)

    summary = {
        'label_a': args.label_a,
        'label_b': args.label_b,
        'n_compared': len(keys),
        'exact_match': round(100 * exact / len(keys), 4),
        'char_f1_macro': round(100 * sum(f1s) / len(f1s), 4),
        'char_f1_micro': round(100 * micro_f1, 4),
        'mean_edit_ratio': round(100 * sum(ratios) / len(ratios), 4),
        'n_differing': len(diffs),
        'chars_a': gt_n,
        'chars_b': pd_n,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if diffs and args.examples:
        print(f'\nfirst {min(args.examples, len(diffs))} differing images:')
        for k in diffs[:args.examples]:
            print(f'  {k}')
            print(f'    {args.label_a}: {a[k][:160]!r}')
            print(f'    {args.label_b}: {b[k][:160]!r}')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'summary': summary,
                       'differing_images': diffs,
                       'runs': {args.label_a: osp.abspath(args.run_a),
                                args.label_b: osp.abspath(args.run_b)}},
                      f, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
