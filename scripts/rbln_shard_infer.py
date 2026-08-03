"""Shard one RBLN inference job across several NPUs.

Why this exists
---------------
``run.py`` parallelises inference by launching one process per rank and
synchronising them with ``torch.distributed`` — but it hardcodes the
``nccl`` backend, which needs CUDA. On an RBLN host the accelerator is an
NPU and torch is the CPU build, so the multi-rank path is unavailable and
a large benchmark is stuck on a single device (10k OCRBench_v2 samples
~= 8h on one NPU vs ~1h across eight).

This script keeps VLMEvalKit's real inference path — it calls
``vlmeval.inference.infer_data``, the same function ``run.py`` calls, and
scores with the dataset's own ``evaluate`` — and only replaces the
process-coordination layer: each worker is an independent process that
takes its shard from ``RANK`` / ``WORLD_SIZE`` (which ``infer_data``
already reads) and is pinned to one NPU via ``RBLN_DEVICES``. The parent
waits on the workers, merges the per-rank result files and scores once.
No ``torch.distributed`` group is created, so no backend is required.

Usage
-----
    python scripts/rbln_shard_infer.py \
        --model stepfun-ai/GOT-OCR-2.0-hf --data OCRBench_v2 \
        --work-dir ./outputs/got_ocr2/wd_full --nproc 8

Run it from the directory holding the compiled artifact (the wrapper
resolves ``./<basename(model)>/`` first — see ``RBLNVLMBase._resolve_model_path``),
exactly as with ``run.py``.

Worker mode (``--rank``) is an internal detail: the parent re-invokes this
same file per shard.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import time

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _register_model(model: str, rbln_kwargs: dict | None) -> str:
    """Route ``model`` through the RBLN auto-dispatch, as ``run.py`` does."""
    from vlmeval.config import supported_VLM
    from vlmeval.vlm.rbln import register_rbln_auto

    alias = osp.basename(model.rstrip('/')) or model
    supported_VLM.pop(alias, None)
    register_rbln_auto(model, supported_VLM, extra_kwargs=rbln_kwargs,
                       register_as=alias)
    return alias


def _shard_file(work_dir: str, rank: int, world_size: int, dataset_name: str) -> str:
    return osp.join(work_dir, f'{rank}{world_size}_{dataset_name}.pkl')


def run_worker(args) -> int:
    """One shard. ``infer_data`` slices the dataset by RANK/WORLD_SIZE."""
    from vlmeval.dataset import build_dataset
    from vlmeval.inference import infer_data

    rbln_kwargs = json.loads(args.rbln_kwargs) if args.rbln_kwargs else None
    alias = _register_model(args.model, rbln_kwargs)

    dataset = build_dataset(args.data)
    if args.limit:
        dataset.data = dataset.data.head(args.limit).reset_index(drop=True)

    out_file = _shard_file(args.work_dir, args.rank, args.nproc, args.data)
    infer_data(alias, alias, args.work_dir, dataset, out_file,
               verbose=False, api_nproc=1)
    print(f'[rank {args.rank}] wrote {out_file}', flush=True)
    return 0


def run_parent(args) -> int:
    import pandas as pd

    from vlmeval.dataset import build_dataset
    from vlmeval.smp import dump, load

    os.makedirs(args.work_dir, exist_ok=True)

    dataset = build_dataset(args.data)
    if args.limit:
        dataset.data = dataset.data.head(args.limit).reset_index(drop=True)
    alias = osp.basename(args.model.rstrip('/')) or args.model

    devices = ([d.strip() for d in args.devices.split(',')] if args.devices
               else [str(i) for i in range(args.nproc)])
    if len(devices) < args.nproc:
        raise SystemExit(f'--devices lists {len(devices)} device(s) for '
                         f'--nproc {args.nproc}')

    procs = []
    for rank in range(args.nproc):
        env = dict(os.environ)
        env.update({
            'RANK': str(rank),
            'WORLD_SIZE': str(args.nproc),
            # One NPU per worker. Without this every worker lands on the
            # same device and they serialise (or fail to allocate).
            'RBLN_DEVICES': devices[rank],
        })
        cmd = [sys.executable, osp.abspath(__file__),
               '--model', args.model, '--data', args.data,
               '--work-dir', args.work_dir, '--nproc', str(args.nproc),
               '--rank', str(rank)]
        if args.limit:
            cmd += ['--limit', str(args.limit)]
        if args.rbln_kwargs:
            cmd += ['--rbln-kwargs', args.rbln_kwargs]
        log = open(osp.join(args.work_dir, f'rank{rank}.log'), 'w')
        print(f'[parent] rank {rank} -> RBLN_DEVICES={devices[rank]}', flush=True)
        procs.append((rank, subprocess.Popen(cmd, env=env, stdout=log,
                                             stderr=subprocess.STDOUT), log))

    t0 = time.time()
    failed = []
    for rank, p, log in procs:
        rc = p.wait()
        log.close()
        status = 'ok' if rc == 0 else f'FAILED rc={rc}'
        print(f'[parent] rank {rank} {status} ({time.time() - t0:.0f}s)', flush=True)
        if rc != 0:
            failed.append(rank)
    if failed:
        raise SystemExit(f'ranks {failed} failed; see {args.work_dir}/rank*.log')

    # Merge shards in dataset order, then score with the dataset's own
    # evaluate() — the same call run.py makes.
    merged: dict = {}
    for rank in range(args.nproc):
        merged.update(load(_shard_file(args.work_dir, rank, args.nproc, args.data)))
    missing = [i for i in dataset.data['index'] if i not in merged]
    if missing:
        raise SystemExit(f'{len(missing)} predictions missing after merge')

    data = dataset.data.copy()
    data['prediction'] = [str(merged[i]) for i in data['index']]
    if 'image' in data:
        # Same as infer_data_job: drop the base64 column, else the result
        # file carries the whole dataset again (1.4GB for OCRBench_v2).
        data.pop('image')
    result_file = osp.join(args.work_dir, f'{alias}_{args.data}.xlsx')
    dump(data, result_file)
    print(f'[parent] merged {len(data)} predictions -> {result_file}', flush=True)

    scores = dataset.evaluate(result_file)
    if isinstance(scores, pd.DataFrame):
        scores = scores.to_dict()
    print('[parent] scores:\n' + json.dumps(scores, indent=2, ensure_ascii=False),
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='HF id or compiled directory')
    ap.add_argument('--data', required=True, help='single dataset name')
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--nproc', type=int, default=8, help='shards = NPUs to use')
    ap.add_argument('--devices', default=None,
                    help='comma-separated RBLN_DEVICES values, one per rank '
                         '(default 0..nproc-1)')
    ap.add_argument('--limit', type=int, default=None, help='head-N of the dataset')
    ap.add_argument('--rbln-kwargs', default=None,
                    help='JSON passed to the wrapper, as in run.py')
    ap.add_argument('--rank', type=int, default=None,
                    help=argparse.SUPPRESS)  # internal: worker mode
    args = ap.parse_args()
    return run_worker(args) if args.rank is not None else run_parent(args)


if __name__ == '__main__':
    raise SystemExit(main())
