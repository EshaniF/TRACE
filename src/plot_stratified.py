
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import Counter


def load_relation_frequencies(data_dir: str) -> dict:
    """Count relation appearances in train.txt. Returns {r_id: count}."""
    freq = Counter()
    path = os.path.join(data_dir, "train.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"train.txt not found in {data_dir}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    freq[int(parts[1])] += 1
                except ValueError:
                    pass
    return dict(freq)


DEFAULT_EDGES = [0, 10, 50, 200, np.inf]
DEFAULT_NAMES = ["very_rare\n(<10)", "rare\n(10-50)",
                 "medium\n(50-200)", "frequent\n(>=200)"]


def assign_relation_buckets(rel_freq: dict, num_rels: int,
                             edges: list, names: list) -> np.ndarray:
    """
    Returns (num_rels,) int array mapping each base relation id to a
    bucket index in [0, len(names)). Relations absent from rel_freq
    (freq=0, i.e. never seen in training) fall into bucket 0
    (very_rare), since edges[0]=0 and freq=0 satisfies
    edges[0] <= freq < edges[1].
    """
    assert len(edges) == len(names) + 1, \
        "edges must have exactly one more element than names"

    bucket_of_rel = np.zeros(num_rels, dtype=np.int64)
    for r in range(num_rels):
        f = rel_freq.get(r, 0)
        for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if lo <= f < hi:
                bucket_of_rel[r] = b
                break
        else:
            bucket_of_rel[r] = len(names) - 1  # fallback: top bucket
    return bucket_of_rel


def compute_metrics(ranks: np.ndarray) -> dict:
    """
    Given a 1D array of 1-indexed filtered ranks, return
    {'MRR': ..., 'H@1': ..., 'H@3': ..., 'H@10': ..., 'n': ...}.
    Returns NaNs for MRR/H@k if ranks is empty (bucket has no triples),
    so empty buckets are visually distinguishable (no bar) rather than
    silently plotted as zero.
    """
    n = len(ranks)
    if n == 0:
        return {'MRR': np.nan, 'H@1': np.nan, 'H@3': np.nan,
                'H@10': np.nan, 'n': 0}
    ranks = ranks.astype(np.float64)
    return {
        'MRR':  float(np.mean(1.0 / ranks)),
        'H@1':  float(np.mean(ranks <= 1)),
        'H@3':  float(np.mean(ranks <= 3)),
        'H@10': float(np.mean(ranks <= 10)),
        'n':    n,
    }


def parse_runs(run_specs: list) -> list:
    """
    Parse --runs entries of the form "path.npz:Label" (label optional,
    defaults to filename without extension). ':' inside paths is not
    supported -- use simple filenames.
    """
    runs = []
    for spec in run_specs:
        if ':' in spec:
            path, label = spec.split(':', 1)
        else:
            path, label = spec, os.path.splitext(os.path.basename(spec))[0]
        runs.append((path, label))
    return runs


def run(args):
    runs = parse_runs(args.runs)

    print(f"Loading relation frequencies from {args.data_dir} ...")
    rel_freq = load_relation_frequencies(args.data_dir)

    edges = args.bucket_edges if args.bucket_edges else DEFAULT_EDGES
    names = args.bucket_names if args.bucket_names else DEFAULT_NAMES
    if args.bucket_edges and not args.bucket_names:
        names = [f"[{edges[i]}, {edges[i+1]})" for i in range(len(edges)-1)]
    n_buckets = len(names)

    # --- load all runs, determine num_rels (must agree across runs) ---
    loaded = []
    num_rels = None
    for path, label in runs:
        data = np.load(path)
        nr = int(data['num_rels'])
        if num_rels is None:
            num_rels = nr
        elif nr != num_rels:
            raise ValueError(
                f"num_rels mismatch: {path} has num_rels={nr}, "
                f"expected {num_rels} (from earlier run). All runs must "
                f"be evaluated on the same dataset/relation count.")
        loaded.append((label, data['rel_ids'], data['ranks']))

    bucket_of_rel = assign_relation_buckets(rel_freq, num_rels, edges, names)

    # Report bucket composition (how many relations fall in each bucket)
    print("\nRelation count per bucket (by training frequency):")
    for b, name in enumerate(names):
        rels_in_bucket = np.where(bucket_of_rel == b)[0]
        print(f"  {name.splitlines()[0]:18s}: "
              f"{len(rels_in_bucket)} relations "
              f"(e.g. ids {list(rels_in_bucket[:5])}"
              f"{'...' if len(rels_in_bucket) > 5 else ''})")

    # --- compute per-bucket metrics for each run ---
    results = {}  # results[label][bucket_idx] = metrics dict
    for label, rel_ids, ranks in loaded:
        bucket_per_triple = bucket_of_rel[rel_ids]
        per_bucket = []
        for b in range(n_buckets):
            mask = bucket_per_triple == b
            per_bucket.append(compute_metrics(ranks[mask]))
        results[label] = per_bucket

        print(f"\n{label}:")
        for b, name in enumerate(names):
            m = per_bucket[b]
            if m['n'] == 0:
                print(f"  {name.splitlines()[0]:18s}: n=0")
            else:
                print(f"  {name.splitlines()[0]:18s}: n={m['n']:6d}  "
                      f"MRR={m['MRR']:.4f}  H@1={m['H@1']:.4f}  "
                      f"H@3={m['H@3']:.4f}  H@10={m['H@10']:.4f}")

    # --- also compute overall (all buckets pooled) for reference ---
    print("\nOverall (all relations pooled):")
    for label, rel_ids, ranks in loaded:
        m = compute_metrics(ranks)
        print(f"  {label:18s}: n={m['n']:6d}  MRR={m['MRR']:.4f}  "
              f"H@1={m['H@1']:.4f}  H@3={m['H@3']:.4f}  H@10={m['H@10']:.4f}")

    # --- plot ---
    metrics_to_plot = ['MRR', 'H@1', 'H@3', 'H@10']
    fig, axes = plt.subplots(1, len(metrics_to_plot),
                              figsize=(4.2 * len(metrics_to_plot), 4.8),
                              sharey=False)

    n_runs = len(loaded)
    bar_width = 0.8 / n_runs
    x = np.arange(n_buckets)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_runs, 2)))

    for mi, metric in enumerate(metrics_to_plot):
        ax = axes[mi]
        for ri, (label, _, _) in enumerate(loaded):
            vals = [results[label][b][metric] for b in range(n_buckets)]
            offsets = x + (ri - (n_runs - 1) / 2) * bar_width
            bars = ax.bar(offsets, vals, width=bar_width,
                          label=label, color=colors[ri])
            # annotate bar with n for the first metric only (avoid clutter)
            if metric == 'MRR':
                for xi, b in zip(offsets, range(n_buckets)):
                    n = results[label][b]['n']
                    if n > 0:
                        ax.text(xi, 0.01, f"n={n}", rotation=90,
                                ha='center', va='bottom', fontsize=7,
                                color='black')
        ax.set_xticks(x)
        ax.set_xticklabels([name.split('\n')[0] for name in names],
                            rotation=20, ha='right', fontsize=9)
        # show frequency range as secondary label
        ax.set_title(metric)
        ax.grid(axis='y', alpha=0.25)
        ax.set_ylim(0, max(0.05,
                    max((results[label][b][metric] or 0)
                        for label, _, _ in loaded
                        for b in range(n_buckets)
                        if not np.isnan(results[label][b][metric])) * 1.15))

    axes[0].legend(fontsize=9, loc='upper right')
    axes[0].set_ylabel("score")

    fig.suptitle("Frequency-stratified evaluation: relations bucketed by "
                  "training frequency\n"
                  f"({os.path.basename(args.data_dir)}, "
                  f"buckets: {' / '.join(n.split(chr(10))[0] for n in names)})")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot 5: frequency-stratified MRR/Hits@k from "
                     "eval_stratified.py rank dumps")
    p.add_argument("--runs", nargs='+', required=True,
                    help="One or more 'path.npz:Label' specs (label "
                         "optional). E.g. ranks_base.npz:'Base LogCL' "
                         "ranks_full.npz:'Full pipeline'")
    p.add_argument("--data-dir", required=True,
                    help="Dataset directory containing train.txt "
                         "(for relation frequencies)")
    p.add_argument("--out", default="stratified.png")
    p.add_argument("--bucket-edges", nargs='+', type=float, default=None,
                    help="Frequency bucket edges, e.g. 0 10 50 200 1e9 "
                         "(len = n_buckets + 1). Default: "
                         "0 10 50 200 inf")
    p.add_argument("--bucket-names", nargs='+', type=str, default=None,
                    help="Names for each bucket (len = len(bucket_edges)-1). "
                         "Default: very_rare/rare/medium/frequent")
    args = p.parse_args()
    run(args)