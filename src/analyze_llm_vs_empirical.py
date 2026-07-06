"""
analyze_llm_vs_empirical.py
============================
PLOT 1 — LLM-assigned transition weights vs. empirical co-occurrence.

For every (anchor relation r, successor/inhibitor relation r2) edge in
the LLM-derived transition graph, this script:

  1. Reads the LLM weight from rel_transition_graph.pkl.
  2. Computes the empirical co-occurrence count for (r, r2) using
     compute_empirical_transitions() (same logic as
     llm_transition_scorer.py, window-based successor counting on
     train.txt).
  3. Normalises empirical counts to [0, 1] (dividing by the max count
     observed across all pairs).
  4. Produces:
       - a scatter plot: LLM weight (x) vs. empirical-normalised
         co-occurrence (y), separately for successor and inhibitor
         edges, with the Pearson correlation coefficient annotated.
       - prints the correlation coefficients to stdout.

Usage
-----
  python analyze_llm_vs_empirical.py \
      --graph ../data/ICEWS14/rel_transition_graph.pkl \
      --data-dir ../data/ICEWS14 \
      --out llm_vs_empirical_ICEWS14.png \
      --window 7
"""

import os
import sys
import pickle
import argparse
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Empirical transitions (mirrors llm_transition_scorer.compute_empirical_transitions)


def compute_empirical_transitions(data_dir: str, window: int = 7) -> dict:
    """
    Compute empirical successor frequencies from training data.
    Returns {r_id: {r_id: count}}.

    """
    path = os.path.join(data_dir, "train.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"train.txt not found in {data_dir}")

    subj_events = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    s, r, o, t = (int(parts[0]), int(parts[1]),
                                  int(parts[2]), int(parts[3]))
                    subj_events[s].append((t, r))
                except ValueError:
                    pass

    transitions = defaultdict(Counter)
    for s, events in subj_events.items():
        events_sorted = sorted(events, key=lambda x: x[0])
        for i, (t1, r1) in enumerate(events_sorted):
            for t2, r2 in events_sorted[i + 1:]:
                if t2 - t1 > window:
                    break
                if r1 != r2:
                    transitions[r1][r2] += 1

    return dict(transitions)


def collect_pairs(graph_data: dict, key: str) -> list:
    """
    Returns list of (r1, r2, llm_weight) for all edges in
    graph_data[key] = {r1: [(r2, weight), ...]}.
    """
    pairs = []
    for r1, entries in graph_data.get(key, {}).items():
        for r2, w in entries:
            pairs.append((r1, r2, w))
    return pairs


def normalise_empirical(pairs: list, empirical: dict) -> tuple:
    """
    For a list of (r1, r2, llm_weight), look up empirical[r1][r2],
    normalise by the global max empirical count, and return
    (llm_weights, empirical_norm, raw_counts) as numpy arrays.
    """
    all_counts = [cnt for d in empirical.values() for cnt in d.values()]
    max_count  = max(all_counts) if all_counts else 1

    llm_w, emp_norm, raw = [], [], []
    for r1, r2, w in pairs:
        cnt = empirical.get(r1, {}).get(r2, 0)
        llm_w.append(w)
        emp_norm.append(cnt / max_count)
        raw.append(cnt)

    return np.array(llm_w), np.array(emp_norm), np.array(raw)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def run(args):
    print(f"Loading transition graph from {args.graph} ...")
    with open(args.graph, "rb") as f:
        graph_data = pickle.load(f)

    id2rel = graph_data.get("id2rel", {})

    print(f"Computing empirical transitions from {args.data_dir} "
          f"(window={args.window}) ...")
    empirical = compute_empirical_transitions(args.data_dir, args.window)

    succ_pairs = collect_pairs(graph_data, "successors")
    inh_pairs  = collect_pairs(graph_data, "inhibitors")

    print(f"  Successor edges: {len(succ_pairs)}")
    print(f"  Inhibitor edges: {len(inh_pairs)}")

    succ_llm, succ_emp, succ_raw = normalise_empirical(succ_pairs, empirical)
    inh_llm,  inh_emp,  inh_raw  = normalise_empirical(inh_pairs,  empirical)

    r_succ = pearson(succ_llm, succ_emp)
    r_inh  = pearson(inh_llm,  inh_emp)

    print(f"\nPearson r (successor LLM weight vs. empirical-normalised "
          f"co-occurrence): {r_succ:.4f}")
    print(f"Pearson r (inhibitor LLM weight vs. empirical-normalised "
          f"co-occurrence): {r_inh:.4f}")


    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(succ_llm, succ_emp, alpha=0.4, s=18,
                    color='#2a6f97', edgecolors='none')
    axes[0].set_xlabel("LLM successor weight (normalised, 0-1)")
    axes[0].set_ylabel("Empirical co-occurrence (normalised)")
    axes[0].set_title(f"Successor edges (n={len(succ_pairs)})\n"
                       f"Pearson r = {r_succ:.3f}")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, max(0.05, succ_emp.max() * 1.05)
                     if len(succ_emp) else 1)
    axes[0].grid(alpha=0.25)

    axes[1].scatter(inh_llm, inh_emp, alpha=0.4, s=18,
                    color='#bb3e03', edgecolors='none')
    axes[1].set_xlabel("LLM inhibitor weight (normalised, 0-1)")
    axes[1].set_ylabel("Empirical co-occurrence (normalised)")
    axes[1].set_title(f"Inhibitor edges (n={len(inh_pairs)})\n"
                       f"Pearson r = {r_inh:.3f}")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, max(0.05, inh_emp.max() * 1.05)
                     if len(inh_emp) else 1)
    axes[1].grid(alpha=0.25)

    fig.suptitle(f"LLM-derived transition weights vs. empirical "
                  f"co-occurrence\n(dataset: {os.path.basename(args.data_dir)}, "
                  f"window={args.window})")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot 1: LLM transition weights vs. empirical co-occurrence")
    p.add_argument("--graph",    required=True,
                    help="Path to rel_transition_graph.pkl")
    p.add_argument("--data-dir", required=True,
                    help="Dataset directory containing train.txt")
    p.add_argument("--out",      default="llm_vs_empirical.png",
                    help="Output image path")
    p.add_argument("--window",   type=int, default=7,
                    help="Empirical transition window (default 7, "
                         "matches llm_transition_scorer.py default)")
    args = p.parse_args()
    run(args)