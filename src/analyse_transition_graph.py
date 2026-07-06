"""
analyse_transition_graph.py
============================
Reads rel_transition_graph.pkl and prints coverage statistics for
successors and inhibitors without re-running the LLM scorer.

Usage
-----
  python analyse_transition_graph.py -d ICEWS18
  python analyse_transition_graph.py -d ICEWS14 --data-root ../data
  python analyse_transition_graph.py -d GDELT   --top-k 5

"""

import os
import pickle
import argparse
import statistics
from collections import Counter

def load_graph(data_root: str, dataset: str) -> dict:
    path = os.path.join(data_root, dataset, "rel_transition_graph.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Transition graph not found at {path}\n"
            f"Run llm_transition_scorer.py -d {dataset} first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def subsection(title: str):
    print(f"\n  --- {title} ---")


def histogram(counts: list, bins: list, labels: list):
    """
    Print a simple text histogram.
    counts : list of ints (one per relation)
    bins   : list of (low, high) inclusive ranges
    labels : matching human-readable labels
    """
    total = len(counts)
    for (lo, hi), label in zip(bins, labels):
        if hi is None:
            n = sum(1 for c in counts if c >= lo)
        else:
            n = sum(1 for c in counts if lo <= c <= hi)
        bar = "█" * min(40, int(40 * n / max(total, 1)))
        print(f"    {label:12s} {n:4d} ({n/total*100:5.1f}%)  {bar}")


def weight_stats(edges: dict, id2rel: dict) -> None:
    """Print weight distribution across all edges."""
    all_weights = []
    for r_id, entries in edges.items():
        for _, w in entries:
            all_weights.append(w)

    if not all_weights:
        print("    No edges found.")
        return

    all_weights_sorted = sorted(all_weights)
    n   = len(all_weights_sorted)
    mn  = all_weights_sorted[0]
    mx  = all_weights_sorted[-1]
    med = all_weights_sorted[n // 2]
    avg = sum(all_weights_sorted) / n

    print(f"    Total edge weights analysed : {n}")
    print(f"    Min    : {mn:.4f}")
    print(f"    Mean   : {avg:.4f}")
    print(f"    Median : {med:.4f}")
    print(f"    Max    : {mx:.4f}")

    # Weight band histogram
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print()
    print("    Weight distribution:")
    for lo, hi in bands:
        n_band = sum(1 for w in all_weights if lo <= w < hi)
        bar    = "█" * min(30, int(30 * n_band / max(n, 1)))
        print(f"      [{lo:.1f},{hi:.1f})  {n_band:4d} ({n_band/n*100:5.1f}%)  {bar}")


def top_relations(edges: dict, id2rel: dict,
                  n: int = 10, most: bool = True) -> None:
    """Print the n most or least connected relations."""
    counts = {r_id: len(entries) for r_id, entries in edges.items()
              if len(entries) > 0}
    ranked = sorted(counts.items(), key=lambda x: x[1],
                    reverse=most)[:n]
    label  = "Most" if most else "Least (non-zero)"
    print(f"\n    {label} connected relations:")
    for r_id, cnt in ranked:
        name = id2rel.get(r_id, str(r_id))
        print(f"      [{r_id:3d}] {cnt:2d} edges  {name}")


def analyse_direction(edges: dict,
                      id2rel: dict,
                      num_rels: int,
                      direction: str) -> dict:
    """
    Full analysis for one direction (successors or inhibitors).
    Returns a summary dict for the cross-comparison table.
    """
    section(f"{direction.upper()}")

    all_rel_ids = set(id2rel.keys())

    # Count edges per relation (0 if not in edges dict)
    edge_counts = {r_id: len(edges.get(r_id, [])) for r_id in all_rel_ids}
    counts_list = list(edge_counts.values())

    total_edges    = sum(counts_list)
    n_zero         = sum(1 for c in counts_list if c == 0)
    n_one          = sum(1 for c in counts_list if c == 1)
    n_two_four     = sum(1 for c in counts_list if 2 <= c <= 4)
    n_five_plus    = sum(1 for c in counts_list if c >= 5)
    n_covered      = num_rels - n_zero
    pct_covered    = 100 * n_covered / num_rels
    pct_multi      = 100 * (n_covered - n_one) / num_rels

    # ── Summary counts ──────────────────────────────────────────────────
    subsection("Coverage summary")
    print(f"    Total relations           : {num_rels}")
    print(f"    Total edges               : {total_edges}")
    print(f"    Avg edges per relation    : {total_edges/num_rels:.2f}")
    print(f"    Relations with 0 edges    : {n_zero:4d}  "
          f"({100*n_zero/num_rels:.1f}%)")
    print(f"    Relations with 1 edge     : {n_one:4d}  "
          f"({100*n_one/num_rels:.1f}%)")
    print(f"    Relations with 2-4 edges  : {n_two_four:4d}  "
          f"({100*n_two_four/num_rels:.1f}%)")
    print(f"    Relations with 5+ edges   : {n_five_plus:4d}  "
          f"({100*n_five_plus/num_rels:.1f}%)")
    print(f"    Coverage (≥1 edge)        : {n_covered:4d}  "
          f"({pct_covered:.1f}%)")
    print(f"    Multi-edge (>1 edge)      : {n_covered-n_one:4d}  "
          f"({pct_multi:.1f}%)")

    # ── Histogram ───────────────────────────────────────────────────────
    subsection("Edge count distribution")
    histogram(
        counts_list,
        bins   = [(0,0), (1,1), (2,2), (3,3), (4,4), (5, None)],
        labels = ["0 edges", "1 edge", "2 edges",
                  "3 edges", "4 edges", "5+ edges"])

    # ── Weight statistics ────────────────────────────────────────────────
    subsection("Weight statistics")
    weight_stats(edges, id2rel)

    # ── Most and least connected ─────────────────────────────────────────
    subsection("Top-10 most connected relations")
    top_relations(edges, id2rel, n=10, most=True)

    subsection("Top-10 least connected (non-zero) relations")
    top_relations(edges, id2rel, n=10, most=False)

    # ── Uncovered relations ──────────────────────────────────────────────
    zero_ids = [r_id for r_id, c in edge_counts.items() if c == 0]
    subsection(f"Relations with NO {direction} ({len(zero_ids)} total)")
    if zero_ids:
        for r_id in sorted(zero_ids):
            name = id2rel.get(r_id, str(r_id))
            print(f"      [{r_id:3d}] {name}")
    else:
        print("      None — all relations have at least one edge.")

    return {
        "total_edges":   total_edges,
        "n_zero":        n_zero,
        "n_one":         n_one,
        "n_two_four":    n_two_four,
        "n_five_plus":   n_five_plus,
        "pct_covered":   pct_covered,
        "pct_multi":     pct_multi,
    }


def cross_analysis(succ_edges: dict, inh_edges: dict,
                   id2rel: dict) -> None:
    """
    Find relations that have successors but no inhibitors, vice versa,
    and relations that have neither.
    """
    section("CROSS-DIRECTION ANALYSIS")

    all_rel_ids = set(id2rel.keys())
    has_succ = {r for r in all_rel_ids if len(succ_edges.get(r, [])) > 0}
    has_inh  = {r for r in all_rel_ids if len(inh_edges.get(r, [])) > 0}

    both      = has_succ & has_inh
    succ_only = has_succ - has_inh
    inh_only  = has_inh  - has_succ
    neither   = all_rel_ids - has_succ - has_inh

    n = len(all_rel_ids)
    print(f"\n    Both successors AND inhibitors : {len(both):4d}  "
          f"({100*len(both)/n:.1f}%)")
    print(f"    Successors ONLY               : {len(succ_only):4d}  "
          f"({100*len(succ_only)/n:.1f}%)")
    print(f"    Inhibitors ONLY               : {len(inh_only):4d}  "
          f"({100*len(inh_only)/n:.1f}%)")
    print(f"    NEITHER (completely uncovered): {len(neither):4d}  "
          f"({100*len(neither)/n:.1f}%)")

    if neither:
        subsection("Completely uncovered relations (no successors AND no inhibitors)")
        for r_id in sorted(neither):
            name = id2rel.get(r_id, str(r_id))
            print(f"      [{r_id:3d}] {name}")


def summary_table(succ_stats: dict, inh_stats: dict) -> None:
    """Print a compact side-by-side comparison table."""
    section("SUMMARY TABLE")
    print()
    print(f"  {'Metric':<30} {'Successors':>12} {'Inhibitors':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    rows = [
        ("Total edges",        "total_edges",  "d"),
        ("Relations with 0",   "n_zero",       "d"),
        ("Relations with 1",   "n_one",        "d"),
        ("Relations with 2-4", "n_two_four",   "d"),
        ("Relations with 5+",  "n_five_plus",  "d"),
        ("Coverage % (≥1)",    "pct_covered",  ".1f"),
        ("Multi-edge % (>1)",  "pct_multi",    ".1f"),
    ]
    for label, key, fmt in rows:
        sv = succ_stats[key]
        iv = inh_stats[key]
        if fmt == "d":
            print(f"  {label:<30} {sv:>12d} {iv:>12d}")
        else:
            print(f"  {label:<30} {sv:>11.1f}% {iv:>11.1f}%")


def main():
    p = argparse.ArgumentParser(
        description="Analyse transition graph coverage statistics.")
    p.add_argument("-d", "--dataset",  required=True)
    p.add_argument("--data-root",      default="../data")
    args = p.parse_args()

    print(f"\nLoading transition graph for {args.dataset}...")
    data = load_graph(args.data_root, args.dataset)

    succ_edges = data.get("successors", {})
    inh_edges  = data.get("inhibitors", {})
    id2rel     = data.get("id2rel", {})
    num_rels   = len(id2rel)

    print(f"  Relations in id2rel  : {num_rels}")
    print(f"  Relations with succs : {len(succ_edges)}")
    print(f"  Relations with inhs  : {len(inh_edges)}")

    succ_stats = analyse_direction(
        succ_edges, id2rel, num_rels, "successors")
    inh_stats  = analyse_direction(
        inh_edges,  id2rel, num_rels, "inhibitors")

    cross_analysis(succ_edges, inh_edges, id2rel)
    summary_table(succ_stats, inh_stats)
    print()


if __name__ == "__main__":
    main()