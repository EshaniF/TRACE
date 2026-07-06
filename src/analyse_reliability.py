"""
Usage
-----
  python analyse_reliability.py \\
      --datasets ICEWS14 ICEWS18 GDELT \\
      --data-root ../data \\
      --min-edges 2

  # Add ICEWS05-15 once transition graph is available:
  python analyse_reliability.py \\
      --datasets ICEWS14 ICEWS18 ICEWS05-15 GDELT \\
      --data-root ../data
"""

import os
import re
import math
import pickle
import argparse
import statistics
from collections import Counter, defaultdict

import numpy as np

def load_graph(data_root, dataset):
    path = os.path.join(data_root, dataset, "rel_transition_graph.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_train_triples(data_root, dataset):
    """Load train.txt as list of (s, r, o, t) integer tuples."""
    path = os.path.join(data_root, dataset, "train.txt")
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    triples.append(tuple(int(p) for p in parts[:4]))
                except ValueError:
                    pass
    return triples


def compute_graph_quality(graph, min_edges=2):
    id2rel     = graph.get("id2rel", {})
    succ_edges = graph.get("successors", {})
    inh_edges  = graph.get("inhibitors",  {})
    num_rels   = len(id2rel)

    if num_rels == 0:
        return {}

    # Edge counts per relation
    succ_counts = [len(succ_edges.get(r, [])) for r in id2rel]
    inh_counts  = [len(inh_edges.get(r, []))  for r in id2rel]

    # Coverage: fraction with >= min_edges in each direction
    succ_covered = sum(1 for c in succ_counts if c >= min_edges)
    inh_covered  = sum(1 for c in inh_counts  if c >= min_edges)
    both_covered = sum(
        1 for r in id2rel
        if len(succ_edges.get(r, [])) >= min_edges
        and len(inh_edges.get(r, []))  >= min_edges)

    # All edge weights
    succ_weights = [w for entries in succ_edges.values() for _, w in entries]
    inh_weights  = [w for entries in inh_edges.values()  for _, w in entries]
    all_weights  = succ_weights + inh_weights

    def wstats(ws):
        if not ws:
            return dict(mean=0, std=0, median=0, min=0, max=0, var=0)
        return dict(
            mean   = float(np.mean(ws)),
            std    = float(np.std(ws)),
            median = float(np.median(ws)),
            min    = float(np.min(ws)),
            max    = float(np.max(ws)),
            var    = float(np.var(ws)),
        )

    succ_stats = wstats(succ_weights)
    inh_stats  = wstats(inh_weights)
    all_stats  = wstats(all_weights)

    # Inhibitor-to-successor edge ratio
    n_succ = sum(succ_counts)
    n_inh  = sum(inh_counts)
    inh_to_succ_ratio = n_inh / max(n_succ, 1)

    # Empirical validation: fraction of succ edges where LLM assigned
    # high confidence (weight > 0.5). Reflects score differentiation.
    high_conf_succ = sum(1 for w in succ_weights if w > 0.5) / max(len(succ_weights), 1)
    high_conf_inh  = sum(1 for w in inh_weights  if w > 0.5) / max(len(inh_weights),  1)

    return dict(
        num_rels        = num_rels,
        total_succ      = n_succ,
        total_inh       = n_inh,
        succ_per_rel    = n_succ / num_rels,
        inh_per_rel     = n_inh  / num_rels,
        succ_cov_pct    = 100 * succ_covered / num_rels,
        inh_cov_pct     = 100 * inh_covered  / num_rels,
        both_cov_pct    = 100 * both_covered  / num_rels,
        inh_succ_ratio  = inh_to_succ_ratio,
        high_conf_succ  = high_conf_succ,
        high_conf_inh   = high_conf_inh,
        succ_mean_w     = succ_stats["mean"],
        succ_std_w      = succ_stats["std"],
        succ_var_w      = succ_stats["var"],
        inh_mean_w      = inh_stats["mean"],
        inh_std_w       = inh_stats["std"],
        inh_var_w       = inh_stats["var"],
        all_mean_w      = all_stats["mean"],
        all_std_w       = all_stats["std"],
        all_var_w       = all_stats["var"],
    )


def print_quality_table(results):
    print("\n" + "="*70)
    print("ANALYSIS A — TRANSITION GRAPH QUALITY")
    print("="*70)

    # Console table
    cols = ["Dataset", "Rels", "Succ", "Inh",
            "Succ/rel", "Inh/rel",
            "SucCov%", "InhCov%", "BothCov%",
            "I/S ratio",
            "Succ μ±σ", "Inh μ±σ",
            "HC-S%", "HC-I%"]
    hdr = f"  {'Dataset':<12} {'Rels':>5} {'Succ':>6} {'Inh':>6} " \
          f"{'S/R':>6} {'I/R':>6} " \
          f"{'SCov%':>7} {'ICov%':>7} {'BCov%':>7} " \
          f"{'I/S':>6} " \
          f"{'Succ μ±σ':>14} {'Inh μ±σ':>14} " \
          f"{'HC-S%':>7} {'HC-I%':>7}"
    print(hdr)
    print("  " + "-"*(len(hdr)-2))

    for ds, q in results.items():
        if not q:
            print(f"  {ds:<12} — no graph")
            continue
        print(
            f"  {ds:<12} "
            f"{q['num_rels']:>5d} "
            f"{q['total_succ']:>6d} "
            f"{q['total_inh']:>6d} "
            f"{q['succ_per_rel']:>6.2f} "
            f"{q['inh_per_rel']:>6.2f} "
            f"{q['succ_cov_pct']:>6.1f}% "
            f"{q['inh_cov_pct']:>6.1f}% "
            f"{q['both_cov_pct']:>6.1f}% "
            f"{q['inh_succ_ratio']:>6.2f} "
            f"{q['succ_mean_w']:>6.4f}±{q['succ_std_w']:.4f} "
            f"{q['inh_mean_w']:>6.4f}±{q['inh_std_w']:.4f} "
            f"{100*q['high_conf_succ']:>6.1f}% "
            f"{100*q['high_conf_inh']:>6.1f}%"
        )


    print("\n\n  --- LaTeX table ---\n")
    print(r"  \begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Transition graph quality across datasets. "
          r"Cov = \% relations with $\geq 2$ edges. "
          r"$\mu\pm\sigma$ = mean $\pm$ std of edge weights. "
          r"HC = \% edges with weight $>0.5$ (high-confidence). "
          r"I/S = inhibitor-to-successor edge ratio.}")
    print(r"  \label{tab:graph_quality}")
    print(r"  \resizebox{\columnwidth}{!}{")
    print(r"  \begin{tabular}{lrrrrrrrrrrr}")
    print(r"  \toprule")
    print(r"  Dataset & Rels & Succ & Inh & S/Rel & I/Rel & "
          r"SCov\% & ICov\% & BCov\% & I/S & HC-S\% & HC-I\% \\")
    print(r"  \midrule")
    for ds, q in results.items():
        if not q:
            continue
        print(
            f"  {ds} & "
            f"{q['num_rels']} & "
            f"{q['total_succ']} & "
            f"{q['total_inh']} & "
            f"{q['succ_per_rel']:.2f} & "
            f"{q['inh_per_rel']:.2f} & "
            f"{q['succ_cov_pct']:.1f} & "
            f"{q['inh_cov_pct']:.1f} & "
            f"{q['both_cov_pct']:.1f} & "
            f"{q['inh_succ_ratio']:.2f} & "
            f"{100*q['high_conf_succ']:.1f} & "
            f"{100*q['high_conf_inh']:.1f} \\\\"
        )
    print(r"  \bottomrule")
    print(r"  \end{tabular}}")
    print(r"  \end{table}")



# Analysis B — Semantic Specificity


def compute_semantic_specificity(id2rel):

    lengths       = []
    unique_ratios = []
    all_tokens    = []

    for r_id, name in id2rel.items():
        # Clean and tokenise
        clean = name.replace('_', ' ').lower().strip()
        tokens = [t for t in re.split(r'\W+', clean) if t]
        if not tokens:
            tokens = [clean]

        lengths.append(len(tokens))
        unique_ratios.append(len(set(tokens)) / len(tokens))
        all_tokens.extend(tokens)

    if not lengths:
        return {}

    vocab_richness = len(set(all_tokens)) / max(len(all_tokens), 1)

    return dict(
        lengths        = lengths,
        mean_len       = float(np.mean(lengths)),
        std_len        = float(np.std(lengths)),
        mean_uniq_ratio= float(np.mean(unique_ratios)),
        long_name_pct  = 100 * sum(1 for l in lengths if l >= 4) / len(lengths),
        vocab_richness = vocab_richness,
        # Composite specificity score: normalised average of the four signals.
        # Each is bounded [0,1] (or will be normalised later across datasets).
        # Stored raw here; normalisation happens across datasets in the caller.
        specificity_raw= float(np.mean(lengths)) * float(np.mean(unique_ratios))
                         * (1 + vocab_richness),
    )


def compute_reliability_score(quality):

    if not quality:
        return 0.0

    # Weight variance: higher = more discriminative scores
    # Normalised against a reference of 0.05 (empirically reasonable max)
    var_score = min(quality["all_var_w"] / 0.05, 1.0)

    # Both-direction coverage normalised (100% = 1.0)
    cov_score = quality["both_cov_pct"] / 100.0

    # High-confidence inhibitor edges: measures LLM inhibitor knowledge
    hc_inh_score = quality["high_conf_inh"]

    # Inhibitor-to-successor ratio: a balanced graph (ratio near 1)
    # scores highest; heavily imbalanced graphs score lower.
    # ratio in [0, inf] → transform to [0, 1] via min(ratio, 1/ratio)
    r = quality["inh_succ_ratio"]
    balance_score = min(r, 1.0 / max(r, 1e-9)) if r > 0 else 0.0

    return (var_score + cov_score + hc_inh_score + balance_score) / 4.0


def print_specificity_table(spec_results, quality_results):
    """Print semantic specificity vs reliability analysis."""
    print("\n" + "="*70)
    print("ANALYSIS B — SEMANTIC SPECIFICITY vs TRANSITION GRAPH RELIABILITY")
    print("="*70)

    datasets = [ds for ds in spec_results if spec_results[ds]]

    # Normalise raw specificity scores across datasets to [0, 1]
    raw_scores = [spec_results[ds]["specificity_raw"] for ds in datasets]
    min_s = min(raw_scores) if raw_scores else 1.0
    max_s = max(raw_scores) if raw_scores else 1.0
    range_s = max(max_s - min_s, 1e-9)
    norm_spec = {ds: (spec_results[ds]["specificity_raw"] - min_s) / range_s
                 for ds in datasets}

    reliability = {ds: compute_reliability_score(quality_results.get(ds, {}))
                   for ds in datasets}

    print(f"\n  {'Dataset':<14} {'MeanLen':>8} {'UniqR':>7} {'LongN%':>8} "
          f"{'VocabR':>8} {'Spec(norm)':>11} {'Reliability':>12}")
    print("  " + "-"*72)
    for ds in datasets:
        sp = spec_results[ds]
        print(
            f"  {ds:<14} "
            f"{sp['mean_len']:>8.2f} "
            f"{sp['mean_uniq_ratio']:>7.3f} "
            f"{sp['long_name_pct']:>7.1f}% "
            f"{sp['vocab_richness']:>8.4f} "
            f"{norm_spec[ds]:>11.4f} "
            f"{reliability[ds]:>12.4f}"
        )

    # Correlation
    if len(datasets) >= 3 and HAS_SCIPY:
        spec_vals = [norm_spec[ds]   for ds in datasets]
        rel_vals  = [reliability[ds] for ds in datasets]
        pearson_r,  pearson_p  = scipy_stats.pearsonr(spec_vals, rel_vals)
        spearman_r, spearman_p = scipy_stats.spearmanr(spec_vals, rel_vals)
        print(f"\n  Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4f})")
        print(f"  Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.4f})")
        if abs(pearson_r) > 0.7:
            direction = "positive" if pearson_r > 0 else "negative"
            print(f"\n  FINDING: Strong {direction} correlation between semantic "
                  f"specificity and transition graph reliability (r={pearson_r:.2f}).")
            print(f"  This supports the hypothesis that LLM transition knowledge "
                  f"is more reliable for semantically specific relation vocabularies.")
        else:
            print(f"\n  NOTE: Correlation is moderate (r={pearson_r:.2f}). "
                  f"More datasets needed for a stronger claim.")
    elif len(datasets) < 3:
        print("\n  NOTE: At least 3 datasets needed for correlation analysis.")

    # LaTeX table
    print("\n\n  --- LaTeX table (paste into paper) ---\n")
    print(r"  \begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Semantic specificity of relation vocabularies and "
          r"LLM transition graph reliability per dataset. "
          r"MeanLen = mean relation name token count. "
          r"UniqR = mean unique token ratio. "
          r"LongN\% = \% names with $\geq 4$ tokens. "
          r"VocabR = vocabulary richness. "
          r"Spec = composite specificity score (normalised). "
          r"Reliability = composite graph reliability score.}")
    print(r"  \label{tab:specificity}")
    print(r"  \begin{tabular}{lrrrrrrr}")
    print(r"  \toprule")
    print(r"  Dataset & MeanLen & UniqR & LongN\% & VocabR & Spec & Reliability \\")
    print(r"  \midrule")
    for ds in datasets:
        sp = spec_results[ds]
        print(
            f"  {ds} & "
            f"{sp['mean_len']:.2f} & "
            f"{sp['mean_uniq_ratio']:.3f} & "
            f"{sp['long_name_pct']:.1f} & "
            f"{sp['vocab_richness']:.4f} & "
            f"{norm_spec[ds]:.4f} & "
            f"{reliability[ds]:.4f} \\\\"
        )
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \end{table}")


def print_inhibitor_gap_analysis(quality_results, train_data_per_ds,
                                  graphs):

    print("\n" + "="*70)
    print("ANALYSIS B (supp) — LLM INHIBITOR KNOWLEDGE GAP")
    print("="*70)
    print("  Hypothesis: LLMs encode event sequences (successors) better")
    print("  than event suppressions (inhibitors) because training corpora")
    print("  contain event sequences explicitly but inhibitions only implicitly.")
    print()
    print(f"  {'Dataset':<14} {'EmpSupp-S%':>12} {'EmpSupp-I%':>12} "
          f"{'Gap':>8} {'Interpretation'}")
    print("  " + "-"*72)

    gap_data = {}
    for ds, triples in train_data_per_ds.items():
        graph = graphs.get(ds)
        if not graph or not triples:
            continue

        # Build empirical co-occurrence within window=3
        subj_events = defaultdict(list)
        for s, r, o, t in triples:
            subj_events[s].append((t, r))

        emp_counts = defaultdict(Counter)
        for s, events in subj_events.items():
            events_sorted = sorted(events, key=lambda x: x[0])
            for i, (t1, r1) in enumerate(events_sorted):
                for t2, r2 in events_sorted[i+1:]:
                    if t2 - t1 > 3:
                        break
                    if r1 != r2:
                        emp_counts[r1][r2] += 1

        succ_edges = graph.get("successors", {})
        inh_edges  = graph.get("inhibitors",  {})

        # For each LLM-assigned edge, check empirical support
        succ_zero = sum(
            1 for r_id, entries in succ_edges.items()
            for r2, _ in entries
            if emp_counts[r_id].get(r2, 0) == 0)
        succ_total = sum(len(v) for v in succ_edges.values())

        inh_zero = sum(
            1 for r_id, entries in inh_edges.items()
            for r2, _ in entries
            if emp_counts[r_id].get(r2, 0) == 0)
        inh_total = sum(len(v) for v in inh_edges.values())

        succ_zero_pct = 100 * succ_zero / max(succ_total, 1)
        inh_zero_pct  = 100 * inh_zero  / max(inh_total,  1)
        gap           = inh_zero_pct - succ_zero_pct

        gap_data[ds] = dict(
            succ_zero_pct=succ_zero_pct,
            inh_zero_pct=inh_zero_pct,
            gap=gap)

        interp = ("inh >> succ: LLM inhibitor knowledge is mostly counterfactual"
                  if gap > 20 else
                  "inh > succ: moderate gap" if gap > 5 else
                  "similar: LLM has balanced knowledge")
        print(
            f"  {ds:<14} "
            f"{succ_zero_pct:>11.1f}% "
            f"{inh_zero_pct:>11.1f}% "
            f"{gap:>7.1f}% "
            f"  {interp}"
        )

    # LaTeX
    print("\n\n  --- LaTeX table ---\n")
    print(r"  \begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Fraction of LLM-assigned transition edges with zero "
          r"empirical co-occurrence in training data (within a 3-step window). "
          r"EmpSupp-S = zero-support rate for successor edges. "
          r"EmpSupp-I = zero-support rate for inhibitor edges. "
          r"Gap = EmpSupp-I $-$ EmpSupp-S. "
          r"A large positive gap indicates LLM inhibitor knowledge is "
          r"primarily counterfactual (not observed in the data).}")
    print(r"  \label{tab:inhibitor_gap}")
    print(r"  \begin{tabular}{lrrr}")
    print(r"  \toprule")
    print(r"  Dataset & EmpSupp-S\% & EmpSupp-I\% & Gap \\")
    print(r"  \midrule")
    for ds, d in gap_data.items():
        print(f"  {ds} & {d['succ_zero_pct']:.1f} & "
              f"{d['inh_zero_pct']:.1f} & {d['gap']:.1f} \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \end{table}")

    return gap_data

def make_plots(spec_results, quality_results, gap_data, datasets, out_path):
    if not HAS_MPL:
        print("\n  [Plots skipped — matplotlib not available]")
        return

    valid_ds = [ds for ds in datasets
                if spec_results.get(ds) and quality_results.get(ds)]
    if len(valid_ds) < 2:
        print("\n  [Plots skipped — need at least 2 datasets]")
        return

    # Normalise specificity
    raw_scores = [spec_results[ds]["specificity_raw"] for ds in valid_ds]
    min_s, max_s = min(raw_scores), max(raw_scores)
    range_s = max(max_s - min_s, 1e-9)
    norm_spec   = [(spec_results[ds]["specificity_raw"] - min_s) / range_s
                   for ds in valid_ds]
    reliability = [compute_reliability_score(quality_results[ds])
                   for ds in valid_ds]

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(valid_ds)))

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # ── Panel 1: Specificity vs Reliability scatter ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, ds in enumerate(valid_ds):
        ax1.scatter(norm_spec[i], reliability[i],
                    color=colors[i], s=120, zorder=3, label=ds)
        ax1.annotate(ds, (norm_spec[i], reliability[i]),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=9)
    if len(valid_ds) >= 3 and HAS_SCIPY:
        m, b, r, p, _ = scipy_stats.linregress(norm_spec, reliability)
        xs = np.linspace(min(norm_spec), max(norm_spec), 50)
        ax1.plot(xs, m*xs+b, '--', color='grey', linewidth=1.2,
                 label=f'r={r:.2f} (p={p:.2f})')
    ax1.set_xlabel("Semantic Specificity (normalised)", fontsize=10)
    ax1.set_ylabel("Transition Graph Reliability", fontsize=10)
    ax1.set_title("(a) Specificity vs Reliability", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Mean relation name length per dataset ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for i, ds in enumerate(valid_ds):
        lengths = spec_results[ds]["lengths"]
        ax2.bar(i, np.mean(lengths), color=colors[i],
                yerr=np.std(lengths), capsize=4, label=ds)
    ax2.set_xticks(range(len(valid_ds)))
    ax2.set_xticklabels(valid_ds, rotation=15, ha='right', fontsize=9)
    ax2.set_ylabel("Mean relation name length (tokens)", fontsize=10)
    ax2.set_title("(b) Relation Name Length by Dataset", fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: Weight distribution per dataset (box plot) ───────────────
    ax3 = fig.add_subplot(gs[1, 0])
    succ_weight_lists = []
    inh_weight_lists  = []
    box_labels = []
    for ds in valid_ds:
        graph = None
        # Reload graph data for weights — stored in quality_results metrics
        # Use weight statistics to reconstruct approximate distribution via
        # a normal approximation for display (exact weights not cached here)
        # Instead, use the stored mean/std to show via error bar plot
        succ_weight_lists.append(quality_results[ds]["succ_mean_w"])
        inh_weight_lists.append(quality_results[ds]["inh_mean_w"])
        box_labels.append(ds)

    x = np.arange(len(valid_ds))
    width = 0.35
    bars1 = ax3.bar(x - width/2,
                    [quality_results[ds]["succ_mean_w"] for ds in valid_ds],
                    width, label='Successors',
                    yerr=[quality_results[ds]["succ_std_w"] for ds in valid_ds],
                    capsize=4, color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2,
                    [quality_results[ds]["inh_mean_w"] for ds in valid_ds],
                    width, label='Inhibitors',
                    yerr=[quality_results[ds]["inh_std_w"] for ds in valid_ds],
                    capsize=4, color='coral', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(valid_ds, rotation=15, ha='right', fontsize=9)
    ax3.set_ylabel("Edge weight (μ ± σ)", fontsize=10)
    ax3.set_title("(c) Transition Edge Weight Distribution", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # ── Panel 4: Inhibitor knowledge gap ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    gap_ds = [ds for ds in valid_ds if ds in gap_data]
    if gap_ds:
        x4 = np.arange(len(gap_ds))
        succ_zeros = [gap_data[ds]["succ_zero_pct"] for ds in gap_ds]
        inh_zeros  = [gap_data[ds]["inh_zero_pct"]  for ds in gap_ds]
        ax4.bar(x4 - width/2, succ_zeros, width,
                label='Successors', color='steelblue', alpha=0.8)
        ax4.bar(x4 + width/2, inh_zeros, width,
                label='Inhibitors', color='coral', alpha=0.8)
        ax4.set_xticks(x4)
        ax4.set_xticklabels(gap_ds, rotation=15, ha='right', fontsize=9)
        ax4.set_ylabel("Zero empirical support (%)", fontsize=10)
        ax4.set_title("(d) LLM Inhibitor Knowledge Gap", fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, "No gap data\n(train.txt required)",
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("(d) LLM Inhibitor Knowledge Gap", fontsize=11)

    fig.suptitle(
        "Transition Graph Quality and Semantic Specificity Analysis",
        fontsize=13, fontweight='bold')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved to: {out_path}")
    plt.close(fig)


def save_stats(spec_results, quality_results, gap_data, datasets, out_path):
    lines = ["TRANSITION GRAPH RELIABILITY ANALYSIS\n",
             "="*60 + "\n\n"]

    for ds in datasets:
        lines.append(f"Dataset: {ds}\n")
        lines.append("-"*40 + "\n")
        if spec_results.get(ds):
            sp = spec_results[ds]
            lines.append(f"  mean_name_length    : {sp['mean_len']:.3f}\n")
            lines.append(f"  unique_token_ratio  : {sp['mean_uniq_ratio']:.4f}\n")
            lines.append(f"  long_name_pct       : {sp['long_name_pct']:.2f}%\n")
            lines.append(f"  vocab_richness      : {sp['vocab_richness']:.4f}\n")
            lines.append(f"  specificity_raw     : {sp['specificity_raw']:.4f}\n")
        if quality_results.get(ds):
            q = quality_results[ds]
            for k, v in q.items():
                lines.append(f"  {k:<22}: {v}\n")
        if ds in gap_data:
            g = gap_data[ds]
            lines.append(f"  succ_zero_emp_pct   : {g['succ_zero_pct']:.2f}%\n")
            lines.append(f"  inh_zero_emp_pct    : {g['inh_zero_pct']:.2f}%\n")
            lines.append(f"  inhibitor_gap       : {g['gap']:.2f}%\n")
        lines.append("\n")

    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"  Statistics saved to: {out_path}")



def main():
    p = argparse.ArgumentParser(
        description="Reliability vs semantic specificity analysis "
                    "for TKG transition graphs.")
    p.add_argument("--datasets",    nargs="+",
                   default=["ICEWS14", "ICEWS18", "GDELT"],
                   help="Dataset names to analyse.")
    p.add_argument("--data-root",   default="../data",
                   help="Root directory containing per-dataset folders.")
    p.add_argument("--min-edges",   type=int, default=2,
                   help="Min edges per direction to count as covered.")
    p.add_argument("--out-dir",     default=".",
                   help="Directory for output plot and stats files.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    graphs        = {}
    spec_results  = {}
    quality_results = {}
    train_data_per_ds = {}

    print(f"\nLoading data for: {args.datasets}\n")

    for ds in args.datasets:
        print(f"  [{ds}]")
        graph = load_graph(args.data_root, ds)
        if graph is None:
            print(f"    No transition graph found — skipping.")
            continue
        graphs[ds] = graph

        id2rel = graph.get("id2rel", {})
        print(f"    Relations: {len(id2rel)}")

        spec_results[ds]    = compute_semantic_specificity(id2rel)
        quality_results[ds] = compute_graph_quality(graph, args.min_edges)

        triples = load_train_triples(args.data_root, ds)
        print(f"    Train triples: {len(triples)}")
        train_data_per_ds[ds] = triples

    if not graphs:
        print("\nNo graphs loaded. Check --data-root and dataset names.")
        return

    # ── Analysis A ────────────────────────────────────────────────────────
    print_quality_table(quality_results)

    # ── Analysis B ────────────────────────────────────────────────────────
    print_specificity_table(spec_results, quality_results)

    # ── Inhibitor gap ─────────────────────────────────────────────────────
    gap_data = print_inhibitor_gap_analysis(
        quality_results, train_data_per_ds, graphs)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_path = os.path.join(args.out_dir, "analyse_reliability_plots.png")
    make_plots(spec_results, quality_results, gap_data,
               args.datasets, plot_path)

    # ── Save stats ────────────────────────────────────────────────────────
    stats_path = os.path.join(args.out_dir, "analyse_reliability_stats.txt")
    save_stats(spec_results, quality_results, gap_data,
               args.datasets, stats_path)

    print("\nDone.")


if __name__ == "__main__":
    main()