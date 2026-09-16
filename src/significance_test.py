"""
significance_test.py
=====================
Paired per-query significance testing between two evaluation runs,
given per-query filtered-rank arrays saved via --save-ranks (see
save_ranks_patch.md).

MRR is tested with a Wilcoxon signed-rank test on paired reciprocal
ranks (1/rank), since reciprocal rank is continuous and the paired
design matches "same queries, two models."

Hits@k is tested with McNemar's test on paired binary outcomes
(rank <= k), since Hits@k is a binary win/loss per query and McNemar's
is the standard test for paired binary classification comparisons.

Also reports a bootstrap 95% CI for the MRR difference, since a CI that
excludes zero is useful complementary evidence alongside the p-value.

Usage
-----
  python significance_test.py \\
      --baseline ../result/ranks/icews14_base_ranks.npy \\
      --ours     ../result/ranks/icews14_full_ranks.npy \\
      --name-baseline "Base LogCL" \\
      --name-ours     "Full method"


"""

import argparse
import numpy as np
from scipy.stats import wilcoxon


def load_ranks(path: str) -> np.ndarray:
    ranks = np.load(path)
    if ranks.ndim != 1:
        raise ValueError(f"{path}: expected 1-D array, got shape {ranks.shape}")
    return ranks


def mcnemar_test(baseline_correct: np.ndarray,
                  ours_correct: np.ndarray) -> dict:


    # b: baseline correct, ours incorrect
    # c: baseline incorrect, ours correct
    b = int(np.sum(baseline_correct & ~ours_correct))
    c = int(np.sum(~baseline_correct & ours_correct))
    n = b + c

    if n == 0:
        # No discordant pairs at all — models agree on every query.
        return {"b": b, "c": c, "n_discordant": n, "p_value": 1.0,
                "method": "no_discordant_pairs"}

    try:
        from scipy.stats import binomtest
        # Two-sided exact binomial test: under H0, each discordant pair
        # is equally likely to favor either model, i.e. c ~ Binomial(n, 0.5).
        result = binomtest(c, n, p=0.5, alternative="two-sided")
        p_value = result.pvalue
        method = "exact_binomial"
    except ImportError:
        # Chi-square approximation with continuity correction (classic
        # McNemar's statistic), for older scipy versions.
        from scipy.stats import chi2
        stat = (abs(b - c) - 1) ** 2 / n
        p_value = 1 - chi2.cdf(stat, df=1)
        method = "chi_square_continuity_corrected"

    return {"b": b, "c": c, "n_discordant": n,
            "p_value": p_value, "method": method}


def bootstrap_mrr_diff_ci(baseline_rr: np.ndarray, ours_rr: np.ndarray,
                            n_boot: int = 10000, seed: int = 0,
                            ci: float = 0.95) -> dict:
    """
    Bootstrap 95% CI for delta_MRR = mean(ours_rr) - mean(baseline_rr),
    resampling queries (the paired unit) with replacement.
    """
    rng = np.random.default_rng(seed)
    n = len(baseline_rr)
    diffs = ours_rr - baseline_rr   # per-query paired difference

    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    alpha = 1.0 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta_mrr": float(diffs.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_level": ci, "n_boot": n_boot,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run(args):
    baseline_ranks = load_ranks(args.baseline)
    ours_ranks     = load_ranks(args.ours)

    if len(baseline_ranks) != len(ours_ranks):
        raise ValueError(
            f"Rank arrays have different lengths "
            f"({len(baseline_ranks)} vs {len(ours_ranks)}). ")

    n = len(baseline_ranks)
    print(f"Paired queries: {n}")
    print(f"  {args.name_baseline}: mean rank {baseline_ranks.mean():.2f}")
    print(f"  {args.name_ours}:     mean rank {ours_ranks.mean():.2f}")

    # ------------------------------------------------------------------
    # MRR: Wilcoxon signed-rank test on paired reciprocal ranks

    baseline_rr = 1.0 / baseline_ranks
    ours_rr     = 1.0 / ours_ranks

    mrr_baseline = baseline_rr.mean()
    mrr_ours     = ours_rr.mean()

    diffs = ours_rr - baseline_rr
    n_nonzero = int(np.sum(diffs != 0))

    print(f"\n=== MRR ===")
    print(f"  {args.name_baseline}: {mrr_baseline:.4f}")
    print(f"  {args.name_ours}:     {mrr_ours:.4f}")
    print(f"  Absolute delta:  {mrr_ours - mrr_baseline:+.4f}")
    print(f"  Relative delta:  {100 * (mrr_ours - mrr_baseline) / mrr_baseline:+.2f}%")

    if n_nonzero == 0:
        print("  Wilcoxon: skipped — all paired differences are exactly zero ")
    else:
        try:
            stat, p_value = wilcoxon(ours_rr, baseline_rr,
                                      alternative="two-sided",
                                      zero_method="wilcox")
            print(f"  Wilcoxon signed-rank: statistic={stat:.2f}, "
                  f"p={p_value:.4g}  "
                  f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)")
        except ValueError as e:
            print(f"  Wilcoxon: failed ({e}) ")

    boot = bootstrap_mrr_diff_ci(baseline_rr, ours_rr,
                                  n_boot=args.n_boot, seed=args.seed)
    print(f"  Bootstrap {int(boot['ci_level']*100)}% CI for delta_MRR: "
          f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]  "
          f"({'excludes zero' if boot['excludes_zero'] else 'includes zero'})")

    # ------------------------------------------------------------------
    # Hits@k: McNemar's test on paired binary outcomes
    # ------------------------------------------------------------------
    print(f"\n=== Hits@k ===")
    for k in args.hits_k:
        baseline_hit = baseline_ranks <= k
        ours_hit     = ours_ranks <= k

        h_baseline = baseline_hit.mean()
        h_ours     = ours_hit.mean()

        mc = mcnemar_test(baseline_hit, ours_hit)

        print(f"\n  Hits@{k}")
        print(f"    {args.name_baseline}: {h_baseline:.4f}")
        print(f"    {args.name_ours}:     {h_ours:.4f}")
        print(f"    Delta: {h_ours - h_baseline:+.4f}")
        print(f"    McNemar: b(baseline-only correct)={mc['b']}, "
              f"c(ours-only correct)={mc['c']}, "
              f"discordant pairs={mc['n_discordant']}")
        print(f"    p={mc['p_value']:.4g} ({mc['method']})  "
              f"({'significant' if mc['p_value'] < 0.05 else 'not significant'} at alpha=0.05)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Paired per-query significance testing (Wilcoxon "
                     "for MRR, McNemar for Hits@k) between two saved "
                     "per-query rank arrays.")
    p.add_argument("--baseline", type=str, required=True,
                   help="Path to baseline's saved ranks .npy")
    p.add_argument("--ours",     type=str, required=True,
                   help="Path to your method's saved ranks .npy")
    p.add_argument("--name-baseline", type=str, default="Baseline")
    p.add_argument("--name-ours",     type=str, default="Ours")
    p.add_argument("--hits-k",   type=int, nargs="+", default=[1, 3, 10],
                   help="Which Hits@k values to test (default 1 3 10).")
    p.add_argument("--n-boot",   type=int, default=10000,
                   help="Bootstrap resamples for the MRR CI (default 10000).")
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args()
    run(args)