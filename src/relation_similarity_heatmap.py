"""
relation_similarity_heatmap.py
===============================
Plots a side-by-side "before / after" heatmap of pairwise cosine
similarity between relation embeddings.

Changes from original
---------------------
- Shared y-axis: y-axis labels appear only once (left panel), right
  panel suppresses them, saving space and avoiding duplication.
- Dual output: saves both a colour version (_color.png) and a
  black-and-white version (_bw.png) from a single run, so you can
  submit the colour PDF and the B&W printed version without re-running.
- B&W design: uses a perceptually uniform grey colormap ('Greys_r')
  for the B&W version. Successor cells are marked with '▲' text
  annotations and inhibitor cells with '▼', so the annotations are
  readable without colour. The colour version retains green/red outlines.
- Shared colourbar anchored to the right of both panels.

Usage
-----
    python relation_similarity_heatmap.py --demo --out heatmap.png

    python relation_similarity_heatmap.py \
        --ckpt-no-reg  ../models/lambda0.pt \
        --ckpt-with-reg ../models/lambda01.pt \
        --num-rels 230 \
        --transition-graph ../data/ICEWS14/rel_transition_graph.pkl \
        --anchor-relation "Make statement" \
        --top-k 2 \
        --out relation_similarity.png
"""

import argparse
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec

# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------

DEMO_LABELS = [
    "Express intent\nto cooperate",
    "Host visit",
    "Make statement",
    "Use military\nforce",
    "Provide aid",
    "Threaten",
    "Reduce relations",
]

DEMO_ANCHOR     = 2
DEMO_SUCCESSORS = [0, 1]
DEMO_INHIBITORS = [3, 5]

DEMO_BEFORE = np.array([
    [ 1.00,  0.12,  0.08, -0.05,  0.22, -0.10,  0.05],
    [ 0.12,  1.00,  0.10,  0.03,  0.15,  0.02, -0.08],
    [ 0.08,  0.10,  1.00,  0.05,  0.12, -0.02,  0.07],
    [-0.05,  0.03,  0.05,  1.00, -0.10,  0.18,  0.03],
    [ 0.22,  0.15,  0.12, -0.10,  1.00, -0.08,  0.10],
    [-0.10,  0.02, -0.02,  0.18, -0.08,  1.00,  0.06],
    [ 0.05, -0.08,  0.07,  0.03,  0.10,  0.06,  1.00],
])

DEMO_AFTER = np.array([
    [ 1.00,  0.12,  0.55, -0.05,  0.22, -0.10,  0.05],
    [ 0.12,  1.00,  0.50,  0.03,  0.15,  0.02, -0.08],
    [ 0.55,  0.50,  1.00, -0.45,  0.12, -0.40,  0.07],
    [-0.05,  0.03, -0.45,  1.00, -0.10,  0.18,  0.03],
    [ 0.22,  0.15,  0.12, -0.10,  1.00, -0.08,  0.10],
    [-0.10,  0.02, -0.40,  0.18, -0.08,  1.00,  0.06],
    [ 0.05, -0.08,  0.07,  0.03,  0.10,  0.06,  1.00],
])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_emb_rel(ckpt_path, num_rels):
    import torch
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    return state["emb_rel"][:num_rels].float().numpy()


def cosine_similarity_matrix(emb):
    norms  = np.linalg.norm(emb, axis=1, keepdims=True)
    normed = emb / np.clip(norms, 1e-8, None)
    return normed @ normed.T


def wrap_label(label, max_chars_per_line=10):
    """
    Wrap a label onto two lines (max) by breaking at the nearest space
    to the midpoint. If the label already contains '\\n', leave as-is.
    Single short words / labels under the threshold are left untouched.
    """
    if "\n" in label:
        return label
    if len(label) <= max_chars_per_line:
        return label

    words = label.split("_")
    if len(words) == 1:
        return label  # can't wrap a single word nicely

    # find split point closest to the midpoint character count
    best_idx, best_diff = 1, float("inf")
    total = len(label)
    running = 0
    for i in range(len(words) - 1):
        running += len(words[i]) + 1
        diff = abs(running - total / 2)
        if diff < best_diff:
            best_diff = diff
            best_idx = i + 1

    line1 = " ".join(words[:best_idx])
    line2 = " ".join(words[best_idx:])
    return f"{line1}\n{line2}"


def relations_for_anchor(transition_graph_path, anchor_name, top_k):
    with open(transition_graph_path, "rb") as f:
        data = pickle.load(f)
    id2rel = data["id2rel"]
    rel2id = data.get("rel2id") or {v: k for k, v in id2rel.items()}
    if anchor_name not in rel2id:
        raise ValueError(
            f"'{anchor_name}' not found. Examples: {list(rel2id)[:10]}")
    anchor_id = rel2id[anchor_name]
    succ = [r for r, _ in data.get("successors", {}).get(anchor_id, [])][:top_k]
    inh  = [r for r, _ in data.get("inhibitors", {}).get(anchor_id, [])][:top_k]
    rel_ids, seen = [], set()
    for r in [anchor_id] + succ + inh:
        if r not in seen:
            seen.add(r)
            rel_ids.append(r)
    return rel_ids, anchor_id, succ, inh, id2rel


# ---------------------------------------------------------------------------
# Core plot function — called twice (colour + B&W)
# ---------------------------------------------------------------------------

def _draw_panel(ax, sim, title, labels,
                anchor, successors, inhibitors,
                show_yticklabels, cmap, colour_mode):
    """Draw one heatmap panel.

    Both colour and B&W versions now use the SAME border convention:
      successor edges  -> solid thick border + up-triangle marker
      inhibitor edges  -> dotted thick border + down-triangle marker
    The only difference between modes is the colormap and the border
    colour (green/red for colour mode, black for B&W mode) — the line
    style (solid vs dotted) and the markers are identical in both, so
    the figure reads consistently whether printed in colour or B&W.
    """
    n  = len(labels)
    wrapped_labels = [wrap_label(l) for l in labels]
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap=cmap, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(wrapped_labels, rotation=45, ha="right", fontsize=11.5)
    ax.set_yticks(range(n))

    if show_yticklabels:
        ax.set_yticklabels(wrapped_labels, fontsize=11.5)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    ax.set_title(title, fontsize=11.5, pad=6)

    # --- anchor row / column outline (dashed black, thin) ---
    ax.add_patch(Rectangle((anchor - 0.5, -0.5), 1, n,
                             fill=False, edgecolor="black", lw=1.0,
                             linestyle="--"))
    ax.add_patch(Rectangle((-0.5, anchor - 0.5), n, 1,
                             fill=False, edgecolor="black", lw=1.0,
                             linestyle="--"))

    succ_col = "#3B6D11" if colour_mode else "black"
    inh_col  = "#A32D2D" if colour_mode else "black"
    lw       = 2.3

    for j in successors:
        for (cx, cy) in [(j, anchor), (anchor, j)]:
            ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1,
                                    fill=False, edgecolor=succ_col,
                                    lw=lw, linestyle="-"))
            ax.text(cx, cy, "▲", ha="center", va="center",
                    fontsize=11.5, color=succ_col, fontweight="bold")
    for j in inhibitors:
        for (cx, cy) in [(j, anchor), (anchor, j)]:
            ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), 1, 1,
                                    fill=False, edgecolor=inh_col,
                                    lw=lw, linestyle=":"))
            ax.text(cx, cy, "▼", ha="center", va="center",
                    fontsize=11.5, color=inh_col, fontweight="bold")

    return im


def plot_before_after(sim_before, sim_after, labels,
                       anchor, successors, inhibitors,
                       out_path_base, colour_mode):
    """
    Produce the figure with shared y-axis.

    out_path_base : path without extension suffix — e.g. 'heatmap.png'
                    The function will save:
                      heatmap_color.png   (colour)
                      heatmap_bw.png      (black & white)
    colour_mode   : True → colour pass, False → B&W pass
    """
    cmap  = "RdBu_r" if colour_mode else "Greys_r"
    title_before = "(a) Without LLM Knowledge"
    title_after  = "(b) With LLM Knowledge"

    # Build figure with two axes sharing the same height.
    # The left axis is slightly wider to accommodate y-tick labels.
    fig = plt.figure(figsize=(11, 5.5))
    gs  = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.06],   # [left panel, right panel, colorbar]
        wspace=0.08,
        left=0.16, right=0.90,
        top=0.82, bottom=0.24,
    )

    ax_left  = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])
    ax_cbar  = fig.add_subplot(gs[2])

    im = _draw_panel(ax_left,  sim_before, title_before, labels,
                     anchor, successors, inhibitors,
                     show_yticklabels=True,
                     cmap=cmap, colour_mode=colour_mode)
    _draw_panel(ax_right, sim_after,  title_after,  labels,
                anchor, successors, inhibitors,
                show_yticklabels=False,
                cmap=cmap, colour_mode=colour_mode)

    # Manually sync y-limits (imshow sets ylim per-axes; without sharey
    # we keep both panels independent but identical in extent).
    ax_right.set_ylim(ax_left.get_ylim())

    # Shared colourbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Cosine Similarity", fontsize=11.5)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=11.5)

    # Legend / caption line
    legend_txt = (
        "— — black dashed = anchor row/column  │  "
        "solid line + ▲ = successor  │  dotted line + ▼ = inhibitor"
    )
    fig.text(0.54, -0.01, legend_txt,
             ha="center", va="center", fontsize=13,
             style="italic", color="#001F3F")

    suffix = "_color" if colour_mode else "_bw"
    # Replace last extension with suffix + same extension
    base, ext = out_path_base.rsplit(".", 1) if "." in out_path_base \
                else (out_path_base, "png")
    save_path = f"{base}{suffix}.{ext}"
    fig.savefig(save_path, dpi=800, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", default=False)
    p.add_argument("--ckpt-no-reg",       default=None)
    p.add_argument("--ckpt-with-reg",     default=None)
    p.add_argument("--num-rels",          type=int, default=None)
    p.add_argument("--relation2id",       default=None)
    p.add_argument("--transition-graph",  default=None)
    p.add_argument("--anchor-relation",   default=None)
    p.add_argument("--top-k",             type=int, default=2)
    p.add_argument("--out",               default="relation_similarity_heatmaps.png")
    args = p.parse_args()

    use_demo = args.demo or not (args.ckpt_no_reg and args.ckpt_with_reg)

    if use_demo:
        print("Using built-in demo data.")
        sim_before = DEMO_BEFORE
        sim_after  = DEMO_AFTER
        labels     = DEMO_LABELS
        anchor     = DEMO_ANCHOR
        successors = DEMO_SUCCESSORS
        inhibitors = DEMO_INHIBITORS
    else:
        if not args.num_rels:
            raise ValueError("--num-rels required with real checkpoints.")
        if not (args.transition_graph and args.anchor_relation):
            raise ValueError("--transition-graph and --anchor-relation required.")

        emb_before = load_emb_rel(args.ckpt_no_reg,  args.num_rels)
        emb_after  = load_emb_rel(args.ckpt_with_reg, args.num_rels)

        rel_ids, anchor_id, succ_ids, inh_ids, id2rel = relations_for_anchor(
            args.transition_graph, args.anchor_relation, args.top_k)

        if args.relation2id:
            id2rel = {}
            with open(args.relation2id, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        try:
                            id2rel[int(parts[1])] = parts[0]
                        except ValueError:
                            pass

        labels     = [id2rel.get(r, str(r)) for r in rel_ids]
        idx        = np.array(rel_ids)
        sim_before = cosine_similarity_matrix(emb_before)[np.ix_(idx, idx)]
        sim_after  = cosine_similarity_matrix(emb_after )[np.ix_(idx, idx)]
        anchor     = rel_ids.index(anchor_id)
        successors = [rel_ids.index(r) for r in succ_ids if r in rel_ids]
        inhibitors = [rel_ids.index(r) for r in inh_ids  if r in rel_ids]

    # Save both colour and B&W versions in one run
    for colour_mode in [True, False]:
        plot_before_after(sim_before, sim_after, labels,
                           anchor, successors, inhibitors,
                           args.out, colour_mode=colour_mode)


if __name__ == "__main__":
    main()

# """
# relation_similarity_heatmap.py
# ===============================
# Plots a side-by-side "before / after" heatmap of pairwise cosine
# similarity between relation embeddings, to visualise the effect of
# relation_reg_loss (mechanism A in the transition graph extension).

# Two modes
# ---------
# 1. --demo (default)
#    Uses a small built-in synthetic example (7 relations, anchor =
#    "Make statement") so the figure can be produced with no checkpoints.
#    Matches the illustrative example used in the presentation slide.

# 2. Real checkpoints
#    Pass --ckpt-no-reg / --ckpt-with-reg (each a torch .pt file with an
#    'state_dict' containing 'emb_rel' of shape (2*num_rels, h_dim)),
#    plus --transition-graph (rel_transition_graph.pkl) and
#    --anchor-relation to focus on a chosen relation's successor /
#    inhibitor edges.

# Usage
# -----
#     # quick demo figure (no dependencies on trained models)
#     python relation_similarity_heatmap.py --demo --out demo.png

#     # real checkpoints
#     python relation_similarity_heatmap.py \
#         --ckpt-no-reg ../models/lambda0.pt \
#         --ckpt-with-reg ../models/lambda01.pt \
#         --num-rels 230 \
#         --relation2id ../data/ICEWS14/relation2id.txt \
#         --transition-graph ../data/ICEWS14/rel_transition_graph.pkl \
#         --anchor-relation "Make statement" \
#         --top-k 2 \
#         --out relation_similarity.png
# """

# import argparse
# import pickle

# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


# # ---------------------------------------------------------------------------
# # Demo data — matches the illustrative example used in the presentation
# # ---------------------------------------------------------------------------

# DEMO_LABELS = [
#     "Express intent\nto cooperate",
#     "Host visit",
#     "Make statement",
#     "Use military\nforce",
#     "Provide aid",
#     "Threaten",
#     "Reduce relations",
# ]

# DEMO_ANCHOR = 2
# DEMO_SUCCESSORS = [0, 1]
# DEMO_INHIBITORS = [3, 5]

# DEMO_BEFORE = np.array([
#     [1.00, 0.12, 0.08, -0.05, 0.22, -0.10, 0.05],
#     [0.12, 1.00, 0.10, 0.03, 0.15, 0.02, -0.08],
#     [0.08, 0.10, 1.00, 0.05, 0.12, -0.02, 0.07],
#     [-0.05, 0.03, 0.05, 1.00, -0.10, 0.18, 0.03],
#     [0.22, 0.15, 0.12, -0.10, 1.00, -0.08, 0.10],
#     [-0.10, 0.02, -0.02, 0.18, -0.08, 1.00, 0.06],
#     [0.05, -0.08, 0.07, 0.03, 0.10, 0.06, 1.00],
# ])

# DEMO_AFTER = np.array([
#     [1.00, 0.12, 0.55, -0.05, 0.22, -0.10, 0.05],
#     [0.12, 1.00, 0.50, 0.03, 0.15, 0.02, -0.08],
#     [0.55, 0.50, 1.00, -0.45, 0.12, -0.40, 0.07],
#     [-0.05, 0.03, -0.45, 1.00, -0.10, 0.18, 0.03],
#     [0.22, 0.15, 0.12, -0.10, 1.00, -0.08, 0.10],
#     [-0.10, 0.02, -0.40, 0.18, -0.08, 1.00, 0.06],
#     [0.05, -0.08, 0.07, 0.03, 0.10, 0.06, 1.00],
# ])


# # ---------------------------------------------------------------------------
# # Loading from real checkpoints
# # ---------------------------------------------------------------------------

# def load_emb_rel(ckpt_path, num_rels):
#     """Load emb_rel (2R, h) from a checkpoint, return base rows [0:R]."""
#     import torch
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
#     return state["emb_rel"][:num_rels].float().numpy()


# def cosine_similarity_matrix(emb):
#     """(R, h) -> (R, R) cosine similarity matrix."""
#     norms = np.linalg.norm(emb, axis=1, keepdims=True)
#     normed = emb / np.clip(norms, 1e-8, None)
#     return normed @ normed.T


# def relations_for_anchor(transition_graph_path, anchor_name, top_k):
#     """
#     Return (rel_ids, anchor_id, successor_ids, inhibitor_ids) for the
#     chosen anchor relation, using rel2id / id2rel from the pickle.
#     """
#     with open(transition_graph_path, "rb") as f:
#         data = pickle.load(f)

#     id2rel = data["id2rel"]
#     rel2id = data.get("rel2id") or {v: k for k, v in id2rel.items()}
#     if anchor_name not in rel2id:
#         raise ValueError(
#             f"'{anchor_name}' not found. Examples: {list(rel2id)[:10]}")

#     anchor_id = rel2id[anchor_name]
#     succ = [r for r, _ in data.get("successors", {}).get(anchor_id, [])][:top_k]
#     inh = [r for r, _ in data.get("inhibitors", {}).get(anchor_id, [])][:top_k]

#     rel_ids, seen = [], set()
#     for r in [anchor_id] + succ + inh:
#         if r not in seen:
#             seen.add(r)
#             rel_ids.append(r)

#     return rel_ids, anchor_id, succ, inh, id2rel


# # ---------------------------------------------------------------------------
# # Plotting
# # ---------------------------------------------------------------------------

# def plot_before_after(sim_before, sim_after, labels,
#                        anchor, successors, inhibitors, out_path):
#     fig, axes = plt.subplots(1, 2, figsize=(13, 6),
#                               gridspec_kw={"wspace": 0.35})
#     titles = ["Without LLM Knowledge", "With LLM Knowledge"]

#     im = None
#     for ax, sim, title in zip(axes, [sim_before, sim_after], titles):
#         im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
#         ax.set_xticks(range(len(labels)))
#         ax.set_yticks(range(len(labels)))
#         ax.set_xticklabels(labels, rotation=90, fontsize=8)
#         ax.set_yticklabels(labels, fontsize=8)
#         ax.set_title(title, fontsize=11)

#         n = len(labels)
#         # outline the anchor's row and column
#         ax.add_patch(Rectangle((anchor - 0.5, -0.5), 1, n,
#                                 fill=False, edgecolor="black", lw=1.2))
#         ax.add_patch(Rectangle((-0.5, anchor - 0.5), n, 1,
#                                 fill=False, edgecolor="black", lw=1.2))

#         # highlight successor cells (green) and inhibitor cells (red)
#         for j in successors:
#             ax.add_patch(Rectangle((j - 0.5, anchor - 0.5), 1, 1,
#                                     fill=False, edgecolor="#3B6D11", lw=2))
#             ax.add_patch(Rectangle((anchor - 0.5, j - 0.5), 1, 1,
#                                     fill=False, edgecolor="#3B6D11", lw=2))
#         for j in inhibitors:
#             ax.add_patch(Rectangle((j - 0.5, anchor - 0.5), 1, 1,
#                                     fill=False, edgecolor="#A32D2D", lw=2))
#             ax.add_patch(Rectangle((anchor - 0.5, j - 0.5), 1, 1,
#                                     fill=False, edgecolor="#A32D2D", lw=2))

#     cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, shrink=0.85)
#     cbar.set_label("cosine similarity")

#     fig.suptitle(
#         "Relation embedding cosine similarity\n"
#         "black box = anchor row/column   |   "
#         "green = successor edges (should increase)   |   "
#         "red = inhibitor edges (should decrease)",
#         fontsize=10, y=1.04)

#     fig.savefig(out_path, dpi=150, bbox_inches="tight")
#     print(f"Saved figure to {out_path}")


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main():
#     p = argparse.ArgumentParser(description=__doc__)
#     p.add_argument("--demo", action="store_true", default=False,
#                    help="Use built-in synthetic example (default if no "
#                         "checkpoints given).")
#     p.add_argument("--ckpt-no-reg", default=None,
#                    help="Checkpoint trained with lambda_trans=0")
#     p.add_argument("--ckpt-with-reg", default=None,
#                    help="Checkpoint trained with lambda_trans>0")
#     p.add_argument("--num-rels", type=int, default=None,
#                    help="Base relation count R (emb_rel is 2R x h)")
#     p.add_argument("--relation2id", default=None,
#                    help="Path to relation2id.txt (for axis labels)")
#     p.add_argument("--transition-graph", default=None,
#                    help="Path to rel_transition_graph.pkl")
#     p.add_argument("--anchor-relation", default=None,
#                    help="Relation name to focus on, e.g. 'Make statement'")
#     p.add_argument("--top-k", type=int, default=2,
#                    help="Number of successor/inhibitor edges to show")
#     p.add_argument("--out", default="relation_similarity_heatmaps.png")
#     args = p.parse_args()

#     use_demo = args.demo or not (args.ckpt_no_reg and args.ckpt_with_reg)

#     if use_demo:
#         print("Using built-in demo data "
#               "(pass --ckpt-no-reg/--ckpt-with-reg for real checkpoints).")
#         plot_before_after(DEMO_BEFORE, DEMO_AFTER, DEMO_LABELS,
#                            DEMO_ANCHOR, DEMO_SUCCESSORS, DEMO_INHIBITORS,
#                            args.out)
#         return

#     if not args.num_rels:
#         raise ValueError("--num-rels is required with real checkpoints.")
#     if not (args.transition_graph and args.anchor_relation):
#         raise ValueError("--transition-graph and --anchor-relation are "
#                           "required with real checkpoints.")

#     emb_before = load_emb_rel(args.ckpt_no_reg, args.num_rels)
#     emb_after = load_emb_rel(args.ckpt_with_reg, args.num_rels)

#     rel_ids, anchor_id, succ_ids, inh_ids, id2rel = relations_for_anchor(
#         args.transition_graph, args.anchor_relation, args.top_k)

#     if args.relation2id:
#         id2rel = {}
#         with open(args.relation2id, encoding="utf-8") as f:
#             for line in f:
#                 parts = line.strip().split("\t")
#                 if len(parts) >= 2:
#                     try:
#                         id2rel[int(parts[1])] = parts[0]
#                     except ValueError:
#                         pass

#     labels = [id2rel.get(r, str(r)) for r in rel_ids]

#     sim_before_full = cosine_similarity_matrix(emb_before)
#     sim_after_full = cosine_similarity_matrix(emb_after)

#     idx = np.array(rel_ids)
#     sim_before = sim_before_full[np.ix_(idx, idx)]
#     sim_after = sim_after_full[np.ix_(idx, idx)]

#     anchor_pos = rel_ids.index(anchor_id)
#     succ_pos = [rel_ids.index(r) for r in succ_ids if r in rel_ids]
#     inh_pos = [rel_ids.index(r) for r in inh_ids if r in rel_ids]

#     plot_before_after(sim_before, sim_after, labels,
#                        anchor_pos, succ_pos, inh_pos, args.out)


# if __name__ == "__main__":
#     main()