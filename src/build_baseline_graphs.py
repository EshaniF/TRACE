"""
build_baseline_graphs.py
=========================
Builds baseline transition graphs to compare against the LLM-derived
rel_transition_graph.pkl:

  1. Empirical    — pure co-occurrence graph from train.txt, no LLM.
                     Includes BOTH successors and inhibitors (the
                     original llm_transition_scorer.py --skip-llm path
                     only produces successors; this version adds a
                     symmetric empirical inhibitor definition so it's
                     comparable to the LLM graph, which has both).

  2. Random        — same shape (top_k edges/relation, weights in
                     [0,1]) as a real graph, but edges point to random
                     relations. Controls for "does having ANY extra
                     signal help, regardless of content."

  3. Shuffled      — takes an existing (e.g. LLM-derived) graph and
                     permutes relation *labels* globally, so the graph's
                     topology/weight distribution is preserved exactly
                     but specific relation-to-relation mappings are
                     destroyed. Controls for "does the graph's
                     statistical shape matter, independent of whether
                     the specific transitions are correct."

All three save in the same pickle format consumed by
TransitionGraph.__init__ / rrgcntr.py:

    {
      "successors": {r_id: [(r_id, weight), ...]},
      "inhibitors": {r_id: [(r_id, weight), ...]},
      "id2rel":     {r_id: name},
      "rel2id":     {name: r_id},
    }

Usage
-----
  # Empirical baseline (from train.txt directly)
  python build_baseline_graphs.py -d ICEWS14 --mode empirical

  # Random baseline
  python build_baseline_graphs.py -d ICEWS14 --mode random --top-k 5 --seed 0

  # Shuffled baseline (requires an existing LLM graph to shuffle)
  python build_baseline_graphs.py -d ICEWS14 --mode shuffled \
      --source ../data/ICEWS14/rel_transition_graph.pkl --seed 0

  # Build all three in one go, each to its own output file
  python build_baseline_graphs.py -d ICEWS14 --mode all --seed 0
"""

import os
import pickle
import random
import argparse
from collections import defaultdict, Counter


def read_id2name(path: str) -> dict:
    d = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"relation2id.txt not found at {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    d[int(parts[1])] = parts[0]
                except ValueError:
                    pass
    return d


def read_train_events(path: str):
    """Yields (s, r, o, t) int tuples from train.txt."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"train.txt not found at {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                try:
                    yield (int(parts[0]), int(parts[1]),
                           int(parts[2]), int(parts[3]))
                except ValueError:
                    pass


def save_graph(payload: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    n_succ = sum(len(v) for v in payload["successors"].values())
    n_inh  = sum(len(v) for v in payload["inhibitors"].values())
    print(f"Saved: {out_path}")
    print(f"  successor edges : {n_succ}")
    print(f"  inhibitor edges : {n_inh}")


# ---------------------------------------------------------------------------
# 1. Empirical graph (successors AND inhibitors, symmetric definition)
# ---------------------------------------------------------------------------

def build_empirical_graph(data_dir: str, id2rel: dict,
                           window: int = 7, top_k: int = 5,
                           min_count: int = 3) -> dict:

    subj_events = defaultdict(list)
    global_freq = Counter()
    for s, r, o, t in read_train_events(os.path.join(data_dir, "train.txt")):
        if r not in id2rel:
            continue
        subj_events[s].append((t, r))
        global_freq[r] += 1

    total_events = sum(global_freq.values()) or 1
    global_rate  = {r: c / total_events for r, c in global_freq.items()}

    followed_by   = defaultdict(Counter)   # r1 -> Counter(r2 -> count within window)
    anchor_count  = Counter()              # how many times r1 appears as an anchor
    for s, events in subj_events.items():
        events_sorted = sorted(events, key=lambda x: x[0])
        for i, (t1, r1) in enumerate(events_sorted):
            anchor_count[r1] += 1
            for t2, r2 in events_sorted[i + 1:]:
                if t2 - t1 > window:
                    break
                if r1 != r2:
                    followed_by[r1][r2] += 1

    successors = {}
    inhibitors = {}
    for r1 in id2rel:
        n_anchor = anchor_count.get(r1, 0)
        counts   = followed_by.get(r1, Counter())

        # Successors: top-k by conditional follow-rate P(r2 | r1 anchors)
        if n_anchor > 0:
            ranked = sorted(counts.items(), key=lambda x: -x[1])
            succ_list = []
            for r2, cnt in ranked:
                if cnt < min_count:
                    continue
                cond_rate = cnt / n_anchor
                succ_list.append((r2, cond_rate))
                if len(succ_list) >= top_k:
                    break
            # normalise weights into [0, 1] within this relation's list
            if succ_list:
                max_w = max(w for _, w in succ_list)
                succ_list = [(r2, round(w / max_w, 4)) for r2, w in succ_list]
            successors[r1] = succ_list
        else:
            successors[r1] = []

        # Inhibitors: relations that are globally common but rarely
        # follow r1 within the window, i.e. conditional rate << global rate.
        if n_anchor > 0:
            deficits = []
            for r2 in id2rel:
                if r2 == r1:
                    continue
                g_rate    = global_rate.get(r2, 0.0)
                if g_rate <= 0:
                    continue
                cond_rate = counts.get(r2, 0) / n_anchor
                # Suppression score: how much rarer than expected.
                deficit = g_rate - cond_rate
                if deficit > 0:
                    deficits.append((r2, deficit))
            deficits.sort(key=lambda x: -x[1])
            inh_list = deficits[:top_k]
            if inh_list:
                max_w = max(w for _, w in inh_list)
                inh_list = [(r2, round(w / max_w, 4)) for r2, w in inh_list]
            inhibitors[r1] = inh_list
        else:
            inhibitors[r1] = []

    return {"successors": successors, "inhibitors": inhibitors}


def build_random_graph(id2rel: dict, top_k: int = 5,
                        seed: int = 0) -> dict:
    """
    Same shape as a real graph (top_k edges per relation, weights in
    [0, 1]), but targets and weights are random. Self-edges excluded.
    Successor and inhibitor lists are drawn independently (an edge can
    appear in both, same as could happen with a real LLM-derived graph).
    """
    rng     = random.Random(seed)
    all_ids = list(id2rel.keys())

    def _random_edges(anchor):
        candidates = [r for r in all_ids if r != anchor]
        k = min(top_k, len(candidates))
        chosen = rng.sample(candidates, k) if k > 0 else []
        return [(r2, round(rng.random(), 4)) for r2 in chosen]

    successors = {r: _random_edges(r) for r in all_ids}
    inhibitors = {r: _random_edges(r) for r in all_ids}
    return {"successors": successors, "inhibitors": inhibitors}


def build_shuffled_graph(source_payload: dict, id2rel: dict,
                          seed: int = 0) -> dict:
    """
    Preserves the exact topology and weight distribution of an existing
    graph (e.g. the LLM-derived one) but destroys which specific
    relation maps to which. Implemented as a single global relabelling:
    draw a random permutation pi over relation ids, then remap every
    relation id r -> pi(r) everywhere (as both anchors and targets).
    """
    rng      = random.Random(seed)
    all_ids  = sorted(id2rel.keys())
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    perm = dict(zip(all_ids, shuffled))   # r -> pi(r)

    def _remap(edges_dict):
        out = {}
        for r_id, entries in edges_dict.items():
            new_r = perm.get(r_id, r_id)
            out[new_r] = [(perm.get(r2, r2), w) for r2, w in entries]
        return out

    successors = _remap(source_payload.get("successors", {}))
    inhibitors = _remap(source_payload.get("inhibitors", {}))
    return {"successors": successors, "inhibitors": inhibitors}



def run(args):
    data_dir = os.path.join(args.data_root, args.dataset)
    id2rel   = read_id2name(os.path.join(data_dir, "relation2id.txt"))
    rel2id   = {name: r_id for r_id, name in id2rel.items()}

    modes = (["empirical", "random", "shuffled"] if args.mode == "all"
             else [args.mode])

    for mode in modes:
        if mode == "empirical":
            graph = build_empirical_graph(
                data_dir, id2rel,
                window=args.window, top_k=args.top_k,
                min_count=args.min_count)
            out_path = args.out or os.path.join(
                data_dir, "rel_transition_graph_empirical.pkl")

        elif mode == "random":
            graph = build_random_graph(
                id2rel, top_k=args.top_k, seed=args.seed)
            out_path = args.out or os.path.join(
                data_dir, f"rel_transition_graph_random_seed{args.seed}.pkl")

        elif mode == "shuffled":
            if not args.source:
                raise ValueError(
                    "--mode shuffled requires --source pointing at an "
                    "existing rel_transition_graph.pkl to permute.")
            with open(args.source, "rb") as f:
                source_payload = pickle.load(f)
            graph = build_shuffled_graph(
                source_payload, id2rel, seed=args.seed)
            out_path = args.out or os.path.join(
                data_dir, f"rel_transition_graph_shuffled_seed{args.seed}.pkl")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        payload = {
            "successors": graph["successors"],
            "inhibitors": graph["inhibitors"],
            "id2rel":     id2rel,
            "rel2id":     rel2id,
        }
        save_graph(payload, out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build empirical / random / shuffled baseline "
                     "transition graphs.")
    p.add_argument("-d", "--dataset",   required=True)
    p.add_argument("--data-root",       type=str, default="../data")
    p.add_argument("--mode",            type=str, required=True,
                   choices=["empirical", "random", "shuffled", "all"])
    p.add_argument("--top-k",           type=int, default=5)
    p.add_argument("--window",          type=int, default=7,
                   help="Empirical mode only: co-occurrence window.")
    p.add_argument("--min-count",       type=int, default=3,
                   help="Empirical mode only: minimum raw count for a "
                        "successor edge to be kept.")
    p.add_argument("--seed",            type=int, default=0,
                   help="Random / shuffled modes only.")
    p.add_argument("--source",          type=str, default=None,
                   help="Shuffled mode only: path to the existing graph "
                        "pickle (e.g. the LLM-derived one) to permute.")
    p.add_argument("--out",             type=str, default=None,
                   help="Output path override. Defaults to a "
                        "mode-specific filename next to the dataset.")
    args = p.parse_args()
    run(args)