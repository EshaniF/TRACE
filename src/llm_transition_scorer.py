"""
llm_transition_scorer.py
========================
Runs ONCE before training. Zero LLM calls at training time.

For each relation r, asks the LLM:
  "In international political events, if relation r occurs between two
   actors at time t, which relations are most likely to follow (successors)
   and which become unlikely (inhibitors)?"

Output
------
  rel_transition_graph.pkl  {
    "successors":  {r_id: [(r_id, weight), ...]},
    "inhibitors":  {r_id: [(r_id, weight), ...]},
    "id2rel":      {r_id: name},
    "rel2id":      {name: r_id},
  }

Usage
-----
  export GROQ_API_KEY=gsk_...
  python llm_transition_scorer.py -d ICEWS14
  python llm_transition_scorer.py -d ICEWS18 --model llama-3.3-70b-versatile
  python llm_transition_scorer.py -d GDELT   --rescale        # recommended
  python llm_transition_scorer.py -d ICEWS14 --skip-llm       # empirical fallback


"""

import os
import json
import re
import time
import pickle
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm

try:
    from groq import Groq
except ImportError:
    raise ImportError("pip install groq  &&  export GROQ_API_KEY=gsk_...")

GROQ_MODELS = {
    "llama3-8b":     "llama3-8b-8192",
    "llama3-70b":    "llama3-70b-8192",
    "llama-3.1-8b":  "llama-3.1-8b-instant",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "mixtral":       "mixtral-8x7b-32768",
}

DOMAIN_HINTS = {
    "ICEWS18":    "international political and military events",
    "ICEWS14":    "international political and military events",
    "ICEWS05-15": "international political and military events",
    "GDELT":      "global news events",
    "YAGO":       "encyclopaedic knowledge",
    "WIKI":       "general world knowledge",
}



def read_id2name(path: str) -> dict:
    """Read relation2id.txt. Returns {id: name}."""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    d[int(parts[1])] = parts[0]
                except ValueError:
                    pass
    return d


def load_relation_frequencies(data_dir: str) -> dict:
    """Count relation appearances in train.txt. Returns {r_id: count}."""
    freq = Counter()
    path = os.path.join(data_dir, "train.txt")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    freq[int(parts[1])] += 1
                except ValueError:
                    pass
    return dict(freq)




def read_entity_id2name(path: str) -> dict:
    """
    Read entity2id.txt.  Returns {entity_id: entity_name}.
    Tries both column orderings:
      name <TAB> id   (most datasets)
      id   <TAB> name (some datasets)
    If the file is absent, returns {} and examples fall back to
    using raw entity IDs, which still produce valid prompts.
    """
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            # Try name <TAB> id first (the common format)
            try:
                d[int(parts[1])] = parts[0]
                continue
            except ValueError:
                pass
            # Fallback: id <TAB> name
            try:
                d[int(parts[0])] = parts[1]
            except ValueError:
                pass
    return d


def load_relation_examples(data_dir: str,
                            id2rel: dict,
                            id2ent: dict,
                            n_examples: int = 3) -> dict:
    """
    For each relation, collect up to n_examples real triples
    from train.txt formatted as (subject_name, relation_name, object_name).

    Returns {r_id: [(subj_name, rel_name, obj_name), ...]}.

    Used to ground the LLM prompt in concrete observed events rather than
    abstract relation names.  This is especially important for generic
    CAMEO-coded relations (GDELT) where bare names like "Make statement"
    give the LLM no causal context to reason from.

    If id2ent is empty (entity2id.txt absent), entity IDs are used as
    names — the prompt still works, just with numeric actors.
    """
    if n_examples <= 0:
        return {}

    examples  = defaultdict(list)
    path      = os.path.join(data_dir, "train.txt")

    if not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                s = int(parts[0])
                r = int(parts[1])
                o = int(parts[2])
            except ValueError:
                continue
            if r not in id2rel:
                continue
            if len(examples[r]) >= n_examples:
                continue
            s_name = id2ent.get(s, f"Entity_{s}")
            r_name = id2rel[r]
            o_name = id2ent.get(o, f"Entity_{o}")
            examples[r].append((s_name, r_name, o_name))

    return dict(examples)


def repair_json(text: str) -> str:

    # Pattern G: extract first complete { ... } block, discarding prose
    def _extract(t):
        start = t.find('{')
        if start == -1:
            return t          # no JSON object found — return as-is
        depth = 0
        for i, ch in enumerate(t[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return t[start:i + 1]
        return t[start:]      # unclosed — Pattern E will close it

    text = _extract(text)

    # Pattern F: strip any remaining markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text.strip())

    # Pattern B: ,"<any string>"null
    text = re.sub(r',\s*"[^"]*"\s*null', '', text)

    # Pattern C: ,"<any key>":null
    text = re.sub(r',\s*"[^"]*"\s*:\s*null', '', text)

    # Pattern A: missing "relation": key
    text = re.sub(
        r'\{\s*"(?!relation|score)([^"]+)"\s*,\s*"score"',
        r'{"relation":"\1","score"',
        text)

    # Pattern D: trailing comma before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Pattern E: unclosed arrays or objects
    open_brackets = text.count('[') - text.count(']')
    open_braces   = text.count('{') - text.count('}')
    text = text.rstrip()
    text += ']' * max(0, open_brackets)
    text += '}' * max(0, open_braces)

    return text

# LLM call with repair + retry


def groq_call(client, msg: str, model: str,
               max_retries: int = 3,
               base_delay: float = 30.0,
               debug: bool = False) -> dict:

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": msg}],
                temperature=0.0,
                max_tokens=512,
                # response_format intentionally omitted — see docstring
            )
            raw = resp.choices[0].message.content.strip()
            if debug:
                print(f"  [DEBUG] raw output (first 200 chars): "
                      f"{repr(raw[:200])}")
            repaired = repair_json(raw)
            return json.loads(repaired)
        except json.JSONDecodeError:
            time.sleep(base_delay)
        except Exception as exc:
            wait = min(base_delay * (2 ** attempt), 120)
            if "rate_limit" in str(exc).lower() or "429" in str(exc):
                print(f"  Rate limited — waiting {wait:.0f}s...")
            else:
                print(f"  Error: {exc} — waiting {wait:.0f}s...")
            time.sleep(wait)
    return {}


# ---------------------------------------------------------------------------
# LLM transition scoring — one call per anchor relation
# ---------------------------------------------------------------------------

def llm_get_transitions(client,
                         anchor_name: str,
                         anchor_examples: list,
                         all_relation_names: list,
                         domain: str,
                         model: str,
                         top_k: int,
                         debug: bool = False) -> dict:
    """
    One LLM call per anchor relation.

    anchor_examples is a list of up to n_examples real training
    triples formatted as (subj_name, rel_name, obj_name).  They are
    injected into the prompt so the LLM can reason about actual event
    patterns rather than abstract relation names.

    For generic relations like GDELT's "Make statement", seeing:
      e.g. United States -> [Make statement] -> Russia
      e.g. China -> [Make statement] -> Japan
    gives the LLM enough context to reason that follow-on events are
    diplomatic in nature, producing differentiated confidence scores
    instead of the flat 0.6–0.7 distribution seen without context.

    FIX-PROMPT: explicit merge-prevention instruction retained.
    FIX-PARSE:  None score guard retained.
    """
    def clean(name):
        return name.replace('_', ' ').strip()

    rel_list = "\n".join(
        f"- {clean(name)}" for name in all_relation_names
        if name != anchor_name)

    clean_to_orig = {clean(name): name for name in all_relation_names}

    # SOL-1: build example block from real training triples
    if anchor_examples:
        example_lines = "\n".join(
            f"  e.g. {s} -> [{clean(r)}] -> {o}"
            for s, r, o in anchor_examples)
        example_block = (
            f'Observed examples of "{clean(anchor_name)}" in the data:\n'
            f'{example_lines}\n\n')
    else:
        example_block = ""
    msg = (
        f'Domain: {domain}\n'
        f'Anchor relation: "{clean(anchor_name)}"\n\n'
        f'{example_block}'
        f'From the relation list below, select exactly {top_k} SUCCESSORS '
        f'and exactly {top_k} INHIBITORS for the anchor relation.\n\n'
        f'SUCCESSOR: a relation likely to occur SHORTLY AFTER the anchor.\n'
        f'  Score 1–10: 10 = almost certain to follow, 1 = unlikely to follow.\n\n'
        f'INHIBITOR: a relation whose occurrence is SUPPRESSED by the anchor.\n'
        f'  Score 1–10: 10 = strongly suppressed, 1 = weakly suppressed.\n\n'
        f'Rules:\n'
        f'  - Use ONLY names from the list below, copied EXACTLY.\n'
        f'  - Each relation is ONE object: {{"relation":"name","score":N}}\n'
        f'  - Never combine two relation names in a single object.\n\n'
        f'Relation list:\n{rel_list}\n\n'
        f'Return ONLY this JSON, no explanation:\n'
        f'{{"successors":[{{"relation":"exact name","score":8}},...],\n'
        f' "inhibitors":[{{"relation":"exact name","score":9}},...]}}'
    )

    result = groq_call(client, msg, model, debug=debug)

    # Key aliases: without response_format the LLM sometimes uses
    # different key names.  Map all observed variants to the canonical
    # "successors" / "inhibitors" keys before calling _parse().
    SUCC_ALIASES = {"successors", "successor", "following",
                    "likely", "likely_relations", "following_relations",
                    "successor_relations", "transitions_successors"}
    INH_ALIASES  = {"inhibitors", "inhibitor", "inhibited",
                    "suppressed", "suppressive", "unlikely",
                    "unlikely_relations", "inhibiting",
                    "inhibitor_relations", "transitions_inhibitors"}

    def _normalise_keys(d):
        """
        Return a dict guaranteed to have 'successors' and 'inhibitors' keys
        by searching for any known alias.  Also handles nesting under a
        'transitions' wrapper key.
        """
        if not isinstance(d, dict):
            return {"successors": [], "inhibitors": []}
        # Unwrap single-level nesting (e.g. {"transitions": {...}})
        if len(d) == 1:
            inner = next(iter(d.values()))
            if isinstance(inner, dict):
                d = inner
        out = {"successors": [], "inhibitors": []}
        for k, v in d.items():
            k_lower = k.lower().replace(" ", "_")
            if k_lower in SUCC_ALIASES and isinstance(v, list):
                out["successors"] = v
            elif k_lower in INH_ALIASES and isinstance(v, list):
                out["inhibitors"] = v
        return out

    result = _normalise_keys(result)

    def _parse(key):
        out = []
        for entry in result.get(key, []):
            try:
                name  = str(entry.get("relation", "")).strip()
                score = entry.get("score")
                if score is None:
                    continue
                score = float(score)

                # Exact match on cleaned name
                if name in clean_to_orig:
                    out.append((clean_to_orig[name],
                                max(0.0, min(1.0, (score - 1) / 9.0))))
                    continue

                # Direct match on original names
                if name in set(all_relation_names):
                    out.append((name,
                                max(0.0, min(1.0, (score - 1) / 9.0))))
                    continue

                # Token-overlap match
                best_orig  = None
                best_score = 0
                name_tokens = set(name.lower().split())
                for orig_name in all_relation_names:
                    orig_tokens = set(clean(orig_name).lower().split())
                    overlap     = len(name_tokens & orig_tokens)
                    if overlap > best_score:
                        best_score = overlap
                        best_orig  = orig_name
                if best_orig is not None and best_score >= 2:
                    out.append((best_orig,
                                max(0.0, min(1.0, (score - 1) / 9.0))))

            except (TypeError, ValueError, AttributeError):
                pass
        return out

    return {
        "successors": _parse("successors"),
        "inhibitors": _parse("inhibitors"),
    }


# ---------------------------------------------------------------------------
# Stage 1 — Empirical transition validation (pure graph, no LLM)
# ---------------------------------------------------------------------------

def compute_empirical_transitions(data_dir: str,
                                   id2rel: dict,
                                   window: int = 3) -> dict:
    """
    Compute empirical successor frequencies from training data.
    Returns {r_id: {r_id: count}}.
    """
    path = os.path.join(data_dir, "train.txt")
    if not os.path.exists(path):
        return {}

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
            for t2, r2 in events_sorted[i+1:]:
                if t2 - t1 > window:
                    break
                if r1 != r2:
                    transitions[r1][r2] += 1

    return dict(transitions)


# ---------------------------------------------------------------------------
# Build transition graph
# ---------------------------------------------------------------------------

def build_transition_graph(id2rel: dict,
                             rel_freq: dict,
                             client,
                             model: str,
                             domain: str,
                             top_k: int,
                             max_rel_calls: int,
                             empirical: dict,
                             examples: dict,
                             args_debug: bool = False) -> dict:
    """
    Score temporal transitions for all relations.
    One LLM call per anchor.  Anchors sorted frequency-descending.

    examples dict {r_id: [(s, r, o), ...]} is passed through to
    llm_get_transitions() so each prompt is grounded in real training data.
    """
    rel2id = {name: r_id for r_id, name in id2rel.items()}

    sorted_rels = sorted(id2rel.keys(),
                         key=lambda r: -rel_freq.get(r, 0))
    if max_rel_calls > 0:
        anchors = sorted_rels[:max_rel_calls]
    else:
        anchors = sorted_rels

    all_rel_names = [id2rel[r] for r in id2rel]

    print(f"  Scoring {len(anchors)} anchor relations "
          f"(one LLM call each, top-{top_k} per direction)")
    print(f"  Examples per anchor : "
          f"{max(len(v) for v in examples.values()) if examples else 0}")
    print(f"  Estimated time: ~{len(anchors) * 2 / 60:.1f} min "
          f"at 2s/call")

    successors = defaultdict(list)
    inhibitors = defaultdict(list)

    for anchor_id in tqdm(anchors, desc="LLM transition scoring"):
        anchor_name     = id2rel.get(anchor_id, str(anchor_id))
        anchor_examples = examples.get(anchor_id, [])   # SOL-1
        cand_names      = [id2rel[r] for r in id2rel if r != anchor_id]

        transitions = llm_get_transitions(
            client, anchor_name, anchor_examples,        # SOL-1
            cand_names, domain, model, top_k,
            debug=(args_debug and anchor_id == anchors[0]))  # debug first only

        for name, score in transitions["successors"]:
            if name not in rel2id:
                continue
            cand_id   = rel2id[name]
            emp_count = empirical.get(anchor_id, {}).get(cand_id, 0)
            if emp_count == 0 and score > 0.7:
                score = score * 0.6
            successors[anchor_id].append((cand_id, round(score, 4)))

        for name, score in transitions["inhibitors"]:
            if name not in rel2id:
                continue
            cand_id = rel2id[name]
            inhibitors[anchor_id].append((cand_id, round(score, 4)))

        time.sleep(2.0)

    return {
        "successors": dict(successors),
        "inhibitors": dict(inhibitors),
    }


def rescale_with_empirical(graph_data: dict,
                            empirical: dict,
                            id2rel: dict,
                            floor: float = 0.3,
                            min_weight: float = 0.05) -> dict:
    """
    Rescale LLM-assigned transition weights by multiplying with
    an empirical support factor derived from training co-occurrence counts.

    Rescaling formula
    -----------------
    emp_norm  = empirical_count(r, r2) / max_empirical_count   ∈ [0, 1]

    For successors (high co-occurrence → higher confidence):
      new_weight = llm_weight × (floor + (1 - floor) × emp_norm)

    For inhibitors (low co-occurrence → higher inhibitor confidence):
      new_weight = llm_weight × (floor + (1 - floor) × (1 - emp_norm))

    The floor parameter (default 0.3) ensures that an LLM-assigned edge
    with zero empirical support is not fully discarded — it retains 30%
    of its original weight.  This preserves LLM knowledge for genuinely
    novel transitions that may not yet appear in the training window.

    Edges whose rescaled weight falls below min_weight are removed.
    This prunes the lowest-quality edges from the graph entirely.

    Parameters
    ----------
    graph_data  : dict with 'successors' and 'inhibitors' keys
    empirical   : {r_id: {r_id: count}} from compute_empirical_transitions
    id2rel      : {r_id: name}
    floor       : minimum fraction of LLM weight retained when emp=0
    min_weight  : edges below this weight after rescaling are dropped

    Returns
    -------
    dict with rescaled 'successors' and 'inhibitors'
    """
    # Find max empirical count across all pairs for normalisation
    all_counts = [cnt for r_dict in empirical.values()
                  for cnt in r_dict.values()]
    max_count  = max(all_counts) if all_counts else 1

    new_succs = {}
    new_inhs  = {}

    for r_id, entries in graph_data.get("successors", {}).items():
        rescaled = []
        for r2, w in entries:
            emp      = empirical.get(r_id, {}).get(r2, 0)
            emp_norm = emp / max_count
            # Successors: boost if empirically common, penalise if never seen
            new_w    = w * (floor + (1.0 - floor) * emp_norm)
            if new_w >= min_weight:
                rescaled.append((r2, round(new_w, 4)))
        # Re-sort by descending weight after rescaling
        new_succs[r_id] = sorted(rescaled, key=lambda x: -x[1])

    for r_id, entries in graph_data.get("inhibitors", {}).items():
        rescaled = []
        for r2, w in entries:
            emp      = empirical.get(r_id, {}).get(r2, 0)
            emp_norm = emp / max_count
            # Inhibitors: boost if empirically rare (low co-occurrence)
            new_w    = w * (floor + (1.0 - floor) * (1.0 - emp_norm))
            if new_w >= min_weight:
                rescaled.append((r2, round(new_w, 4)))
        new_inhs[r_id] = sorted(rescaled, key=lambda x: -x[1])

    n_succ_before = sum(len(v) for v in graph_data.get("successors", {}).values())
    n_inh_before  = sum(len(v) for v in graph_data.get("inhibitors", {}).values())
    n_succ_after  = sum(len(v) for v in new_succs.values())
    n_inh_after   = sum(len(v) for v in new_inhs.values())

    print(f"  [Rescale] Successors : {n_succ_before} → {n_succ_after} edges "
          f"({n_succ_before - n_succ_after} pruned below min_weight={min_weight})")
    print(f"  [Rescale] Inhibitors : {n_inh_before} → {n_inh_after} edges "
          f"({n_inh_before - n_inh_after} pruned below min_weight={min_weight})")

    return {"successors": new_succs, "inhibitors": new_inhs}


def empirical_only_transitions(id2rel: dict,
                                 empirical: dict,
                                 top_k: int) -> dict:
    """Build transition graph from empirical co-occurrence only."""
    successors = {}
    for r_id, counts in empirical.items():
        if r_id not in id2rel:
            continue
        total  = sum(counts.values()) + 1e-9
        ranked = sorted(counts.items(), key=lambda x: -x[1])[:top_k]
        successors[r_id] = [(r2, round(cnt / total, 4))
                             for r2, cnt in ranked
                             if r2 in id2rel]

    print(f"  Empirical fallback: transitions for "
          f"{len(successors)} relations")
    return {"successors": successors, "inhibitors": {}}


def find_undercovered_relations(graph_data: dict,
                                 id2rel: dict,
                                 min_edges: int = 2) -> dict:
    """
    Identify relations with fewer than min_edges in either direction.

    Returns a dict with two lists:
      {
        "successors":  [r_id, ...],   # relations needing more successor edges
        "inhibitors":  [r_id, ...],   # relations needing more inhibitor edges
      }

    A relation appears in a list if its current edge count in that direction
    is strictly less than min_edges.  Relations with zero edges are included.
    The lists may overlap — a relation can be undercovered in both directions.
    """
    succ_edges = graph_data.get("successors", {})
    inh_edges  = graph_data.get("inhibitors", {})
    all_ids    = set(id2rel.keys())

    need_succ = sorted(
        r for r in all_ids
        if len(succ_edges.get(r, [])) < min_edges)

    need_inh = sorted(
        r for r in all_ids
        if len(inh_edges.get(r, [])) < min_edges)

    print(f"\n  Relations with < {min_edges} successor edges : {len(need_succ)}")
    print(f"  Relations with < {min_edges} inhibitor edges : {len(need_inh)}")
    print(f"  Union (need patching in at least one direction): "
          f"{len(set(need_succ) | set(need_inh))}")

    return {"successors": need_succ, "inhibitors": need_inh}


def patch_transition_graph(graph_data: dict,
                            undercovered: dict,
                            id2rel: dict,
                            rel_freq: dict,
                            client,
                            model: str,
                            domain: str,
                            top_k: int,
                            empirical: dict,
                            examples: dict,
                            args_debug: bool = False) -> dict:
    """
    Re-score undercovered relations and merge results into graph_data.

    Strategy
    --------
    For each undercovered relation, call the LLM again and merge the
    new edges with the existing ones:

    Merge rule (per direction):
      - Keep all existing edges that are NOT in the new result.
      - For edges present in both, take the HIGHER weight (the LLM may
        have returned a better score on the second attempt).
      - Add any new edges from the patch call.
      - Deduplicate by target relation ID (keep highest weight).
      - Re-sort descending by weight.

    The union of existing + new edges is used so no previously valid
    edge is lost during patching.

    Relations are scored in frequency-descending order so that the most
    important undercovered relations are patched first in case the run
    is interrupted.

    Parameters
    ----------
    graph_data    : existing graph dict (modified in-place and returned)
    undercovered  : output of find_undercovered_relations()
    id2rel        : {r_id: name}
    rel_freq      : {r_id: frequency count}
    client        : Groq client
    model         : model string
    domain        : domain hint string
    top_k         : K for this patch run (can differ from original)
    empirical     : empirical co-occurrence counts
    examples      : {r_id: [(s,r,o), ...]} training examples
    args_debug    : if True, print raw output for first patched relation

    Returns
    -------
    Updated graph_data dict with patched edges merged in.
    """
    succ_edges = graph_data.get("successors", {})
    inh_edges  = graph_data.get("inhibitors", {})
    rel2id     = {name: r_id for r_id, name in id2rel.items()}

    # Union of all relations that need patching in at least one direction
    all_ids      = set(id2rel.keys())
    need_succ    = set(undercovered["successors"])
    need_inh     = set(undercovered["inhibitors"])
    patch_ids    = need_succ | need_inh

    # Score frequency-descending so most important come first
    patch_ids_ordered = sorted(
        patch_ids, key=lambda r: -rel_freq.get(r, 0))

    all_rel_names = [id2rel[r] for r in id2rel]

    print(f"\n  Patching {len(patch_ids_ordered)} relations "
          f"(one LLM call each)...")

    first = True
    for r_id in tqdm(patch_ids_ordered, desc="Patch scoring"):
        anchor_name     = id2rel.get(r_id, str(r_id))
        anchor_examples = examples.get(r_id, [])
        cand_names      = [id2rel[r] for r in id2rel if r != r_id]
        debug_this      = args_debug and first

        transitions = llm_get_transitions(
            client, anchor_name, anchor_examples,
            cand_names, domain, model, top_k,
            debug=debug_this)
        first = False

        # ── Merge successors ────────────────────────────────────────────
        if r_id in need_succ:
            existing = {r2: w for r2, w in succ_edges.get(r_id, [])}
            for name, score in transitions["successors"]:
                if name not in rel2id:
                    continue
                r2  = rel2id[name]
                emp = empirical.get(r_id, {}).get(r2, 0)
                if emp == 0 and score > 0.7:
                    score = score * 0.6
                # Keep higher weight if already present
                existing[r2] = max(existing.get(r2, 0.0),
                                   round(score, 4))
            succ_edges[r_id] = sorted(
                existing.items(), key=lambda x: -x[1])

        # ── Merge inhibitors ────────────────────────────────────────────
        if r_id in need_inh:
            existing = {r2: w for r2, w in inh_edges.get(r_id, [])}
            for name, score in transitions["inhibitors"]:
                if name not in rel2id:
                    continue
                r2 = rel2id[name]
                existing[r2] = max(existing.get(r2, 0.0),
                                   round(score, 4))
            inh_edges[r_id] = sorted(
                existing.items(), key=lambda x: -x[1])

        time.sleep(2.0)

    graph_data["successors"] = succ_edges
    graph_data["inhibitors"] = inh_edges
    return graph_data


def run(args):
    data_dir = os.path.join(args.data_root, args.dataset)
    out_path = os.path.join(data_dir, "rel_transition_graph.pkl")

    # ── Patch mode ────────────────────────────────────────────────────────
    # Load the existing graph, find undercovered relations, re-score them,
    # merge, and save.  Does not touch well-covered relations.
    if args.patch:
        if not os.path.exists(out_path):
            print(f"No existing graph at {out_path}.")
            print("Run the scorer first (without --patch), then patch.")
            return

        print(f"=== PATCH MODE — {args.dataset} ===")
        print(f"  Loading existing graph from {out_path}...")
        with open(out_path, "rb") as f:
            payload = pickle.load(f)

        id2rel   = payload["id2rel"]
        rel_freq = load_relation_frequencies(data_dir)

        # Report current coverage before patching
        undercovered = find_undercovered_relations(
            payload, id2rel, min_edges=args.patch_min_edges)

        if not undercovered["successors"] and not undercovered["inhibitors"]:
            print(f"\n  All relations already have ≥{args.patch_min_edges} "
                  f"edges in both directions.  Nothing to patch.")
            return

        # Load examples for grounding 
        ent_path = os.path.join(data_dir, "entity2id.txt")
        id2ent   = read_entity_id2name(ent_path)
        examples = {}
        if args.n_examples > 0:
            examples = load_relation_examples(
                data_dir, id2rel, id2ent, n_examples=args.n_examples)

        # Empirical transitions for dampening
        print(f"\n  Computing empirical transitions (window={args.window})...")
        empirical = compute_empirical_transitions(
            data_dir, id2rel, window=args.window)

        api_key  = os.environ.get("GROQ_API_KEY", "")
        model_id = GROQ_MODELS.get(args.model, args.model)
        domain   = DOMAIN_HINTS.get(args.dataset,
                                     f"temporal KG ({args.dataset})")

        if not api_key:
            print("  No GROQ_API_KEY found — cannot patch without LLM.")
            return

        client = Groq(api_key=api_key)
        print(f"  Model : {model_id}")
        print(f"  Patch top-k : {args.patch_top_k or args.top_k}")

        patch_k = args.patch_top_k if args.patch_top_k else args.top_k

        payload = patch_transition_graph(
            graph_data    = payload,
            undercovered  = undercovered,
            id2rel        = id2rel,
            rel_freq      = rel_freq,
            client        = client,
            model         = model_id,
            domain        = domain,
            top_k         = patch_k,
            empirical     = empirical,
            examples      = examples,
            args_debug    = args.debug)

        # Report coverage after patching
        print("\n  Coverage after patch:")
        find_undercovered_relations(
            payload, id2rel, min_edges=args.patch_min_edges)

        # Optionally rescale after patch
        if args.rescale:
            print(f"\n  Applying SOL-3 rescaling after patch...")
            rescaled = rescale_with_empirical(
                payload, empirical, id2rel,
                floor      = args.rescale_floor,
                min_weight = args.rescale_min_weight)
            payload["successors"] = rescaled["successors"]
            payload["inhibitors"] = rescaled["inhibitors"]

        with open(out_path, "wb") as f:
            pickle.dump(payload, f)

        n_succ = sum(len(v) for v in payload["successors"].values())
        n_inh  = sum(len(v) for v in payload["inhibitors"].values())
        print(f"\nSaved patched graph to {out_path}")
        print(f"  Total successor edges : {n_succ}")
        print(f"  Total inhibitor edges : {n_inh}")
        print("Done.")
        return


    if not args.force and os.path.exists(out_path):
        print(f"Already exists: {out_path}. Use --force to recompute.")
        return

    id2rel   = read_id2name(os.path.join(data_dir, "relation2id.txt"))
    rel_freq = load_relation_frequencies(data_dir)
    if not id2rel:
        raise FileNotFoundError(
            f"relation2id.txt not found in {data_dir}")

    print(f"Relations : {len(id2rel)}")
    print(f"Top-5 by frequency: "
          f"{sorted(rel_freq.items(), key=lambda x: -x[1])[:5]}")


    ent_path = os.path.join(data_dir, "entity2id.txt")
    id2ent   = read_entity_id2name(ent_path)
    if id2ent:
        print(f"Entities : {len(id2ent)} loaded from entity2id.txt")
    else:
        print(f"Entities : entity2id.txt not found — "
              f"using numeric IDs in examples")

    examples = {}
    if args.n_examples > 0:
        print(f"\n=== SOL-1: Loading {args.n_examples} training examples "
              f"per relation ===")
        examples = load_relation_examples(
            data_dir, id2rel, id2ent, n_examples=args.n_examples)
        covered = sum(1 for v in examples.values() if len(v) > 0)
        print(f"  Relations with at least 1 example: "
              f"{covered}/{len(id2rel)}")
    else:
        print("\nSOL-1 disabled (--n-examples 0)")

    print(f"\n=== Stage 1: Computing empirical transitions "
          f"(window={args.window}) ===")
    empirical = compute_empirical_transitions(
        data_dir, id2rel, window=args.window)
    print(f"  Empirical transitions computed for "
          f"{len(empirical)} relations")

    domain   = DOMAIN_HINTS.get(args.dataset,
                                 f"temporal KG ({args.dataset})")
    model_id = GROQ_MODELS.get(args.model, args.model)
    api_key  = os.environ.get("GROQ_API_KEY", "")

    if api_key and not args.skip_llm:
        print(f"\n=== Stage 2: LLM transition scoring ({model_id}) ===")
        print(f"  API key: {api_key[:8]}...")
        client     = Groq(api_key=api_key)
        graph_data = build_transition_graph(
            id2rel, rel_freq, client, model_id, domain,
            top_k         = args.top_k,
            max_rel_calls = args.max_rel_calls,
            empirical     = empirical,
            examples      = examples,
            args_debug    = args.debug)          # debug first anchor
    else:
        msg = ("No GROQ_API_KEY — using empirical transitions only."
               if not api_key else "--skip-llm set.")
        print(f"\n{msg}")
        graph_data = empirical_only_transitions(
            id2rel, empirical, top_k=args.top_k)


    if args.rescale and api_key and not args.skip_llm:
        print(f"\n=== SOL-3: Rescaling LLM weights with empirical "
              f"co-occurrence (floor={args.rescale_floor}, "
              f"min_weight={args.rescale_min_weight}) ===")
        rescaled   = rescale_with_empirical(
            graph_data, empirical, id2rel,
            floor      = args.rescale_floor,
            min_weight = args.rescale_min_weight)
        graph_data["successors"] = rescaled["successors"]
        graph_data["inhibitors"] = rescaled["inhibitors"]
    elif args.rescale:
        print("\nSOL-3 skipped — rescaling only applies to LLM-scored graphs.")

    rel2id = {name: r_id for r_id, name in id2rel.items()}
    os.makedirs(data_dir, exist_ok=True)
    payload = {
        "successors": graph_data["successors"],
        "inhibitors": graph_data.get("inhibitors", {}),
        "id2rel":     id2rel,
        "rel2id":     rel2id,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved to {out_path}")

    print(f"\n--- Sample transitions ---")
    shown = 0
    for r_id, succs in list(graph_data["successors"].items())[:5]:
        if not succs:
            continue
        r_name     = id2rel.get(r_id, str(r_id))
        succ_names = [(id2rel.get(r2, str(r2)), w)
                      for r2, w in succs[:3]]
        print(f"  {r_name!r}")
        print(f"    successors : {succ_names}")
        inhs = graph_data.get("inhibitors", {}).get(r_id, [])
        if inhs:
            inh_names = [(id2rel.get(r2, str(r2)), w)
                         for r2, w in inhs[:3]]
            print(f"    inhibitors : {inh_names}")
        shown += 1
        if shown >= 5:
            break

    n_succ = sum(len(v) for v in graph_data["successors"].values())
    n_inh  = sum(len(v) for v in
                 graph_data.get("inhibitors", {}).values())
    print(f"\nTotal successor edges : {n_succ}")
    print(f"Total inhibitor edges : {n_inh}")
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Offline LLM temporal transition scorer.")
    p.add_argument("-d", "--dataset",           required=True)
    p.add_argument("--top-k",       type=int,   default=5)
    p.add_argument("--window",      type=int,   default=7)
    p.add_argument("--max-rel-calls", type=int, default=-1)
    p.add_argument("--skip-llm",    action="store_true", default=False)
    p.add_argument("--model",       type=str,
                   default="llama-3.1-8b-instant")
    p.add_argument("--data-root",   type=str,   default="../data")
    p.add_argument("--force",       action="store_true", default=False)
    p.add_argument("--debug",       action="store_true", default=False,
                   help="Print raw LLM output for the first anchor relation.")
    # Patch mode
    p.add_argument("--patch",       action="store_true", default=False,
                   help="Re-score only undercovered relations (those with "
                        "fewer than --patch-min-edges in either direction). "
                        "Merges new edges into the existing .pkl without "
                        "touching well-covered relations.")
    p.add_argument("--patch-min-edges", type=int, default=2,
                   help="Minimum edges per direction to be considered "
                        "covered.  Relations with fewer edges are re-scored. "
                        "(default 2)")
    p.add_argument("--patch-top-k", type=int, default=0,
                   help="top-k to use during patch run.  0 = use --top-k. "
                        "Set higher (e.g. 10) to request more edges for "
                        "undercovered relations.")
    # SOL-1: number of real training examples to inject per anchor prompt
    # Set to 0 to disable example grounding (reverts to original behaviour).
    p.add_argument("--n-examples",  type=int,   default=3,
                   help="Real training examples per anchor relation injected "
                        "into the LLM prompt (SOL-1). 0 = disabled.")
    # SOL-3: empirical rescaling flags
    p.add_argument("--rescale",     action="store_true", default=False,
                   help="Apply post-hoc empirical rescaling to LLM weights "
                        "(SOL-3). Recommended for GDELT.")
    p.add_argument("--rescale-floor", type=float, default=0.3,
                   help="Minimum fraction of LLM weight retained when "
                        "empirical support is zero (default 0.3).")
    p.add_argument("--rescale-min-weight", type=float, default=0.05,
                   help="Edges below this weight after rescaling are pruned "
                        "(default 0.05).")
    args = p.parse_args()
    run(args)
