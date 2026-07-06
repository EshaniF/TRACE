"""
transition_graph.py
===================
Standalone TransitionGraph class.

Loaded once at model initialisation from rel_transition_graph.pkl.
Provides two mechanisms over base LogCL (zero LLM at training time):

  A. Relation embedding regularisation  (training)
     Adds a soft geometric constraint to the relation embedding matrix:
       - successor pairs  → embeddings pulled CLOSER   (cosine similarity ↑)
       - inhibitor pairs  → embeddings pushed APART    (cosine similarity ↓)
     Grounded in the observation that temporal causal structure should be
     reflected in the geometry of the representation space.

  B. Inference-time re-ranking  (evaluation)
     After the base model scores all candidate entities, applies a
     post-hoc additive correction in log-space derived from the
     transition graph:
       - entities recently involved in INHIBITOR-relation events with the
         query subject receive a score penalty.
       - entities recently involved in SUCCESSOR-relation events with the
         query subject receive a score boost.
     No gradient, no retraining — purely a test-time correction.

     PERFORMANCE NOTE (vectorised rewrite)
     --------------------------------------
     The previous implementation looped over every (test triple, recent
     history event) pair in plain Python and called `.item()` inside the
     inner loop to accumulate `boost`/`penalty`. Each `.item()` forces a
     CPU<->GPU synchronisation. On datasets with dense event graphs per
     snapshot (e.g. GDELT, which has far more events per timestamp than
     ICEWS14/ICEWS18), this produced tens of thousands of blocking syncs
     per snapshot, per epoch (re-ranking runs inside `test()`, which is
     called every `--evaluate-every` epochs during training, not just at
     final test time). That is the dominant source of the GDELT slowdown
     relative to base LogCL.

     This rewrite replaces the double loop with:
       1. A cached dense (R, R) "successor-weight minus inhibitor-weight"
          matrix, built once per device instead of re-derived per event.
       2. A masked, chunked, fully-vectorised computation using
          `scatter_add_`, so subject-matching and relation-weight lookup
          happen as batched tensor ops with zero `.item()` calls in the
          hot path.
     The output is numerically identical to the original loop-based
     version (same boost/penalty formula, same accumulation semantics
     for duplicate matches, same masking of invalid/padded slots and of
     out-of-range relation ids). No methodology change — same inputs,
     same outputs, just computed without a serial Python/GPU-sync loop.

Sentinel convention
-------------------
Index tensors use -1 for empty/padding slots.  Every consumer method
masks out sentinel entries before use so relation 0 is never affected.
"""

import pickle
import torch
import torch.nn.functional as F

_PAD = -1


class TransitionGraph:
    """
    Holds LLM-derived temporal transition structure.

    After .to(device):
      succ_idx  (R, K) long   — top-K successor  relation IDs; _PAD for empty
      succ_wt   (R, K) float  — successor  weights (0 at _PAD slots)
      inh_idx   (R, K) long   — top-K inhibitor relation IDs; _PAD for empty
      inh_wt    (R, K) float  — inhibitor  weights (0 at _PAD slots)
    """

    def __init__(self, successors: dict, inhibitors: dict,
                 id2rel: dict, num_rels: int, top_k: int = 5,
                 rerank_chunk: int = 8192):
        self.num_rels     = num_rels
        self.top_k        = top_k
        self._device       = None
        # Max number of recent-history events processed per vectorised
        # chunk inside rerank_scores(). Bounds the size of the (N, chunk)
        # intermediate tensors so memory stays predictable even on GDELT
        # snapshots with many recent events. Tune down if you hit OOM,
        # tune up for more throughput if memory allows.
        self.rerank_chunk = rerank_chunk

        k = min(top_k, num_rels - 1)

        succ_idx = torch.full((num_rels, k), _PAD, dtype=torch.long)
        succ_wt  = torch.zeros(num_rels, k)
        inh_idx  = torch.full((num_rels, k), _PAD, dtype=torch.long)
        inh_wt   = torch.zeros(num_rels, k)

        for r_id, entries in successors.items():
            if r_id >= num_rels:
                continue
            for i, (r2, w) in enumerate(
                    sorted(entries, key=lambda x: -x[1])[:k]):
                if r2 < num_rels:
                    succ_idx[r_id, i] = r2
                    succ_wt[r_id, i]  = w

        for r_id, entries in inhibitors.items():
            if r_id >= num_rels:
                continue
            for i, (r2, w) in enumerate(
                    sorted(entries, key=lambda x: -x[1])[:k]):
                if r2 < num_rels:
                    inh_idx[r_id, i] = r2
                    inh_wt[r_id, i]  = w

        # CPU copies kept until .to() is called
        self._succ_idx_cpu = succ_idx
        self._succ_wt_cpu  = succ_wt
        self._inh_idx_cpu  = inh_idx
        self._inh_wt_cpu   = inh_wt

        self.succ_idx = None
        self.succ_wt  = None
        self.inh_idx  = None
        self.inh_wt   = None

        # Cached dense (R, R) "successor_weight - inhibitor_weight" matrix,
        # built lazily per-device in _build_weight_diff(). Unscaled by
        # rerank_alpha so the same cache serves any alpha at call time.
        self._weight_diff = None

        n_succ = sum(len(v) for v in successors.values())
        n_inh  = sum(len(v) for v in inhibitors.values())
        print(f"[TransitionGraph] Loaded — "
              f"{n_succ} successor edges, {n_inh} inhibitor edges, "
              f"top-{k} per relation")

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device):
        if self._device == device:
            return
        self.succ_idx = self._succ_idx_cpu.to(device)
        self.succ_wt  = self._succ_wt_cpu.to(device)
        self.inh_idx  = self._inh_idx_cpu.to(device)
        self.inh_wt   = self._inh_wt_cpu.to(device)
        self._device  = device
        self._succ_idx_cpu = None
        self._succ_wt_cpu  = None
        self._inh_idx_cpu  = None
        self._inh_wt_cpu   = None
        # Invalidate cached weight-diff matrix; rebuilt lazily on next use.
        self._weight_diff = None

    def _si(self): return self.succ_idx if self.succ_idx is not None else self._succ_idx_cpu
    def _sw(self): return self.succ_wt  if self.succ_wt  is not None else self._succ_wt_cpu
    def _ii(self): return self.inh_idx  if self.inh_idx  is not None else self._inh_idx_cpu
    def _iw(self): return self.inh_wt   if self.inh_wt   is not None else self._inh_wt_cpu

    def _build_weight_diff(self, device):
        """
        Materialise the sparse top-K successor/inhibitor lists into a
        dense (R, R) matrix: weight_diff[r, r2] = succ_weight(r, r2)
        - inh_weight(r, r2). Built once per device and cached; this is
        what lets rerank_scores() do a single tensor gather instead of
        looping over K candidate relations per event.
        """
        R = self.num_rels
        si, sw = self._si(), self._sw()
        ii, iw = self._ii(), self._iw()

        dense_succ = torch.zeros(R, R, device=device)
        valid_s    = (si != _PAD)
        idx_s      = si.clamp(min=0)
        dense_succ.scatter_add_(1, idx_s, sw * valid_s.float())

        dense_inh = torch.zeros(R, R, device=device)
        valid_i   = (ii != _PAD)
        idx_i     = ii.clamp(min=0)
        dense_inh.scatter_add_(1, idx_i, iw * valid_i.float())

        self._weight_diff = dense_succ - dense_inh  # (R, R)

    # ------------------------------------------------------------------
    # A. Relation embedding regularisation
    # ------------------------------------------------------------------

    def relation_reg_loss(self,
                           rel_emb: torch.Tensor,
                           lambda_succ: float = 1.0,
                           lambda_inh:  float = 1.0) -> torch.Tensor:
        """
        Soft geometric constraint on the relation embedding matrix.

        For each relation r and each of its top-K successors / inhibitors,
        compute cosine similarity between their embeddings and apply:

          successor  loss : relu(margin - cos(r, r_succ))
            → penalise when successor embeddings are NOT similar enough.
            → pulls successor pairs closer in embedding space.

          inhibitor loss  : relu(margin + cos(r, r_inh))
            → penalise when inhibitor embeddings are NOT dissimilar enough.
            → pushes inhibitor pairs apart (towards orthogonality / opposition).

        Both use margin = 0.5, balancing the two objectives symmetrically.
        Sentinel slots (_PAD) are masked out and contribute zero loss.

        Parameters
        ----------
        rel_emb     : (2R, h_dim) — full relation embedding matrix.
                      Only the base R rows [0:num_rels] are used; inverse
                      relations are not constrained directly.
        lambda_succ : weight for successor similarity term
        lambda_inh  : weight for inhibitor dissimilarity term

        Returns
        -------
        Scalar loss tensor in the computation graph.
        """
        device  = rel_emb.device
        R       = self.num_rels
        # Only base relation embeddings, normalised for cosine similarity
        base    = F.normalize(rel_emb[:R], dim=1)          # (R, h)

        si = self._si()   # (R, K)
        sw = self._sw()   # (R, K)
        ii = self._ii()   # (R, K)
        iw = self._iw()   # (R, K)

        margin = 0.5

        # ── successor term ──────────────────────────────────────────────
        succ_valid = (si != _PAD)                           # (R, K)
        succ_ids   = si.clamp(min=0)                        # safe index

        # (R, K, h) — embeddings of all successor targets
        succ_emb   = base[succ_ids]                         # (R, K, h)
        # (R, K, h) → (R, K) cosine similarity
        cos_succ   = (base.unsqueeze(1) * succ_emb).sum(dim=2)  # (R, K)
        # Weight by transition confidence and mask padding
        hinge_succ = F.relu(margin - cos_succ) * sw * succ_valid.float()
        loss_succ  = (lambda_succ * hinge_succ.sum()
                      / succ_valid.float().sum().clamp(min=1))

        # ── inhibitor term ───────────────────────────────────────────────
        inh_valid  = (ii != _PAD)                           # (R, K)
        inh_ids    = ii.clamp(min=0)
        inh_emb    = base[inh_ids]                          # (R, K, h)
        cos_inh    = (base.unsqueeze(1) * inh_emb).sum(dim=2)   # (R, K)
        # Penalise when cosine > -margin (not sufficiently dissimilar)
        hinge_inh  = F.relu(margin + cos_inh) * iw * inh_valid.float()
        loss_inh   = (lambda_inh * hinge_inh.sum()
                      / inh_valid.float().sum().clamp(min=1))

        return loss_succ + loss_inh

    # ------------------------------------------------------------------
    # B. Inference-time re-ranking (vectorised)
    # ------------------------------------------------------------------

    def rerank_scores(self,
                       log_scores:    torch.Tensor,
                       test_triplets: torch.Tensor,
                       recent_events: list,
                       rerank_alpha:  float = 0.3) -> torch.Tensor:
        """
        Post-hoc re-ranking of entity scores using recent event history.

        log_scores has shape (N, num_ents) where N = number of test triples.
        Each row i corresponds to test_triplets[i] = (s_i, r_i, o_i, ...).

        Vectorised equivalent of the original per-(triple, event) Python
        loop. For every test row i and every recent event (s_m, r_m, o_m)
        with s_m == s_i, the contribution to delta[i, o_m] is:

            alpha * (succ_weight(r_i, r_m) - inh_weight(r_i, r_m))

        which is exactly the original `alpha * (succ_match.sum()
        - inh_match.sum())`, just looked up from the cached dense
        (R, R) weight_diff matrix instead of scanning the K successor/
        inhibitor slots of r_i in a Python loop. Contributions from
        multiple matching events to the same (i, o_m) are summed via
        `scatter_add_`, matching the original's `delta[i, o_hist] += ...`
        accumulation. Events with an out-of-range relation id (which the
        original loop would simply never match against anything, since
        successor/inhibitor ids are always < num_rels) are masked to
        contribute zero, preserving identical semantics.

        Parameters
        ----------
        log_scores    : (N, num_ents) — log-softmax scores from base model
        test_triplets : (N, 3+) LongTensor — columns 0=s, 1=r (further
                        columns are ignored)
        recent_events : list of (s, r, o) int tuples from recent history
        rerank_alpha  : correction strength (default 0.3)

        Returns
        -------
        (N, num_ents) corrected log-scores, same shape as input
        """
        if not recent_events:
            return log_scores

        device = log_scores.device
        N, E = log_scores.shape

        if self._weight_diff is None or self._weight_diff.device != device:
            self._build_weight_diff(device)

        s_col = test_triplets[:, 0].to(device).long()
        r_col = test_triplets[:, 1].to(device).long() % self.num_rels

        events = torch.as_tensor(recent_events, dtype=torch.long,
                                  device=device)  # (M, 3)
        s_h, r_h, o_h = events[:, 0], events[:, 1], events[:, 2]

        # Drop events whose object is out of range for this score matrix.
        valid_o = o_h < E
        if not bool(valid_o.all()):
            s_h, r_h, o_h = s_h[valid_o], r_h[valid_o], o_h[valid_o]

        # Only keep history events whose subject actually appears among
        # the current test triples — shrinks the (N, M) working set for
        # dense/high-degree datasets like GDELT without changing results
        # (events for other subjects never match anyway).
        test_subjects = torch.unique(s_col)
        keep = torch.isin(s_h, test_subjects)
        s_h, r_h, o_h = s_h[keep], r_h[keep], o_h[keep]

        M = s_h.shape[0]
        if M == 0:
            return log_scores

        # Relation ids outside [0, num_rels) can never match a successor/
        # inhibitor entry (those are always < num_rels) — mask instead of
        # modding, so behaviour matches the original loop exactly.
        r_h_valid = (r_h >= 0) & (r_h < self.num_rels)
        r_h_safe  = r_h.clamp(min=0, max=self.num_rels - 1)

        delta = torch.zeros_like(log_scores)

        chunk = max(1, min(M, self.rerank_chunk))
        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            s_blk   = s_h[start:end]
            rh_blk  = r_h_safe[start:end]
            rhv_blk = r_h_valid[start:end]
            o_blk   = o_h[start:end]
            width   = end - start

            # (N, width) subject match mask
            mask = s_col.unsqueeze(1).eq(s_blk.unsqueeze(0))
            if not bool(mask.any()):
                continue

            # (N, width) weight lookup: weight_diff[r_i, r_hist]
            w = self._weight_diff[r_col][:, rh_blk]
            w = w * rhv_blk.unsqueeze(0).float()

            contrib = rerank_alpha * w * mask.float()

            idx = o_blk.unsqueeze(0).expand(N, width)
            delta.scatter_add_(1, idx, contrib)

        return log_scores + delta

# """
# transition_graph.py
# ===================
# Standalone TransitionGraph class.

# Loaded once at model initialisation from rel_transition_graph.pkl.
# Provides two mechanisms over base LogCL (zero LLM at training time):

#   A. Relation embedding regularisation  (training)
#      Adds a soft geometric constraint to the relation embedding matrix:
#        - successor pairs  → embeddings pulled CLOSER   (cosine similarity ↑)
#        - inhibitor pairs  → embeddings pushed APART    (cosine similarity ↓)
#      Grounded in the observation that temporal causal structure should be
#      reflected in the geometry of the representation space.

#   B. Inference-time re-ranking  (evaluation)
#      After the base model scores all candidate entities, applies a
#      post-hoc multiplicative penalty derived from the transition graph:
#        - entities recently involved in INHIBITOR-relation events with the
#          query subject receive a score penalty.
#        - entities recently involved in SUCCESSOR-relation events with the
#          query subject receive a score boost.
#      No gradient, no retraining — purely a test-time correction.

# Sentinel convention
# -------------------
# Index tensors use -1 for empty/padding slots.  Every consumer method
# masks out sentinel entries before use so relation 0 is never affected.
# """

# import pickle
# import torch
# import torch.nn.functional as F

# _PAD = -1


# class TransitionGraph:
#     """
#     Holds LLM-derived temporal transition structure.

#     After .to(device):
#       succ_idx  (R, K) long   — top-K successor  relation IDs; _PAD for empty
#       succ_wt   (R, K) float  — successor  weights (0 at _PAD slots)
#       inh_idx   (R, K) long   — top-K inhibitor relation IDs; _PAD for empty
#       inh_wt    (R, K) float  — inhibitor  weights (0 at _PAD slots)
#     """

#     def __init__(self, successors: dict, inhibitors: dict,
#                  id2rel: dict, num_rels: int, top_k: int = 5):
#         self.num_rels = num_rels
#         self.top_k    = top_k
#         self._device  = None

#         k = min(top_k, num_rels - 1)

#         succ_idx = torch.full((num_rels, k), _PAD, dtype=torch.long)
#         succ_wt  = torch.zeros(num_rels, k)
#         inh_idx  = torch.full((num_rels, k), _PAD, dtype=torch.long)
#         inh_wt   = torch.zeros(num_rels, k)

#         for r_id, entries in successors.items():
#             if r_id >= num_rels:
#                 continue
#             for i, (r2, w) in enumerate(
#                     sorted(entries, key=lambda x: -x[1])[:k]):
#                 if r2 < num_rels:
#                     succ_idx[r_id, i] = r2
#                     succ_wt[r_id, i]  = w

#         for r_id, entries in inhibitors.items():
#             if r_id >= num_rels:
#                 continue
#             for i, (r2, w) in enumerate(
#                     sorted(entries, key=lambda x: -x[1])[:k]):
#                 if r2 < num_rels:
#                     inh_idx[r_id, i] = r2
#                     inh_wt[r_id, i]  = w

#         # CPU copies kept until .to() is called
#         self._succ_idx_cpu = succ_idx
#         self._succ_wt_cpu  = succ_wt
#         self._inh_idx_cpu  = inh_idx
#         self._inh_wt_cpu   = inh_wt

#         self.succ_idx = None
#         self.succ_wt  = None
#         self.inh_idx  = None
#         self.inh_wt   = None

#         n_succ = sum(len(v) for v in successors.values())
#         n_inh  = sum(len(v) for v in inhibitors.values())
#         print(f"[TransitionGraph] Loaded — "
#               f"{n_succ} successor edges, {n_inh} inhibitor edges, "
#               f"top-{k} per relation")

#     # ------------------------------------------------------------------
#     # Device management
#     # ------------------------------------------------------------------

#     def to(self, device):
#         if self._device == device:
#             return
#         self.succ_idx = self._succ_idx_cpu.to(device)
#         self.succ_wt  = self._succ_wt_cpu.to(device)
#         self.inh_idx  = self._inh_idx_cpu.to(device)
#         self.inh_wt   = self._inh_wt_cpu.to(device)
#         self._device  = device
#         self._succ_idx_cpu = None
#         self._succ_wt_cpu  = None
#         self._inh_idx_cpu  = None
#         self._inh_wt_cpu   = None

#     def _si(self): return self.succ_idx if self.succ_idx is not None else self._succ_idx_cpu
#     def _sw(self): return self.succ_wt  if self.succ_wt  is not None else self._succ_wt_cpu
#     def _ii(self): return self.inh_idx  if self.inh_idx  is not None else self._inh_idx_cpu
#     def _iw(self): return self.inh_wt   if self.inh_wt   is not None else self._inh_wt_cpu

#     # ------------------------------------------------------------------
#     # A. Relation embedding regularisation
#     # ------------------------------------------------------------------

#     def relation_reg_loss(self,
#                            rel_emb: torch.Tensor,
#                            lambda_succ: float = 1.0,
#                            lambda_inh:  float = 1.0) -> torch.Tensor:
#         """
#         Soft geometric constraint on the relation embedding matrix.

#         For each relation r and each of its top-K successors / inhibitors,
#         compute cosine similarity between their embeddings and apply:

#           successor  loss : relu(margin - cos(r, r_succ))
#             → penalise when successor embeddings are NOT similar enough.
#             → pulls successor pairs closer in embedding space.

#           inhibitor loss  : relu(margin + cos(r, r_inh))
#             → penalise when inhibitor embeddings are NOT dissimilar enough.
#             → pushes inhibitor pairs apart (towards orthogonality / opposition).

#         Both use margin = 0.5, balancing the two objectives symmetrically.
#         Sentinel slots (_PAD) are masked out and contribute zero loss.

#         Parameters
#         ----------
#         rel_emb     : (2R, h_dim) — full relation embedding matrix.
#                       Only the base R rows [0:num_rels] are used; inverse
#                       relations are not constrained directly.
#         lambda_succ : weight for successor similarity term
#         lambda_inh  : weight for inhibitor dissimilarity term

#         Returns
#         -------
#         Scalar loss tensor in the computation graph.
#         """
#         device  = rel_emb.device
#         R       = self.num_rels
#         # Only base relation embeddings, normalised for cosine similarity
#         base    = F.normalize(rel_emb[:R], dim=1)          # (R, h)

#         si = self._si()   # (R, K)
#         sw = self._sw()   # (R, K)
#         ii = self._ii()   # (R, K)
#         iw = self._iw()   # (R, K)

#         margin = 0.5

#         # ── successor term ──────────────────────────────────────────────
#         succ_valid = (si != _PAD)                           # (R, K)
#         succ_ids   = si.clamp(min=0)                        # safe index

#         # (R, K, h) — embeddings of all successor targets
#         succ_emb   = base[succ_ids]                         # (R, K, h)
#         # (R, K, h) → (R, K) cosine similarity
#         cos_succ   = (base.unsqueeze(1) * succ_emb).sum(dim=2)  # (R, K)
#         # Weight by transition confidence and mask padding
#         hinge_succ = F.relu(margin - cos_succ) * sw * succ_valid.float()
#         loss_succ  = (lambda_succ * hinge_succ.sum()
#                       / succ_valid.float().sum().clamp(min=1))

#         # ── inhibitor term ───────────────────────────────────────────────
#         inh_valid  = (ii != _PAD)                           # (R, K)
#         inh_ids    = ii.clamp(min=0)
#         inh_emb    = base[inh_ids]                          # (R, K, h)
#         cos_inh    = (base.unsqueeze(1) * inh_emb).sum(dim=2)   # (R, K)
#         # Penalise when cosine > -margin (not sufficiently dissimilar)
#         hinge_inh  = F.relu(margin + cos_inh) * iw * inh_valid.float()
#         loss_inh   = (lambda_inh * hinge_inh.sum()
#                       / inh_valid.float().sum().clamp(min=1))

#         return loss_succ + loss_inh

#     # ------------------------------------------------------------------
#     # B. Inference-time re-ranking
#     # ------------------------------------------------------------------

#     def rerank_scores(self,
#                        log_scores:    torch.Tensor,
#                        test_triplets: torch.Tensor,
#                        recent_events: list,
#                        rerank_alpha:  float = 0.3) -> torch.Tensor:
#         """
#         Post-hoc re-ranking of entity scores using recent event history.

#         log_scores has shape (N, num_ents) where N = number of test triples.
#         Each row i corresponds to test_triplets[i] = (s_i, r_i, o_i, ...).
#         The loop runs over N rows so row index i always aligns correctly.

#         Root cause of the previous IndexError
#         --------------------------------------
#         The old signature accepted query_r_ids (B,) and query_s_ids (B,)
#         derived from unique subject entities (B = |uniq_e|).  But
#         log_scores has N rows where N = number of triples, and B != N in
#         general (multiple triples can share the same subject).  Iterating
#         range(B) and indexing log_scores[i] caused an IndexError whenever
#         N < B, and silently applied the wrong correction whenever N > B.

#         Fix: derive s_i and r_i directly from test_triplets row i so the
#         loop variable i is always a valid index into log_scores.

#         Parameters
#         ----------
#         log_scores    : (N, num_ents) — log-softmax scores from base model
#         test_triplets : (N, 3+) LongTensor — columns 0=s, 1=r (further
#                         columns are ignored)
#         recent_events : list of (s, r, o) int tuples from recent history
#         rerank_alpha  : correction strength (default 0.3)

#         Returns
#         -------
#         (N, num_ents) corrected log-scores, same shape as input
#         """
#         N, E = log_scores.shape
#         delta = torch.zeros_like(log_scores)

#         si = self._si()
#         sw = self._sw()
#         ii = self._ii()
#         iw = self._iw()

#         # Build subject → [(relation, object)] lookup from recent history.
#         subj_hist: dict = {}
#         for (s, r, o) in recent_events:
#             if s not in subj_hist:
#                 subj_hist[s] = []
#             subj_hist[s].append((r, o))

#         # test_triplets may be on GPU — bring columns to CPU as plain lists
#         # once rather than calling .item() inside the inner loop.
#         s_col = test_triplets[:, 0].tolist()   # length N
#         r_col = test_triplets[:, 1].tolist()   # length N

#         for i in range(N):
#             s_i   = int(s_col[i])
#             r_i   = int(r_col[i]) % self.num_rels
#             hist  = subj_hist.get(s_i, [])
#             if not hist:
#                 continue

#             s_ids_r   = si[r_i]               # (K,)
#             s_wts_r   = sw[r_i]               # (K,)
#             s_valid_r = s_ids_r != _PAD

#             i_ids_r   = ii[r_i]               # (K,)
#             i_wts_r   = iw[r_i]               # (K,)
#             i_valid_r = i_ids_r != _PAD

#             for (r_hist, o_hist) in hist:
#                 if o_hist >= E:
#                     continue

#                 succ_match = ((s_ids_r == r_hist) & s_valid_r).float() * s_wts_r
#                 boost      = rerank_alpha * succ_match.sum().item()

#                 inh_match  = ((i_ids_r == r_hist) & i_valid_r).float() * i_wts_r
#                 penalty    = rerank_alpha * inh_match.sum().item()

#                 delta[i, o_hist] += boost - penalty

#         return log_scores + delta

# # """
# # transition_graph.py
# # ===================
# # Standalone TransitionGraph class.

# # Loaded once at model initialisation from rel_transition_graph.pkl.
# # Provides three training-time additions over LogCL:

# #   1. Transition-aware hard negatives
# #      For query (s, r, ?, t), entities that recently appeared in
# #      inhibited-relation events with s are strong hard negatives.

# #   2. Transition regularisation loss
# #      Pushes entity prediction scores higher for successor-relation
# #      events and lower for inhibitor-relation events.

# #   3. Transition-conditioned history reweighting
# #      Upweights history snapshots containing predecessor relations
# #      of the current query relation.

# #   4. Zero-shot unseen relation initialisation
# #      Initialises unseen relation embeddings as a weighted combination
# #      of known successor and predecessor relation embeddings.

# # Fix (3): Index tensors now use sentinel value -1 for empty/padding slots
# # instead of 0.  Every consumer method masks out sentinel entries before use,
# # so relation 0 never receives spurious signal from padding.
# # """

# # import os
# # import pickle
# # import torch
# # import torch.nn.functional as F
# # from collections import defaultdict


# # # Sentinel value stored in index tensors for empty padding slots.
# # # Must be negative so it is never a valid relation ID (IDs are >= 0).
# # _PAD = -1


# # class TransitionGraph:
# #     """
# #     Holds LLM-derived temporal transition structure.

# #     After .to(device):
# #       succ_idx     (R, K) long   — top-K successor relation IDs per rel;
# #                                    empty slots filled with _PAD (-1)
# #       succ_wt      (R, K) float  — their transition weights (0 at _PAD)
# #       inh_idx      (R, K) long   — top-K inhibitor relation IDs per rel;
# #                                    empty slots filled with _PAD (-1)
# #       inh_wt       (R, K) float  — their inhibition weights (0 at _PAD)
# #     """

# #     def __init__(self, successors: dict, inhibitors: dict,
# #                  id2rel: dict, num_rels: int, top_k: int = 5):
# #         """
# #         successors : {r_id: [(r_id, weight), ...]}
# #         inhibitors : {r_id: [(r_id, weight), ...]}
# #         num_rels   : base relation count (no inverses)
# #         top_k      : neighbourhood size stored per relation
# #         """
# #         self.num_rels = num_rels
# #         self.top_k    = top_k
# #         self._device  = None

# #         k = min(top_k, num_rels - 1)

# #         # Build dense (R, K) tensors.
# #         # FIX: pad with _PAD (-1) instead of 0 so that relation 0 is never
# #         # spuriously matched during masking or broadcasting operations.
# #         succ_idx = torch.full((num_rels, k), _PAD, dtype=torch.long)
# #         succ_wt  = torch.zeros(num_rels, k, dtype=torch.float)
# #         inh_idx  = torch.full((num_rels, k), _PAD, dtype=torch.long)
# #         inh_wt   = torch.zeros(num_rels, k, dtype=torch.float)

# #         for r_id, entries in successors.items():
# #             if r_id >= num_rels:
# #                 continue
# #             entries_sorted = sorted(entries, key=lambda x: -x[1])[:k]
# #             for i, (r2, w) in enumerate(entries_sorted):
# #                 if r2 < num_rels:
# #                     succ_idx[r_id, i] = r2
# #                     succ_wt[r_id, i]  = w

# #         for r_id, entries in inhibitors.items():
# #             if r_id >= num_rels:
# #                 continue
# #             entries_sorted = sorted(entries, key=lambda x: -x[1])[:k]
# #             for i, (r2, w) in enumerate(entries_sorted):
# #                 if r2 < num_rels:
# #                     inh_idx[r_id, i] = r2
# #                     inh_wt[r_id, i]  = w

# #         self._succ_idx_cpu = succ_idx
# #         self._succ_wt_cpu  = succ_wt
# #         self._inh_idx_cpu  = inh_idx
# #         self._inh_wt_cpu   = inh_wt

# #         # GPU tensors set by .to()
# #         self.succ_idx = None
# #         self.succ_wt  = None
# #         self.inh_idx  = None
# #         self.inh_wt   = None

# #         n_succ = sum(len(v) for v in successors.values())
# #         n_inh  = sum(len(v) for v in inhibitors.values())
# #         print(f"[TransitionGraph] Loaded — "
# #               f"{n_succ} successor edges, {n_inh} inhibitor edges, "
# #               f"top-{k} per relation")

# #     def to(self, device):
# #         if self._device == device:
# #             return
# #         self.succ_idx = self._succ_idx_cpu.to(device)
# #         self.succ_wt  = self._succ_wt_cpu.to(device)
# #         self.inh_idx  = self._inh_idx_cpu.to(device)
# #         self.inh_wt   = self._inh_wt_cpu.to(device)
# #         self._device  = device
# #         # Free CPU copies
# #         self._succ_idx_cpu = None
# #         self._succ_wt_cpu  = None
# #         self._inh_idx_cpu  = None
# #         self._inh_wt_cpu   = None

# #     # ------------------------------------------------------------------
# #     # Internal helpers
# #     # ------------------------------------------------------------------

# #     def _active_succ_idx(self) -> torch.Tensor:
# #         """Return the live successor index tensor (GPU or CPU)."""
# #         return self.succ_idx if self.succ_idx is not None \
# #                else self._succ_idx_cpu

# #     def _active_succ_wt(self) -> torch.Tensor:
# #         return self.succ_wt if self.succ_wt is not None \
# #                else self._succ_wt_cpu

# #     def _active_inh_idx(self) -> torch.Tensor:
# #         return self.inh_idx if self.inh_idx is not None \
# #                else self._inh_idx_cpu

# #     def _active_inh_wt(self) -> torch.Tensor:
# #         return self.inh_wt if self.inh_wt is not None \
# #                else self._inh_wt_cpu

# #     # ------------------------------------------------------------------
# #     # Addition 1 — transition-aware hard negative entity IDs
# #     # ------------------------------------------------------------------

# #     def get_inhibitor_relations(self,
# #                                  r_ids: torch.Tensor) -> torch.Tensor:
# #         """
# #         For each query relation r, return the IDs of its inhibitor
# #         relations — those that are unlikely to co-occur with r.

# #         r_ids  : (B,)   query relation IDs
# #         Returns: (B, K) inhibitor relation IDs; _PAD (-1) for empty slots.
# #                  Callers must mask entries where value == _PAD.
# #         """
# #         r = r_ids.clamp(0, self.num_rels - 1)
# #         return self._active_inh_idx()[r]   # (B, K)

# #     def get_successor_relations(self,
# #                                   r_ids: torch.Tensor) -> torch.Tensor:
# #         """
# #         For each query relation r, return the IDs of its successor
# #         relations — those likely to follow r.

# #         r_ids  : (B,)   query relation IDs
# #         Returns: (B, K) successor relation IDs; _PAD (-1) for empty slots.
# #                  Callers must mask entries where value == _PAD.
# #         """
# #         r = r_ids.clamp(0, self.num_rels - 1)
# #         return self._active_succ_idx()[r]   # (B, K)

# #     # ------------------------------------------------------------------
# #     # Addition 3 — history snapshot reweighting
# #     # ------------------------------------------------------------------

# #     def history_reweight_masks(self,
# #                                 query_r_ids: torch.Tensor,
# #                                 snapshot_r_ids: torch.Tensor) -> tuple:
# #         """
# #         Return per-event successor/inhibitor boolean masks for a snapshot.

# #         Hardcoded scalar weights (previously 2.0 for successors, 0.5 for
# #         inhibitors) have been removed entirely.  The caller (RecurrentRGCN)
# #         owns the weighting arithmetic and applies its own learned scalar
# #         parameters reweight_succ and reweight_inh.  This keeps
# #         TransitionGraph as a pure data structure with no nn.Parameters.

# #         query_r_ids    : (B,)  relation ID of each query in the batch
# #         snapshot_r_ids : (T,)  relation IDs present in one history snapshot

# #         Returns
# #         -------
# #         is_succ : (B, T) bool — True where snapshot event relation is a
# #                                 successor of the query relation
# #         is_inh  : (B, T) bool — True where snapshot event relation is an
# #                                 inhibitor of the query relation
# #         """
# #         B = query_r_ids.size(0)
# #         T = snapshot_r_ids.size(0)

# #         r    = query_r_ids.clamp(0, self.num_rels - 1)
# #         succ = self._active_succ_idx()[r]    # (B, K_s)
# #         inh  = self._active_inh_idx()[r]     # (B, K_i)

# #         succ_valid = succ != _PAD            # (B, K_s)
# #         inh_valid  = inh  != _PAD            # (B, K_i)

# #         K_s = succ.size(1)
# #         K_i = inh.size(1)

# #         # Broadcast: snap (1, T, 1) vs succ/inh (B, 1, K)
# #         snap_exp_s = snapshot_r_ids.view(1, T, 1).expand(B, T, K_s)
# #         snap_exp_i = snapshot_r_ids.view(1, T, 1).expand(B, T, K_i)
# #         succ_exp   = succ.unsqueeze(1).expand(B, T, K_s)
# #         inh_exp    = inh.unsqueeze(1).expand(B, T, K_i)

# #         succ_valid_exp = succ_valid.unsqueeze(1).expand(B, T, K_s)
# #         inh_valid_exp  = inh_valid.unsqueeze(1).expand(B, T, K_i)

# #         is_succ = ((snap_exp_s == succ_exp) & succ_valid_exp).any(dim=2)  # (B,T)
# #         is_inh  = ((snap_exp_i == inh_exp)  & inh_valid_exp).any(dim=2)   # (B,T)

# #         return is_succ, is_inh

# #     # ------------------------------------------------------------------
# #     # Addition 4 — zero-shot unseen relation initialisation
# #     # ------------------------------------------------------------------

# #     def init_unseen_relation(self,
# #                               r_unseen: int,
# #                               rel_emb: torch.Tensor) -> torch.Tensor:
# #         """
# #         Initialise embedding for an unseen relation using its known
# #         successor relations from the transition graph.

# #         FIX: sentinel entries (_PAD) are explicitly excluded from the
# #         weight mask so they do not contribute to the mean embedding.

# #         r_unseen : scalar int — relation ID not seen in training
# #         rel_emb  : (2R, h_dim) — current relation embedding matrix
# #         Returns  : (h_dim,) initialised embedding
# #         """
# #         if self._device is None:
# #             device = rel_emb.device
# #             self.to(device)

# #         succ_ids = self._active_succ_idx()
# #         succ_wts = self._active_succ_wt()

# #         if r_unseen >= self.num_rels:
# #             return rel_emb[r_unseen % self.num_rels].clone()

# #         ids = succ_ids[r_unseen]    # (K,)
# #         wts = succ_wts[r_unseen]    # (K,)

# #         # FIX: mask out sentinel entries in addition to zero-weight entries
# #         mask = (wts > 0) & (ids != _PAD)
# #         if not mask.any():
# #             return rel_emb[:self.num_rels].mean(dim=0)

# #         ids = ids[mask].to(rel_emb.device)
# #         wts = wts[mask].to(rel_emb.device)
# #         wts = wts / wts.sum()

# #         emb = (wts.unsqueeze(1) * rel_emb[ids]).sum(dim=0)
# #         return emb

# # """
# # transition_graph.py
# # ===================
# # Standalone TransitionGraph class.

# # Loaded once at model initialisation from rel_transition_graph.pkl.
# # Provides three training-time additions over LogCL:

# #   1. Transition-aware hard negatives
# #      For query (s, r, ?, t), entities that recently appeared in
# #      inhibited-relation events with s are strong hard negatives.

# #   2. Transition regularisation loss
# #      Pushes entity prediction scores higher for successor-relation
# #      events and lower for inhibitor-relation events.

# #   3. Transition-conditioned history reweighting
# #      Upweights history snapshots containing predecessor relations
# #      of the current query relation.

# #   4. Zero-shot unseen relation initialisation
# #      Initialises unseen relation embeddings as a weighted combination
# #      of known successor and predecessor relation embeddings.
# # """

# # import os
# # import pickle
# # import torch
# # import torch.nn.functional as F
# # from collections import defaultdict


# # class TransitionGraph:
# #     """
# #     Holds LLM-derived temporal transition structure.

# #     After .to(device):
# #       succ_idx     (R, K) long   — top-K successor relation IDs per rel
# #       succ_wt      (R, K) float  — their transition weights
# #       inh_idx      (R, K) long   — top-K inhibitor relation IDs per rel
# #       inh_wt       (R, K) float  — their inhibition weights
# #     """

# #     def __init__(self, successors: dict, inhibitors: dict,
# #                  id2rel: dict, num_rels: int, top_k: int = 5):
# #         """
# #         successors : {r_id: [(r_id, weight), ...]}
# #         inhibitors : {r_id: [(r_id, weight), ...]}
# #         num_rels   : base relation count (no inverses)
# #         top_k      : neighbourhood size stored per relation
# #         """
# #         self.num_rels = num_rels
# #         self.top_k    = top_k
# #         self._device  = None

# #         k = min(top_k, num_rels - 1)

# #         # Build dense (R, K) tensors — pad with zeros for missing entries
# #         succ_idx = torch.zeros(num_rels, k, dtype=torch.long)
# #         succ_wt  = torch.zeros(num_rels, k, dtype=torch.float)
# #         inh_idx  = torch.zeros(num_rels, k, dtype=torch.long)
# #         inh_wt   = torch.zeros(num_rels, k, dtype=torch.float)

# #         for r_id, entries in successors.items():
# #             if r_id >= num_rels:
# #                 continue
# #             entries_sorted = sorted(entries, key=lambda x: -x[1])[:k]
# #             for i, (r2, w) in enumerate(entries_sorted):
# #                 if r2 < num_rels:
# #                     succ_idx[r_id, i] = r2
# #                     succ_wt[r_id, i]  = w

# #         for r_id, entries in inhibitors.items():
# #             if r_id >= num_rels:
# #                 continue
# #             entries_sorted = sorted(entries, key=lambda x: -x[1])[:k]
# #             for i, (r2, w) in enumerate(entries_sorted):
# #                 if r2 < num_rels:
# #                     inh_idx[r_id, i] = r2
# #                     inh_wt[r_id, i]  = w

# #         self._succ_idx_cpu = succ_idx
# #         self._succ_wt_cpu  = succ_wt
# #         self._inh_idx_cpu  = inh_idx
# #         self._inh_wt_cpu   = inh_wt

# #         # GPU tensors set by .to()
# #         self.succ_idx = None
# #         self.succ_wt  = None
# #         self.inh_idx  = None
# #         self.inh_wt   = None

# #         n_succ = sum(len(v) for v in successors.values())
# #         n_inh  = sum(len(v) for v in inhibitors.values())
# #         print(f"[TransitionGraph] Loaded — "
# #               f"{n_succ} successor edges, {n_inh} inhibitor edges, "
# #               f"top-{k} per relation")

# #     def to(self, device):
# #         if self._device == device:
# #             return
# #         self.succ_idx = self._succ_idx_cpu.to(device)
# #         self.succ_wt  = self._succ_wt_cpu.to(device)
# #         self.inh_idx  = self._inh_idx_cpu.to(device)
# #         self.inh_wt   = self._inh_wt_cpu.to(device)
# #         self._device  = device
# #         # Free CPU copies
# #         self._succ_idx_cpu = None
# #         self._succ_wt_cpu  = None
# #         self._inh_idx_cpu  = None
# #         self._inh_wt_cpu   = None

# #     # ------------------------------------------------------------------
# #     # Addition 1 — transition-aware hard negative entity IDs
# #     # ------------------------------------------------------------------

# #     def get_inhibitor_relations(self,
# #                                  r_ids: torch.Tensor) -> torch.Tensor:
# #         """
# #         For each query relation r, return the IDs of its inhibitor
# #         relations — those that are unlikely to co-occur with r.

# #         r_ids  : (B,)   query relation IDs
# #         Returns: (B, K) inhibitor relation IDs
# #         """
# #         r = r_ids.clamp(0, self.num_rels - 1)
# #         return self.inh_idx[r]   # (B, K)

# #     def get_successor_relations(self,
# #                                   r_ids: torch.Tensor) -> torch.Tensor:
# #         """
# #         For each query relation r, return the IDs of its successor
# #         relations — those likely to follow r.

# #         r_ids  : (B,)   query relation IDs
# #         Returns: (B, K) successor relation IDs
# #         """
# #         r = r_ids.clamp(0, self.num_rels - 1)
# #         return self.succ_idx[r]   # (B, K)

# #     # ------------------------------------------------------------------
# #     # Addition 3 — history snapshot reweighting
# #     # ------------------------------------------------------------------

# #     def history_reweight(self,
# #                         query_r_ids: torch.Tensor,
# #                         snapshot_r_ids: torch.Tensor) -> torch.Tensor:
# #         """
# #         Fully vectorised history reweighting. Zero Python loops.
# #         """
# #         B      = query_r_ids.size(0)
# #         T      = snapshot_r_ids.size(0)
# #         device = query_r_ids.device

# #         weights = torch.ones(B, T, device=device)

# #         r      = query_r_ids.clamp(0, self.num_rels - 1)
# #         succ   = self.succ_idx[r]    # (B, K)
# #         inh    = self.inh_idx[r]     # (B, K)

# #         # snap_r: (T,) → expand to (B, T, 1)
# #         # succ:   (B, K) → expand to (B, 1, K)
# #         snap_exp = snapshot_r_ids.unsqueeze(0).unsqueeze(2).expand(
# #             B, T, succ.size(1))                             # (B, T, K)
# #         succ_exp = succ.unsqueeze(1).expand(B, T, succ.size(1))  # (B, T, K)
# #         inh_exp  = inh.unsqueeze(1).expand(B, T, inh.size(1))    # (B, T, K)

# #         is_succ = (snap_exp == succ_exp).any(dim=2)         # (B, T) bool
# #         is_inh  = (snap_exp == inh_exp).any(dim=2)          # (B, T) bool

# #         weights = torch.where(is_succ,
# #                             torch.full_like(weights, 2.0), weights)
# #         weights = torch.where(is_inh,
# #                             torch.full_like(weights, 0.5), weights)
# #         return weights

# #     # ------------------------------------------------------------------
# #     # Addition 4 — zero-shot unseen relation initialisation
# #     # ------------------------------------------------------------------

# #     def init_unseen_relation(self,
# #                               r_unseen: int,
# #                               rel_emb: torch.Tensor) -> torch.Tensor:
# #         """
# #         Initialise embedding for an unseen relation using its known
# #         successor and inhibitor relations from the transition graph.

# #         The unseen relation's embedding is the weighted mean of its
# #         successor relation embeddings. Inhibitors are excluded — they
# #         represent what the relation prevents, not what it resembles.

# #         r_unseen : scalar int — relation ID not seen in training
# #         rel_emb  : (2R, h_dim) — current relation embedding matrix
# #         Returns  : (h_dim,) initialised embedding
# #         """
# #         if self._device is None:
# #             device = rel_emb.device
# #             self.to(device)

# #         succ_ids = self._succ_idx_cpu if self.succ_idx is None \
# #                    else self.succ_idx
# #         succ_wts = self._succ_wt_cpu  if self.succ_wt  is None \
# #                    else self.succ_wt

# #         if r_unseen >= self.num_rels:
# #             return rel_emb[r_unseen % self.num_rels].clone()

# #         ids = succ_ids[r_unseen]    # (K,)
# #         wts = succ_wts[r_unseen]    # (K,)

# #         # Only use entries with non-zero weight
# #         mask = wts > 0
# #         if not mask.any():
# #             # No successors known — return mean of all relation embeddings
# #             return rel_emb[:self.num_rels].mean(dim=0)

# #         ids = ids[mask].to(rel_emb.device)
# #         wts = wts[mask].to(rel_emb.device)
# #         wts = wts / wts.sum()

# #         emb = (wts.unsqueeze(1) * rel_emb[ids]).sum(dim=0)
# #         return emb