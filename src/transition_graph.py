"""
transition_graph.py
===================
Standalone Transition Graph class.

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


        self._succ_idx_cpu = succ_idx
        self._succ_wt_cpu  = succ_wt
        self._inh_idx_cpu  = inh_idx
        self._inh_wt_cpu   = inh_wt

        self.succ_idx = None
        self.succ_wt  = None
        self.inh_idx  = None
        self.inh_wt   = None


        self._weight_diff = None

        n_succ = sum(len(v) for v in successors.values())
        n_inh  = sum(len(v) for v in inhibitors.values())
        print(f"[TransitionGraph] Loaded — "
              f"{n_succ} successor edges, {n_inh} inhibitor edges, "
              f"top-{k} per relation")



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

        self._weight_diff = None
        self._weight_diff_T = None

    def _si(self): return self.succ_idx if self.succ_idx is not None else self._succ_idx_cpu
    def _sw(self): return self.succ_wt  if self.succ_wt  is not None else self._succ_wt_cpu
    def _ii(self): return self.inh_idx  if self.inh_idx  is not None else self._inh_idx_cpu
    def _iw(self): return self.inh_wt   if self.inh_wt   is not None else self._inh_wt_cpu

    def _build_weight_diff(self, device):

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
        self._weight_diff_T = self._weight_diff.t().contiguous()  # (R, R)

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


        test_subjects = torch.unique(s_col)
        keep = torch.isin(s_h, test_subjects)
        s_h, r_h, o_h = s_h[keep], r_h[keep], o_h[keep]

        M = s_h.shape[0]
        if M == 0:
            return log_scores


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
            w = self._weight_diff_T[r_col][:, rh_blk]
            w = w * rhv_blk.unsqueeze(0).float()

            contrib = rerank_alpha * w * mask.float()

            idx = o_blk.unsqueeze(0).expand(N, width)
            delta.scatter_add_(1, idx, contrib)

        return log_scores + delta

