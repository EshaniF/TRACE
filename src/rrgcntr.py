"""
rrgcntr.py — Recurrent RGCN with Transition Graph Extensions
==============================================================
Base model : LogCL (Local-Global History-aware Contrastive Learning)
Extension  : LLM-Augmented Temporal Transition Rules

Two mechanisms replace the original four additions:

  A. Relation embedding regularisation  [training, --use-transition]
     Adds a soft geometric constraint to the relation embedding matrix
     via TransitionGraph.relation_reg_loss().  Successor relation pairs
     are pulled closer in embedding space; inhibitor pairs are pushed
     apart.  Fires on every training step — no sparsity concerns.
     Controlled by --lambda-trans (default 0.1).

  B. Inference-time re-ranking  [evaluation, --use-transition]
     After the base model scores candidate entities, applies a post-hoc
     additive correction in log-space via TransitionGraph.rerank_scores().
     Entities recently involved in inhibitor-relation events with the
     query subject are penalised; entities in successor-relation events
     are boosted.  No gradient, no retraining risk.
     Controlled by --rerank-alpha (default 0.3).

All four original additions (inhibitor hard negatives, transition reg
loss, history reweighting, zero-shot init) have been removed entirely.
The model is otherwise identical to base LogCL.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from rgcn.layers import (UnionRGCNLayer, RGCNBlockLayer, UnionRGCNLayer2,
                          UnionRGATLayer, CompGCNLayer)
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from src.transition_graph import TransitionGraph


# ---------------------------------------------------------------------------
# Sub-modules (unchanged from LogCL)
# ---------------------------------------------------------------------------

class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act    = nn.LeakyReLU(0.2)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        return F.normalize(self.act(self.linear(x)), p=2, dim=1)


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        sc = (False if idx == 0 else True) if self.skip_connect else False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(
                self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                activation=act, self_loop=self.self_loop,
                dropout=self.dropout, skip_connect=sc,
                rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(
                self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                activation=act, self_loop=self.self_loop,
                dropout=self.dropout, skip_connect=sc,
                rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(
                self.h_dim, self.h_dim, self.num_rels, self.opn,
                self.num_bases, activation=act, self_loop=self.self_loop,
                dropout=self.dropout, skip_connect=sc,
                rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        for i, layer in enumerate(self.layers):
            layer(g, [], init_rel_emb[i])
        return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        sc = (False if idx == 0 else True) if self.skip_connect else False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer2(
                self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                activation=act, dropout=self.dropout,
                self_loop=self.self_loop, skip_connect=sc,
                rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        for i, layer in enumerate(self.layers):
            layer(g, [], init_rel_emb[i])
        return g.ndata.pop('h')


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class RecurrentRGCN(nn.Module):
    def __init__(self,
                 decoder_name, encoder_name,
                 num_ents, num_rels,
                 num_static_rels, num_words,
                 h_dim, opn, sequence_len,
                 num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0,
                 self_loop=False, skip_connect=False, layer_norm=False,
                 input_dropout=0, hidden_dropout=0, feat_dropout=0,
                 aggregation='cat', weight=1, pre_weight=0.7,
                 discount=0, angle=0,
                 use_static=False, pre_type='short',
                 use_cl=False, temperature=0.007,
                 entity_prediction=False, relation_prediction=False,
                 use_cuda=False, gpu=0, analysis=False,
                 use_transition=False,
                 transition_top_k=5,
                 lambda_trans=0.1,
                 rerank_alpha=0.3,
                 dataset="",
                 data_root="../data"):
        super().__init__()

        self.decoder_name        = decoder_name
        self.encoder_name        = encoder_name
        self.num_rels            = num_rels
        self.num_ents            = num_ents
        self.opn                 = opn
        self.num_words           = num_words
        self.num_static_rels     = num_static_rels
        self.sequence_len        = sequence_len
        self.h_dim               = h_dim
        self.layer_norm          = layer_norm
        self.h                   = None
        self.run_analysis        = analysis
        self.aggregation         = aggregation
        self.weight              = weight
        self.pre_weight          = pre_weight
        self.discount            = discount
        self.use_static          = use_static
        self.pre_type            = pre_type
        self.use_cl              = use_cl
        self.temp                = temperature
        self.angle               = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction   = entity_prediction
        self.gpu                 = gpu
        self.lambda_trans        = lambda_trans
        self.rerank_alpha        = rerank_alpha
        self.use_transition      = False

        # --- Transition graph ---
        self.trans_graph = None
        if use_transition and dataset:
            _path = os.path.join(data_root, dataset,
                                  "rel_transition_graph.pkl")
            if os.path.exists(_path):
                with open(_path, "rb") as f:
                    _data = pickle.load(f)
                self.trans_graph = TransitionGraph(
                    successors = _data["successors"],
                    inhibitors = _data.get("inhibitors", {}),
                    id2rel     = _data["id2rel"],
                    num_rels   = num_rels,
                    top_k      = transition_top_k)
                self.use_transition = True
                print(f"[TransitionGraph] Loaded from {_path}")
                print(f"[TransitionGraph] "
                      f"Relation reg (lambda={lambda_trans}) + "
                      f"Inference re-ranking (alpha={rerank_alpha}) active.")
            else:
                print(f"[TransitionGraph] Not found at {_path}. "
                      f"Run llm_transition_scorer.py first. "
                      f"Running as base LogCL.")

        # --- embeddings ---
        self.emb_rel     = nn.Parameter(torch.empty(num_rels * 2, h_dim))
        nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = nn.Parameter(torch.empty(num_ents, h_dim))
        nn.init.normal_(self.dynamic_emb)

        # --- linear layers ---
        self.w1   = nn.Linear(h_dim * 2, h_dim)
        self.w2   = nn.Linear(h_dim, h_dim)
        self.w4   = nn.Linear(h_dim * 2, h_dim)
        self.w5   = nn.Linear(h_dim, h_dim)
        self.w_cl = nn.Linear(h_dim * 2, h_dim)

        self.weight_t2      = nn.Parameter(torch.randn(1, h_dim))
        self.bias_t2        = nn.Parameter(torch.randn(1, h_dim))

        self.time_gate_weight = nn.Parameter(torch.empty(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight,
                                gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.zeros(h_dim))

        self.projection_model = MLPLinear(h_dim, h_dim)
        self.entity_cell      = nn.GRUCell(h_dim, h_dim)

        if use_static:
            self.words_emb = nn.Parameter(torch.empty(num_words, h_dim))
            nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(
                h_dim, h_dim, num_static_rels * 2, num_bases,
                activation=F.rrelu, dropout=dropout,
                self_loop=False, skip_connect=False)
            self.static_loss = nn.MSELoss()

        self.loss_e  = nn.CrossEntropyLoss()
        self.loss_r  = nn.CrossEntropyLoss()
        self.loss_cl = nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(
            num_ents, h_dim, h_dim, num_rels * 2,
            num_bases, num_basis, num_hidden_layers, dropout,
            self_loop, skip_connect, encoder_name, opn,
            self.emb_rel, use_cuda, analysis)

        self.his_rgcn_layer = RGCNCell2(
            num_ents, h_dim, h_dim, num_rels * 2,
            num_bases, num_basis, num_hidden_layers, dropout,
            self_loop, skip_connect, encoder_name, opn,
            self.emb_rel, use_cuda, analysis)

        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(
                num_ents, h_dim,
                input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder   = ConvTransR(
                num_rels, h_dim,
                input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    # -----------------------------------------------------------------------
    # One-time device transfer — call after model.cuda() in run_experiment()
    # -----------------------------------------------------------------------

    def move_transition_graph_to_device(self, device):
        if self.use_transition and self.trans_graph is not None:
            self.trans_graph.to(device)

    # -----------------------------------------------------------------------
    # Encoder forward  (pure LogCL — no transition additions)
    # -----------------------------------------------------------------------

    def forward(self, sub_graph, T_idx, query_mask, g_list,
                static_graph, use_cuda):

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat(
                (self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents]
            static_emb = (F.normalize(static_emb)
                          if self.layer_norm else static_emb)
            self.h = static_emb
        else:
            self.h     = (F.normalize(self.dynamic_emb)
                          if self.layer_norm else self.dynamic_emb[:])
            static_emb = None

        self.his_ent, _ = self.all_GCN(self.h, sub_graph, use_cuda)
        his_r_emb       = F.normalize(self.emb_rel)

        his_att = F.softmax(
            self.w5(query_mask + self.his_ent), dim=1)
        his_emb = F.normalize(his_att * self.his_ent)

        history_embs  = []
        att_embs      = []
        his_temp_embs = []
        his_rel_embs  = []

        if self.pre_type == "all":
            for i, g in enumerate(g_list):
                g   = g.to(self.gpu)
                t2  = len(g_list) - i + 1
                h_t = torch.cos(
                    self.weight_t2 * t2 + self.bias_t2
                ).repeat(self.num_ents, 1)
                self.h = self.w4(torch.cat([self.h, h_t], dim=1))

                g.r_to_e  = g.r_to_e.type(torch.LongTensor)
                temp_e    = self.h[g.r_to_e]
                x_input   = (
                    torch.zeros(self.num_rels * 2, self.h_dim).cuda()
                    if use_cuda
                    else torch.zeros(self.num_rels * 2, self.h_dim))

                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x_input[r_idx] = temp_e[span[0]:span[1]].mean(dim=0)
                x_input = self.emb_rel + x_input

                current_h = self.rgcn.forward(
                    g, self.h, [self.emb_rel, self.emb_rel])
                current_h = (F.normalize(current_h)
                              if self.layer_norm else current_h)

                att_e = F.softmax(
                    self.w2(query_mask + current_h), dim=1)

                self.h_0 = (self.entity_cell(current_h, self.h)
                             if i == 0
                             else self.entity_cell(current_h, self.h_0))
                self.h_0 = (F.normalize(self.h_0)
                             if self.layer_norm else self.h_0)

                time_weight = torch.sigmoid(
                    x_input @ self.time_gate_weight + self.time_gate_bias)
                self.hr = (time_weight * x_input
                           + (1 - time_weight) * self.emb_rel)
                self.hr = (F.normalize(self.hr)
                           if self.layer_norm else self.hr)

                history_embs.append(self.h_0)
                his_rel_embs.append(self.hr)
                his_temp_embs.append(self.h_0)
                self.h = self.h_0
                att_embs.append((att_e * self.h_0).unsqueeze(0))

            att_ent     = F.normalize(
                torch.cat(att_embs, dim=0).mean(dim=0))
            history_emb = att_ent + history_embs[-1]
            history_emb = (F.normalize(history_emb)
                           if self.layer_norm else history_emb)
        else:
            self.hr     = None
            history_emb = None

        return (history_emb, static_emb, self.hr, his_emb,
                his_r_emb, his_temp_embs, his_rel_embs)

    # -----------------------------------------------------------------------
    # Inference  — base scoring + transition re-ranking
    # -----------------------------------------------------------------------

    def predict(self, que_pair, sub_graph, T_id, test_graph,
                num_rels, static_graph, test_triplets, use_cuda,
                recent_events=None):
        """
        Parameters
        ----------
        recent_events : list of (s, r, o) int tuples from the most recent
                        history snapshots.  Passed in from test() in
                        mainneg.py.  Used only when use_transition=True.
                        If None, re-ranking is skipped silently.
        """
        with torch.no_grad():
            query_mask, _ = self._build_query(que_pair, use_cuda)

            (embedding, _, r_emb, his_emb,
             _, _, _) = self.forward(
                sub_graph, T_id, query_mask, test_graph,
                static_graph, use_cuda)

            scores_ob, _ = self.decoder_ob.forward(
                embedding, r_emb, test_triplets,
                his_emb, self.pre_weight, self.pre_type)

            scores_en = torch.log(
                F.softmax(scores_ob.clamp(-30, 30), dim=1
                          ).clamp(min=1e-10))

            # --- B. Inference-time re-ranking ----------------------------
            if (self.use_transition
                    and self.trans_graph is not None
                    and recent_events is not None
                    and len(recent_events) > 0):

                # Pass test_triplets directly so rerank_scores derives
                # s_i and r_i per row, keeping loop index i aligned with
                # the N rows of scores_en.  The old approach pre-derived
                # query_r_ids from unique entities (shape B != N).
                scores_en = self.trans_graph.rerank_scores(
                    log_scores    = scores_en,
                    test_triplets = test_triplets,
                    recent_events = recent_events,
                    rerank_alpha  = self.rerank_alpha)

            return test_triplets, scores_en

    # -----------------------------------------------------------------------
    # Training loss  — LogCL + relation embedding regularisation
    # -----------------------------------------------------------------------

    def get_loss(self, que_pair, sub_graph, T_idx, glist, triples,
                 static_graph, use_cuda):

        dev         = self.emb_rel.device
        loss_ent    = torch.zeros(1, device=dev)
        loss_cl     = torch.zeros(1, device=dev)
        loss_rel    = torch.zeros(1, device=dev)
        loss_static = torch.zeros(1, device=dev)

        query_mask, _ = self._build_query(que_pair, use_cuda)

        (embedding, static_emb, r_emb, his_emb,
         his_r_emb, his_temp_embs, his_rel_embs) = self.forward(
            sub_graph, T_idx, query_mask, glist, static_graph, use_cuda)

        # --- entity prediction loss ---
        scores_ob, _ = self.decoder_ob.forward(
            embedding, r_emb, triples, his_emb,
            self.pre_weight, self.pre_type)
        scores_en = torch.log(
            F.softmax(scores_ob.clamp(-30, 30), dim=1).clamp(min=1e-10))
        loss_ent  = F.nll_loss(scores_en, triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(
                embedding, r_emb, triples, mode="train"
            ).view(-1, 2 * self.num_rels)
            loss_rel  = self.loss_r(score_rel, triples[:, 1])

        # --- A. Relation embedding regularisation -----------------------
        # Fires on every training step regardless of which relations appear
        # in the current batch — dense gradient signal.
        if self.use_transition and self.trans_graph is not None:
            loss_reg  = self.trans_graph.relation_reg_loss(self.emb_rel)
            loss_ent  = loss_ent + self.lambda_trans * loss_reg

        # --- contrastive loss (unchanged LogCL) -------------------------
        if (self.use_cl
                and self.pre_type == "all"
                and len(his_temp_embs) > 0):

            cl_sum = torch.zeros(1, device=dev)
            for step, evolve_emb in enumerate(his_temp_embs):
                x1 = self.w_cl(torch.cat(
                    [self.his_ent[triples[:, 0]],
                     his_r_emb[triples[:, 1]]], dim=1))
                x2 = self.w_cl(torch.cat(
                    [evolve_emb[triples[:, 0]],
                     his_rel_embs[step][triples[:, 1]]], dim=1))
                cl_sum += self._logcl(x1, x2, dev)

            loss_cl = cl_sum / len(his_temp_embs)

        return loss_ent, loss_rel, loss_static, loss_cl

    # -----------------------------------------------------------------------
    # Standard LogCL contrastive loss (no transition additions)
    # -----------------------------------------------------------------------

    def _logcl(self, z1_raw, z2_raw, device):
        z1 = self.projection_model(z1_raw)
        z2 = self.projection_model(z2_raw)
        B  = z1.size(0)
        labels = torch.arange(B, device=device)

        def _ce(a, b):
            return self.loss_cl(torch.mm(a, b.T) / self.temp, labels)

        return (_ce(z1, z2) + _ce(z2, z1)
                + _ce(z1, z1) + _ce(z2, z2)) / 4

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def all_GCN(self, ent_emb, sub_graph, use_cuda):
        sub_graph = sub_graph.to(self.gpu)
        sub_graph.ndata['h'] = ent_emb
        his_emb    = self.his_rgcn_layer.forward(
            sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
        subg_index = torch.masked_select(
            torch.arange(sub_graph.number_of_nodes(),
                         dtype=torch.long, device=sub_graph.device),
            sub_graph.in_degrees(
                range(sub_graph.number_of_nodes())) > 0)
        return F.normalize(his_emb), subg_index

    def _build_query(self, que_pair, use_cuda):
        uniq_e, r_len, r_idx = que_pair
        temp_r  = self.emb_rel[r_idx]
        e_input = (torch.zeros(self.num_ents, self.h_dim).cuda()
                   if use_cuda
                   else torch.zeros(self.num_ents, self.h_dim))
        for span, e_idx in zip(r_len, uniq_e):
            e_input[e_idx] = temp_r[span[0]:span[1]].mean(dim=0)

        q_t    = torch.cos(self.bias_t2).repeat(self.num_ents, 1)
        qe_emb = self.w4(torch.cat([self.dynamic_emb, q_t], dim=1))
        q_emb  = self.w1(torch.cat(
            [qe_emb[uniq_e], e_input[uniq_e]], dim=1))

        query_mask = (torch.zeros(self.num_ents, self.h_dim).to(self.gpu)
                      if use_cuda else torch.zeros(1))
        query_mask[uniq_e] = q_emb
        return query_mask, e_input

# """
# rrgcnneg.py — Recurrent RGCN with Temporal Transition-Aware Contrastive Learning
# ==================================================================================
# Base model : LogCL (Local-Global History-aware Contrastive Learning)
# Extension  : LLM-Augmented Temporal Transition Rules

# Four additions over base LogCL (zero LLM at training time):

#   1. Transition-aware hard negatives  [--use-add1]
#   2. Transition regularisation loss   [--use-add2]
#   3. Transition-conditioned history reweighting  [--use-add3]
#   4. Zero-shot unseen relation initialisation (evaluation only)  [--use-add4]

# Fixes applied in this version
# ------------------------------
#   FIX-1  r_ids % self.num_ents bug — subject embeddings now use s_ids.
#   FIX-2  Addition 3 was never applied in forward() — now active.
#   FIX-3  Zero-padding sentinel -1 for inhibitor/successor index tensors.
#   FIX-4  Asymmetric extra_scores — two anchor-matched tensors now used.

#   FIX-BUG1  entity_weight scatter used wrong indices (0..B-1 instead of
#              the actual subject entity IDs from uniq_e).

#   FIX-BUG2  predict() never passed query_r_ids to forward(), silently
#              disabling Addition 3 at inference time.

#   FIX-RISK1  trans_graph.to(device) moved to one-time call via
#              move_transition_graph_to_device().

#   FIX-RISK3  Assertion added: Additions 1 and 3 require pre_type='all'.

#   FIX-ADD1-HOLLOW  inh_proj is semantically empty at init — gated with
#              learnable warmup scalar add1_scale (init -3 → softplus ≈ 0.05).

#   FIX-ADD2-SPARSITY  same_subj constraint removed from reg loss; signal
#              increased ~45× on ICEWS14.  lambda_trans default raised to 1.0.

#   FIX-ADD3-GRADIENT  snap_weight mean() replaced with any()-based gate so
#              gradient is O(1) not O(succ_frac/T) ≈ O(0.00014).

#   FLAW-2 FIX  Addition 3 reweighting moved from entity level (inside each
#              snapshot step) to snapshot level (weighted average of att_embs).
#              The transition graph encodes relation-level temporal knowledge;
#              the correct abstraction is to weight the T_hist snapshot windows
#              themselves, not individual entities within a step.

#   FLAW-3 FIX  Addition 2 reg loss extended with a symmetric successor
#              pull-up term.  Previously only inhibitor objects were pushed
#              down; now successor objects are also pulled up.  The two hinge
#              terms use the same margin so their contributions are balanced.
# """

# import os
# import pickle
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from rgcn.layers import (UnionRGCNLayer, RGCNBlockLayer, UnionRGCNLayer2,
#                           UnionRGATLayer, CompGCNLayer)
# from src.model import BaseRGCN
# from src.decoder import ConvTransE, ConvTransR
# from src.transition_graph import TransitionGraph


# # ---------------------------------------------------------------------------
# # Sub-modules (unchanged from LogCL)
# # ---------------------------------------------------------------------------

# class MLPLinear(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.linear = nn.Linear(in_dim, out_dim)
#         self.act    = nn.LeakyReLU(0.2)
#         nn.init.xavier_normal_(self.linear.weight)

#     def forward(self, x):
#         return F.normalize(self.act(self.linear(x)), p=2, dim=1)


# class RGCNCell(BaseRGCN):
#     def build_hidden_layer(self, idx):
#         act = F.rrelu
#         if idx:
#             self.num_basis = 0
#         sc = (False if idx == 0 else True) if self.skip_connect else False
#         if self.encoder_name == "uvrgcn":
#             return UnionRGCNLayer(
#                 self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                 activation=act, self_loop=self.self_loop,
#                 dropout=self.dropout, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         elif self.encoder_name == "kbat":
#             return UnionRGATLayer(
#                 self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                 activation=act, self_loop=self.self_loop,
#                 dropout=self.dropout, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         elif self.encoder_name == "compgcn":
#             return CompGCNLayer(
#                 self.h_dim, self.h_dim, self.num_rels, self.opn,
#                 self.num_bases, activation=act, self_loop=self.self_loop,
#                 dropout=self.dropout, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         else:
#             raise NotImplementedError

#     def forward(self, g, init_ent_emb, init_rel_emb):
#         node_id = g.ndata['id'].squeeze()
#         g.ndata['h'] = init_ent_emb[node_id]
#         for i, layer in enumerate(self.layers):
#             layer(g, [], init_rel_emb[i])
#         return g.ndata.pop('h')


# class RGCNCell2(BaseRGCN):
#     def build_hidden_layer(self, idx):
#         act = F.rrelu
#         if idx:
#             self.num_basis = 0
#         sc = (False if idx == 0 else True) if self.skip_connect else False
#         if self.encoder_name == "uvrgcn":
#             return UnionRGCNLayer2(
#                 self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                 activation=act, dropout=self.dropout,
#                 self_loop=self.self_loop, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         else:
#             raise NotImplementedError

#     def forward(self, g, init_ent_emb, init_rel_emb):
#         node_id = g.ndata['id'].squeeze()
#         g.ndata['h'] = init_ent_emb[node_id]
#         for i, layer in enumerate(self.layers):
#             layer(g, [], init_rel_emb[i])
#         return g.ndata.pop('h')


# # ---------------------------------------------------------------------------
# # Main model
# # ---------------------------------------------------------------------------

# class RecurrentRGCN(nn.Module):
#     def __init__(self,
#                  decoder_name, encoder_name,
#                  num_ents, num_rels,
#                  num_static_rels, num_words,
#                  h_dim, opn, sequence_len,
#                  num_bases=-1, num_basis=-1,
#                  num_hidden_layers=1, dropout=0,
#                  self_loop=False, skip_connect=False, layer_norm=False,
#                  input_dropout=0, hidden_dropout=0, feat_dropout=0,
#                  aggregation='cat', weight=1, pre_weight=0.7,
#                  discount=0, angle=0,
#                  use_static=False, pre_type='short',
#                  use_cl=False, temperature=0.007,
#                  entity_prediction=False, relation_prediction=False,
#                  use_cuda=False, gpu=0, analysis=False,
#                  use_transition=False,
#                  transition_top_k=5,
#                  lambda_trans=0.1,
#                  dataset="",
#                  data_root="../data",
#                  use_add1=True,
#                  use_add2=True,
#                  use_add3=True,
#                  use_add4=True,
#                  ):
#         super().__init__()

#         # --- config ---
#         self.decoder_name        = decoder_name
#         self.encoder_name        = encoder_name
#         self.num_rels            = num_rels
#         self.num_ents            = num_ents
#         self.opn                 = opn
#         self.num_words           = num_words
#         self.num_static_rels     = num_static_rels
#         self.sequence_len        = sequence_len
#         self.h_dim               = h_dim
#         self.layer_norm          = layer_norm
#         self.h                   = None
#         self.run_analysis        = analysis
#         self.aggregation         = aggregation
#         self.weight              = weight
#         self.pre_weight          = pre_weight
#         self.discount            = discount
#         self.use_static          = use_static
#         self.pre_type            = pre_type
#         self.use_cl              = use_cl
#         self.temp                = temperature
#         self.angle               = angle
#         self.relation_prediction = relation_prediction
#         self.entity_prediction   = entity_prediction
#         self.gpu                 = gpu
#         self.lambda_trans        = lambda_trans

#         self.use_add1 = use_add1
#         self.use_add2 = use_add2
#         self.use_add3 = use_add3
#         self.use_add4 = use_add4

#         # FIX-RISK3: Additions 1 and 3 only fire inside the pre_type=="all"
#         # branch.  Catch a misconfiguration early rather than silently
#         # producing a no-op.
#         if use_transition and (use_add1 or use_add3) and pre_type != "all":
#             raise ValueError(
#                 f"Additions 1 and 3 require pre_type='all' "
#                 f"(got pre_type='{pre_type}'). "
#                 f"Pass --pre-type all or disable --use-add1 / --use-add3.")

#         # --- Transition graph ---
#         self.trans_graph    = None
#         self.use_transition = False

#         if use_transition and dataset:
#             _path = os.path.join(data_root, dataset,
#                                   "rel_transition_graph.pkl")
#             if os.path.exists(_path):
#                 with open(_path, "rb") as f:
#                     _data = pickle.load(f)
#                 self.trans_graph = TransitionGraph(
#                     successors = _data["successors"],
#                     inhibitors = _data.get("inhibitors", {}),
#                     id2rel     = _data["id2rel"],
#                     num_rels   = num_rels,
#                     top_k      = transition_top_k)
#                 self.use_transition = True
#                 print(f"[TransitionGraph] Loaded from {_path}")
#                 active = self.active_additions()
#                 print(f"[TransitionGraph] Active additions: "
#                       f"{active if active else 'none (ablation: all off)'}")
#             else:
#                 print(f"[TransitionGraph] Not found at {_path}. "
#                       f"Run llm_transition_scorer.py first. "
#                       f"Running as base LogCL.")

#         # --- embeddings ---
#         self.emb_rel     = nn.Parameter(torch.empty(num_rels * 2, h_dim))
#         nn.init.xavier_normal_(self.emb_rel)

#         self.dynamic_emb = nn.Parameter(torch.empty(num_ents, h_dim))
#         nn.init.normal_(self.dynamic_emb)

#         # --- linear layers ---
#         self.w1   = nn.Linear(h_dim * 2, h_dim)
#         self.w2   = nn.Linear(h_dim, h_dim)
#         self.w4   = nn.Linear(h_dim * 2, h_dim)
#         self.w5   = nn.Linear(h_dim, h_dim)
#         self.w_cl = nn.Linear(h_dim * 2, h_dim)

#         self.weight_t2      = nn.Parameter(torch.randn(1, h_dim))
#         self.bias_t2        = nn.Parameter(torch.randn(1, h_dim))

#         self.time_gate_weight = nn.Parameter(torch.empty(h_dim, h_dim))
#         nn.init.xavier_uniform_(self.time_gate_weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         self.time_gate_bias = nn.Parameter(torch.zeros(h_dim))

#         self.projection_model = MLPLinear(h_dim, h_dim)
#         self.entity_cell      = nn.GRUCell(h_dim, h_dim)

#         # Addition 3: relation-specific learned history reweighting gates.
#         self.reweight_succ = nn.Parameter(torch.zeros(num_rels))
#         self.reweight_inh  = nn.Parameter(torch.zeros(num_rels))

#         # Addition 1: learnable warmup scalar for inhibitor hard negatives.
#         # Initialised to -3 so softplus(-3) ≈ 0.049 — near-zero at the start.
#         # The model gradually turns on the inhibitor columns as it learns which
#         # inhibitor projections are informative. Without this gate, randomly
#         # initialised inh_proj vectors immediately inflate the CL loss and
#         # destabilise early training before the encoder has converged.
#         self.add1_scale = nn.Parameter(torch.tensor(-1.0))

#         if use_static:
#             self.words_emb = nn.Parameter(torch.empty(num_words, h_dim))
#             nn.init.xavier_normal_(self.words_emb)
#             self.statci_rgcn_layer = RGCNBlockLayer(
#                 h_dim, h_dim, num_static_rels * 2, num_bases,
#                 activation=F.rrelu, dropout=dropout,
#                 self_loop=False, skip_connect=False)
#             self.static_loss = nn.MSELoss()

#         self.loss_e  = nn.CrossEntropyLoss()
#         self.loss_r  = nn.CrossEntropyLoss()
#         self.loss_cl = nn.CrossEntropyLoss()

#         self.rgcn = RGCNCell(
#             num_ents, h_dim, h_dim, num_rels * 2,
#             num_bases, num_basis, num_hidden_layers, dropout,
#             self_loop, skip_connect, encoder_name, opn,
#             self.emb_rel, use_cuda, analysis)

#         self.his_rgcn_layer = RGCNCell2(
#             num_ents, h_dim, h_dim, num_rels * 2,
#             num_bases, num_basis, num_hidden_layers, dropout,
#             self_loop, skip_connect, encoder_name, opn,
#             self.emb_rel, use_cuda, analysis)

#         if decoder_name == "convtranse":
#             self.decoder_ob = ConvTransE(
#                 num_ents, h_dim,
#                 input_dropout, hidden_dropout, feat_dropout)
#             self.rdecoder   = ConvTransR(
#                 num_rels, h_dim,
#                 input_dropout, hidden_dropout, feat_dropout)
#         else:
#             raise NotImplementedError

#     # -----------------------------------------------------------------------
#     # FIX-RISK1: one-time device transfer for the transition graph.
#     # Call this once after model.cuda() in run_experiment() instead of
#     # calling trans_graph.to(dev) inside every get_loss() invocation.
#     # -----------------------------------------------------------------------

#     def move_transition_graph_to_device(self, device):
#         """
#         Move the transition graph tensors to `device` exactly once.
#         Called from run_experiment() immediately after model.cuda().
#         Safe to call when use_transition=False or trans_graph is None.
#         """
#         if self.use_transition and self.trans_graph is not None:
#             self.trans_graph.to(device)

#     # -----------------------------------------------------------------------
#     # Ablation query helper
#     # -----------------------------------------------------------------------

#     def active_additions(self) -> list:
#         if not self.use_transition:
#             return []
#         return [i for i, flag in enumerate(
#             [self.use_add1, self.use_add2, self.use_add3, self.use_add4],
#             start=1) if flag]

#     # -----------------------------------------------------------------------
#     # Encoder forward
#     #
#     # FIX-BUG1: accepts uniq_e so that Addition 3 can scatter snap_weight
#     # into the correct entity slots rather than into positions 0..B-1.
#     #
#     # query_r_ids : (B,) LongTensor — relation IDs for reweighting (Add. 3)
#     # uniq_e      : (B,) LongTensor — subject entity IDs from e2r, parallel
#     #               to query_r_ids.  Required when do_reweight is True.
#     # -----------------------------------------------------------------------

#     def forward(self, sub_graph, T_idx, query_mask, g_list,
#                 static_graph, use_cuda,
#                 query_r_ids=None, uniq_e=None):
#         """
#         query_r_ids : (B,) LongTensor — relation IDs for the current batch.
#         uniq_e      : (B,) LongTensor — unique subject entity IDs (from e2r).
#                       Must be provided together with query_r_ids for Add. 3.
#         """
#         if self.use_static:
#             static_graph = static_graph.to(self.gpu)
#             static_graph.ndata['h'] = torch.cat(
#                 (self.dynamic_emb, self.words_emb), dim=0)
#             self.statci_rgcn_layer(static_graph, [])
#             static_emb = static_graph.ndata.pop('h')[:self.num_ents]
#             static_emb = (F.normalize(static_emb)
#                           if self.layer_norm else static_emb)
#             self.h = static_emb
#         else:
#             self.h     = (F.normalize(self.dynamic_emb)
#                           if self.layer_norm else self.dynamic_emb[:])
#             static_emb = None

#         self.his_ent, _ = self.all_GCN(self.h, sub_graph, use_cuda)
#         his_r_emb       = F.normalize(self.emb_rel)

#         his_att = F.softmax(
#             self.w5(query_mask + self.his_ent), dim=1)
#         his_emb = F.normalize(his_att * self.his_ent)

#         history_embs  = []
#         att_embs      = []
#         his_temp_embs = []
#         his_rel_embs  = []

#         # Addition 3 gate: uniq_e is no longer needed since reweighting now
#         # operates at snapshot level (not entity level), but query_r_ids is
#         # still required to compute per-query successor/inhibitor masks.
#         do_reweight = (
#             self.use_transition
#             and self.use_add3
#             and self.trans_graph is not None
#             and query_r_ids is not None
#         )

#         if self.pre_type == "all":
#             # Addition 3 (FLAW-2 FIX): collect one scalar weight per snapshot
#             # step, applied when averaging att_embs at the end of the loop.
#             #
#             # Previous design applied the transition signal at entity level
#             # inside each step — scaling individual entity attention rows by
#             # entity_weight[e].  That is a category mismatch: the transition
#             # graph encodes relation-level temporal knowledge ("snapshot X
#             # contains a successor relation → snapshot X is more informative"),
#             # not entity-level knowledge.  Weighting entities by which
#             # relations appear in the same snapshot has no principled link
#             # between the two.
#             #
#             # Correct design: weight the SNAPSHOT STEPS themselves before
#             # taking the average over the T_hist history windows.  Each step
#             # gets a single scalar that reflects how transition-relevant that
#             # window is for the current batch of queries, then the final
#             # att_ent is a weighted sum over steps instead of a uniform mean.
#             #
#             # snap_step_weights[i] is a scalar tensor computed from the
#             # any()-based gate (not mean() — avoids the 1/T dilution).
#             # It stays in the computation graph so gradients flow back to
#             # reweight_succ and reweight_inh.
#             snap_step_weights = []

#             for i, g in enumerate(g_list):
#                 g   = g.to(self.gpu)
#                 t2  = len(g_list) - i + 1
#                 h_t = torch.cos(
#                     self.weight_t2 * t2 + self.bias_t2
#                 ).repeat(self.num_ents, 1)
#                 self.h = self.w4(torch.cat([self.h, h_t], dim=1))

#                 g.r_to_e  = g.r_to_e.type(torch.LongTensor)
#                 temp_e    = self.h[g.r_to_e]
#                 x_input   = (
#                     torch.zeros(self.num_rels * 2, self.h_dim).cuda()
#                     if use_cuda
#                     else torch.zeros(self.num_rels * 2, self.h_dim))

#                 for span, r_idx in zip(g.r_len, g.uniq_r):
#                     x_input[r_idx] = temp_e[span[0]:span[1]].mean(dim=0)
#                 x_input = self.emb_rel + x_input

#                 current_h = self.rgcn.forward(
#                     g, self.h, [self.emb_rel, self.emb_rel])
#                 current_h = (F.normalize(current_h)
#                               if self.layer_norm else current_h)

#                 # att_e is computed as usual — no entity-level reweighting.
#                 att_e = F.softmax(
#                     self.w2(query_mask + current_h), dim=1)

#                 # Addition 3: compute a per-step scalar weight for this
#                 # snapshot.  The scalar represents "how transition-relevant
#                 # is this history window for the current query batch?"
#                 #
#                 # any() over the T relation dimension gives a (B,) bool:
#                 # True if ANY relation in this snapshot is a successor /
#                 # inhibitor of the query relation.  The learned scalars
#                 # succ_boost / inh_suppress are then applied directly —
#                 # gradient magnitude is O(1) not O(succ_frac / T).
#                 #
#                 # mean over B gives one scalar per step; this is the weight
#                 # used in the final weighted average of att_embs below.
#                 if do_reweight and hasattr(g, 'uniq_r') and len(g.uniq_r) > 0:
#                     snap_r_ids = torch.tensor(
#                         g.uniq_r, dtype=torch.long,
#                         device=query_r_ids.device)

#                     is_succ, is_inh = self.trans_graph.history_reweight_masks(
#                         query_r_ids, snap_r_ids)   # (B, T)

#                     r_base       = query_r_ids % self.num_rels          # (B,)
#                     succ_boost   = 1.0 + F.softplus(
#                         self.reweight_succ[r_base])                     # (B,)
#                     inh_suppress = torch.sigmoid(
#                         self.reweight_inh[r_base])                      # (B,)

#                     has_succ = is_succ.any(dim=1)   # (B,) bool
#                     has_inh  = is_inh.any(dim=1)    # (B,) bool

#                     # Per-query scalar: successor wins over inhibitor where both.
#                     per_query_w = torch.ones(
#                         query_r_ids.size(0), device=query_r_ids.device)  # (B,)
#                     per_query_w = torch.where(has_inh,  inh_suppress, per_query_w)
#                     per_query_w = torch.where(has_succ, succ_boost,   per_query_w)

#                     # Reduce to a single step scalar (mean over B queries).
#                     # This stays in the graph — gradient flows to reweight params.
#                     step_w = per_query_w.mean()                          # scalar
#                 else:
#                     step_w = torch.ones(1, device=self.emb_rel.device)

#                 snap_step_weights.append(step_w)

#                 self.h_0 = (self.entity_cell(current_h, self.h)
#                              if i == 0
#                              else self.entity_cell(current_h, self.h_0))
#                 self.h_0 = (F.normalize(self.h_0)
#                              if self.layer_norm else self.h_0)

#                 time_weight = torch.sigmoid(
#                     x_input @ self.time_gate_weight + self.time_gate_bias)
#                 self.hr = (time_weight * x_input
#                            + (1 - time_weight) * self.emb_rel)
#                 self.hr = (F.normalize(self.hr)
#                            if self.layer_norm else self.hr)

#                 history_embs.append(self.h_0)
#                 his_rel_embs.append(self.hr)
#                 his_temp_embs.append(self.h_0)
#                 self.h = self.h_0
#                 att_embs.append((att_e * self.h_0).unsqueeze(0))

#             # Weighted average over snapshot steps.
#             # snap_step_weights[i] is a scalar in the computation graph.
#             # Stack to (T_hist,), normalise, then apply as a weighted sum.
#             # When do_reweight is False every weight is 1.0 → identical to
#             # the original uniform mean (no regression for base LogCL).
#             step_w_tensor = torch.stack(snap_step_weights)          # (T_hist,)
#             step_w_norm   = step_w_tensor / step_w_tensor.sum().clamp(min=1e-9)
#             # att_embs is list of (1, num_ents, h_dim); cat → (T_hist, …)
#             att_stack = torch.cat(att_embs, dim=0)                  # (T_hist, E, h)
#             att_ent   = F.normalize(
#                 (att_stack * step_w_norm.view(-1, 1, 1)).sum(dim=0))  # (E, h)

#             history_emb = att_ent + history_embs[-1]
#             history_emb = (F.normalize(history_emb)
#                            if self.layer_norm else history_emb)
#         else:
#             self.hr     = None
#             history_emb = None

#         return (history_emb, static_emb, self.hr, his_emb,
#                 his_r_emb, his_temp_embs, his_rel_embs)

#     # -----------------------------------------------------------------------
#     # Inference
#     #
#     # FIX-BUG2: derive query_r_ids and uniq_e from test_triplets and the
#     # que_pair returned by e2r, then pass both to forward() so that
#     # Addition 3 is active during evaluation just as it is during training.
#     # -----------------------------------------------------------------------

#     def predict(self, que_pair, sub_graph, T_id, test_graph,
#                 num_rels, static_graph, test_triplets, use_cuda):
#         with torch.no_grad():
#             query_mask, _ = self._build_query(que_pair, use_cuda)

#             # FIX-BUG2: build query_r_ids and uniq_e for Addition 3.
#             # que_pair = [uniq_e_tensor, r_len, r_idx] from e2r().
#             # test_triplets[:, 1] gives the relation for each triple, but
#             # forward() expects one entry per unique subject entity (B,).
#             # uniq_e (que_pair[0]) is exactly that set — use it to index
#             # query_r_ids so the lengths stay aligned.
#             if self.use_transition and self.trans_graph is not None:
#                 uniq_e_t    = que_pair[0]                    # (B,) entity IDs
#                 subj_col  = test_triplets[:, 0]              # (N,)
#                 rel_col   = test_triplets[:, 1]              # (N,)
#                 # Build subject→relation lookup; first occurrence wins.
#                 # Every entity in uniq_e_t is guaranteed to appear in subj_col
#                 # (both come from the same snapshot), so the .get() fallback of
#                 # 0 is a safety guard only and should never trigger.
#                 subj_to_rel = {}
#                 for s, r in zip(subj_col.tolist(), rel_col.tolist()):
#                     if s not in subj_to_rel:
#                         subj_to_rel[s] = r
#                 query_r_ids = torch.tensor(
#                     [subj_to_rel.get(int(e), 0) for e in uniq_e_t.tolist()],
#                     dtype=torch.long, device=uniq_e_t.device)  # (B,)
#                 uniq_e_out  = uniq_e_t
#             else:
#                 query_r_ids = None
#                 uniq_e_out  = None

#             (embedding, _, r_emb, his_emb,
#              _, _, _) = self.forward(
#                 sub_graph, T_id, query_mask, test_graph,
#                 static_graph, use_cuda,
#                 query_r_ids=query_r_ids,
#                 uniq_e=uniq_e_out)

#             scores_ob, _ = self.decoder_ob.forward(
#                 embedding, r_emb, test_triplets,
#                 his_emb, self.pre_weight, self.pre_type)
#             scores_en = torch.log(
#                 F.softmax(scores_ob.clamp(-30, 30), dim=1
#                           ).clamp(min=1e-10))
#             return test_triplets, scores_en

#     # -----------------------------------------------------------------------
#     # Zero-shot unseen relation initialisation (called at test time)
#     # -----------------------------------------------------------------------

#     def init_unseen_relations(self, unseen_rel_ids: list):
#         if not self.use_transition or not self.use_add4 or self.trans_graph is None:
#             print("[ZeroShot] Skipped — transition graph absent or "
#                   "Addition 4 disabled (--no-add4).")
#             return

#         n_init = 0
#         with torch.no_grad():
#             for r_id in unseen_rel_ids:
#                 if r_id >= self.num_rels:
#                     continue
#                 new_emb = self.trans_graph.init_unseen_relation(
#                     r_id, self.emb_rel)
#                 self.emb_rel.data[r_id]                  = new_emb
#                 self.emb_rel.data[r_id + self.num_rels]  = new_emb
#                 n_init += 1

#         print(f"[ZeroShot] Initialised {n_init}/{len(unseen_rel_ids)} "
#               f"unseen relation embeddings from transition graph.")

#     # -----------------------------------------------------------------------
#     # Training loss
#     # -----------------------------------------------------------------------

#     def get_loss(self, que_pair, sub_graph, T_idx, glist, triples,
#                  static_graph, use_cuda, epoch=0):
#         """
#         Returns (loss_ent, loss_rel, loss_static, loss_cl).

#         FIX-RISK1: trans_graph.to(dev) is NO LONGER called here.
#         It must be called once via move_transition_graph_to_device()
#         after model.cuda() in run_experiment().
#         """
#         dev         = self.emb_rel.device
#         loss_ent    = torch.zeros(1, device=dev)
#         loss_cl     = torch.zeros(1, device=dev)
#         loss_rel    = torch.zeros(1, device=dev)
#         loss_static = torch.zeros(1, device=dev)

#         query_mask, _ = self._build_query(que_pair, use_cuda)

#         # Derive query_r_ids and uniq_e for forward() (Add. 2 & 3).
#         # que_pair[0] is uniq_e from e2r — parallel to triples[:, 1]
#         # bucketed by unique subject, so we build the same subject→rel
#         # mapping used in predict().
#         if self.use_transition and self.trans_graph is not None:
#             uniq_e_t  = que_pair[0]                          # (B,)
#             subj_col  = triples[:, 0]
#             rel_col   = triples[:, 1]
#             subj_to_rel = {}
#             for s, r in zip(subj_col.tolist(), rel_col.tolist()):
#                 if s not in subj_to_rel:
#                     subj_to_rel[s] = r
#             query_r_ids = torch.tensor(
#                 [subj_to_rel.get(int(e), 0) for e in uniq_e_t.tolist()],
#                 dtype=torch.long, device=dev)                 # (B,)
#             uniq_e_out  = uniq_e_t
#         else:
#             query_r_ids = None
#             uniq_e_out  = None

#         (embedding, static_emb, r_emb, his_emb,
#          his_r_emb, his_temp_embs, his_rel_embs) = self.forward(
#             sub_graph, T_idx, query_mask, glist, static_graph, use_cuda,
#             query_r_ids=query_r_ids,
#             uniq_e=uniq_e_out)

#         # --- entity prediction loss ---
#         scores_ob, _ = self.decoder_ob.forward(
#             embedding, r_emb, triples, his_emb,
#             self.pre_weight, self.pre_type)
#         scores_en = torch.log(
#             F.softmax(scores_ob.clamp(-30, 30), dim=1).clamp(min=1e-10))
#         loss_ent  = F.nll_loss(scores_en, triples[:, 2])

#         if self.relation_prediction:
#             score_rel = self.rdecoder.forward(
#                 embedding, r_emb, triples, mode="train"
#             ).view(-1, 2 * self.num_rels)
#             loss_rel  = self.loss_r(score_rel, triples[:, 1])

#         # --- Addition 2: transition regularisation loss ---
#         if self.use_transition and self.use_add2 and self.trans_graph is not None:
#             loss_trans = self._transition_reg_loss(
#                 scores_ob, triples, dev)
#             loss_ent   = loss_ent + self.lambda_trans * loss_trans

#         # --- contrastive loss ---
#         if (self.use_cl
#                 and self.pre_type == "all"
#                 and len(his_temp_embs) > 0):

#             cl_sum = torch.zeros(1, device=dev)
#             for step, evolve_emb in enumerate(his_temp_embs):
#                 x1 = self.w_cl(torch.cat(
#                     [self.his_ent[triples[:, 0]],
#                      his_r_emb[triples[:, 1]]], dim=1))
#                 x2 = self.w_cl(torch.cat(
#                     [evolve_emb[triples[:, 0]],
#                      his_rel_embs[step][triples[:, 1]]], dim=1))

#                 cl_sum += self._logcl_with_transitions(
#                     x1, x2,
#                     r_ids    = triples[:, 1],
#                     s_ids    = triples[:, 0],
#                     device   = dev,
#                     use_add1 = self.use_add1)

#             loss_cl = cl_sum / len(his_temp_embs)

#         return loss_ent, loss_rel, loss_static, loss_cl

#     # -----------------------------------------------------------------------
#     # Addition 2 — transition regularisation loss (symmetric)
#     #
#     # FLAW-3 FIX: original loss only pushed inhibitor objects down.  The
#     # transition graph encodes a directed preference ordering — successors
#     # should score HIGH, inhibitors should score LOW.  Only implementing
#     # the inhibitor side trains on half the available signal.
#     #
#     # This version adds a symmetric successor pull-up term alongside the
#     # existing inhibitor push-down term.  Both use the same hinge margin
#     # so their contributions are naturally balanced.  A separate
#     # lambda_succ hyperparameter is not needed; lambda_trans scales both.
#     #
#     # Inhibitor push-down (existing):
#     #   For query (s_i, r_i, o_i): if triple j has relation r_j that is an
#     #   inhibitor of r_i, penalise when score(o_j | context_i) is too high.
#     #   hinge_inh = relu(score_inh_obj - score_true + margin)
#     #
#     # Successor pull-up (new):
#     #   For query (s_i, r_i, o_i): if triple j has relation r_j that is a
#     #   successor of r_i, penalise when score(o_j | context_i) is too low.
#     #   hinge_succ = relu(score_true - score_succ_obj + margin)
#     # -----------------------------------------------------------------------

#     def _transition_reg_loss(self,
#                               scores_ob: torch.Tensor,
#                               triples: torch.Tensor,
#                               device: torch.device) -> torch.Tensor:
#         if self.trans_graph is None:
#             return torch.zeros(1, device=device)

#         r_ids = triples[:, 1]
#         o_ids = triples[:, 2]
#         B     = r_ids.size(0)

#         r_base      = r_ids % self.num_rels
#         true_scores = scores_ob[torch.arange(B, device=device), o_ids]  # (B,)

#         # Shared building block: (B, B) matrix of relation IDs for all pairs.
#         # r_exp[i, j, :] = r_base[j] — the relation of triple j, tested
#         # against the successor/inhibitor lists of triple i's relation.
#         r_ids_exp = (r_ids.unsqueeze(0).expand(B, B) % self.num_rels
#                      ).unsqueeze(2)                                      # (B, B, 1)

#         # Object score matrix: [i, j] = score of triple j's object given
#         # the context/history representation of query i.
#         obj_scores_mat   = scores_ob[:, o_ids]                          # (B, B)
#         true_sc_expanded = true_scores.unsqueeze(1)                     # (B, 1)
#         margin           = 0.5

#         # ── inhibitor push-down ──────────────────────────────────────────
#         inh_rels  = self.trans_graph.get_inhibitor_relations(r_base)    # (B, K)
#         K_i       = inh_rels.size(1)
#         inh_exp   = inh_rels.unsqueeze(1).expand(B, B, K_i)
#         inh_valid = (inh_rels != -1).unsqueeze(1).expand(B, B, K_i)
#         is_inh    = ((r_ids_exp == inh_exp) & inh_valid).any(dim=2)    # (B, B)
#         inh_mask  = is_inh.clone()
#         inh_mask.fill_diagonal_(False)

#         loss_inh = torch.zeros(1, device=device)
#         if inh_mask.any():
#             hinge_inh = F.relu(obj_scores_mat - true_sc_expanded + margin)
#             hinge_inh = hinge_inh * inh_mask.float()
#             loss_inh  = hinge_inh.sum() / inh_mask.float().sum().clamp(min=1)

#         # ── successor pull-up (FLAW-3 FIX) ──────────────────────────────
#         succ_rels  = self.trans_graph.get_successor_relations(r_base)   # (B, K)
#         K_s        = succ_rels.size(1)
#         succ_exp   = succ_rels.unsqueeze(1).expand(B, B, K_s)
#         succ_valid = (succ_rels != -1).unsqueeze(1).expand(B, B, K_s)
#         is_succ    = ((r_ids_exp == succ_exp) & succ_valid).any(dim=2)  # (B, B)
#         succ_mask  = is_succ.clone()
#         succ_mask.fill_diagonal_(False)

#         loss_succ = torch.zeros(1, device=device)
#         if succ_mask.any():
#             # Penalise when successor object scores BELOW true object score.
#             # relu(true - succ + margin) fires when succ < true - margin.
#             hinge_succ = F.relu(true_sc_expanded - obj_scores_mat + margin)
#             hinge_succ = hinge_succ * succ_mask.float()
#             loss_succ  = hinge_succ.sum() / succ_mask.float().sum().clamp(min=1)

#         return loss_inh + loss_succ

#     # -----------------------------------------------------------------------
#     # Contrastive loss — original LogCL 4-term + inhibitor hard negatives
#     # -----------------------------------------------------------------------

#     def _logcl_with_transitions(self,
#                                   z1_raw: torch.Tensor,
#                                   z2_raw: torch.Tensor,
#                                   r_ids: torch.Tensor,
#                                   s_ids: torch.Tensor,
#                                   device: torch.device,
#                                   use_add1: bool = True) -> torch.Tensor:
#         loss_fn = self.loss_cl

#         z1 = self.projection_model(z1_raw)
#         z2 = self.projection_model(z2_raw)
#         B  = z1.size(0)
#         D  = z1.size(1)

#         labels = torch.arange(B, device=device)

#         extra_scores_z1 = None
#         extra_scores_z2 = None

#         if self.use_transition and use_add1 and self.trans_graph is not None:
#             r_base   = r_ids % self.num_rels
#             inh_rels = self.trans_graph.get_inhibitor_relations(r_base)
#             K        = inh_rels.size(1)

#             inh_valid        = (inh_rels != -1)
#             inh_rels_clamped = inh_rels.clamp(min=0)

#             inh_rel_embs = self.emb_rel[inh_rels_clamped.view(-1)
#                            ].view(B, K, -1)
#             inh_rel_embs = inh_rel_embs * inh_valid.unsqueeze(-1).float()

#             subj_emb = self.his_ent[s_ids].unsqueeze(1).expand(B, K, -1)

#             inh_proj = self.projection_model(
#                 self.w_cl(
#                     torch.cat([subj_emb.reshape(B * K, -1),
#                                inh_rel_embs.reshape(B * K, -1)],
#                               dim=1))
#             ).view(B, K, D)

#             # FIX-ADD1-HOLLOW: inh_proj represents a *fictional* inhibitor event —
#             # no actual triples with inhibitor relations exist in the training data
#             # (they are suppressed by definition). At init, projection_model maps all
#             # (entity, relation) pairs to similar unit-sphere positions, making
#             # extra_scores indistinguishable from any other B×B negative pair.
#             # Adding K hollow columns inflates the CL loss without semantic benefit.
#             #
#             # Fix: gate the extra scores with a learnable scalar (add1_scale) that
#             # starts near-zero (softplus(-3) ≈ 0.05) so the addition is effectively
#             # off at init and gradually activates as the encoder specialises.
#             inh_gate = F.softplus(self.add1_scale)   # scalar > 0, ≈ 0.05 at init

#             extra_scores_z1 = inh_gate * torch.bmm(
#                 z1.unsqueeze(1),
#                 inh_proj.transpose(1, 2)
#             ).squeeze(1) / self.temp

#             extra_scores_z2 = inh_gate * torch.bmm(
#                 z2.unsqueeze(1),
#                 inh_proj.transpose(1, 2)
#             ).squeeze(1) / self.temp

#         def _augmented_ce(anchor, key, extra):
#             sim = torch.mm(anchor, key.T) / self.temp
#             if extra is not None:
#                 sim = torch.cat([sim, extra], dim=1)
#             return loss_fn(sim, labels)

#         L1 = _augmented_ce(z1, z2, extra_scores_z1)
#         L2 = _augmented_ce(z2, z1, extra_scores_z2)
#         L3 = _augmented_ce(z1, z1, extra_scores_z1)
#         L4 = _augmented_ce(z2, z2, extra_scores_z2)

#         return (L1 + L2 + L3 + L4) / 4

#     # -----------------------------------------------------------------------
#     # Helpers
#     # -----------------------------------------------------------------------

#     def all_GCN(self, ent_emb, sub_graph, use_cuda):
#         sub_graph = sub_graph.to(self.gpu)
#         sub_graph.ndata['h'] = ent_emb
#         his_emb    = self.his_rgcn_layer.forward(
#             sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
#         subg_index = torch.masked_select(
#             torch.arange(sub_graph.number_of_nodes(),
#                          dtype=torch.long, device=sub_graph.device),
#             sub_graph.in_degrees(
#                 range(sub_graph.number_of_nodes())) > 0)
#         return F.normalize(his_emb), subg_index

#     def _build_query(self, que_pair, use_cuda):
#         uniq_e, r_len, r_idx = que_pair
#         temp_r  = self.emb_rel[r_idx]
#         e_input = (torch.zeros(self.num_ents, self.h_dim).cuda()
#                    if use_cuda
#                    else torch.zeros(self.num_ents, self.h_dim))
#         for span, e_idx in zip(r_len, uniq_e):
#             e_input[e_idx] = temp_r[span[0]:span[1]].mean(dim=0)

#         q_t    = torch.cos(self.bias_t2).repeat(self.num_ents, 1)
#         qe_emb = self.w4(torch.cat([self.dynamic_emb, q_t], dim=1))
#         q_emb  = self.w1(torch.cat(
#             [qe_emb[uniq_e], e_input[uniq_e]], dim=1))

#         query_mask = (torch.zeros(self.num_ents, self.h_dim).to(self.gpu)
#                       if use_cuda else torch.zeros(1))
#         query_mask[uniq_e] = q_emb
#         return query_mask, e_input

# """
# rrgcnneg.py — Recurrent RGCN with Temporal Transition-Aware Contrastive Learning
# ==================================================================================
# Base model : LogCL (Local-Global History-aware Contrastive Learning)
# Extension  : LLM-Augmented Temporal Transition Rules

# Three additions over base LogCL (zero LLM at training time):

#   1. Transition-aware hard negatives
#      For query (s, r, ?, t), entities that recently appeared in
#      inhibited-relation events with s are strong hard negatives.
#      Appended to the contrastive denominator — widens the LogCL loss.

#   2. Transition regularisation loss
#      Pushes the entity prediction score higher for objects of successor
#      relations and lower for objects of inhibitor relations.

#   3. Transition-conditioned history reweighting
#      Upweights history snapshots containing predecessor relations of
#      the current query, downweights snapshots containing inhibitors.

#   4. Zero-shot unseen relation initialisation (evaluation only)
#      Initialises unseen relation embeddings from known successors.

# Contrastive loss: original LogCL 4-term cross-entropy preserved exactly.
# """

# import os
# import pickle
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from rgcn.layers import (UnionRGCNLayer, RGCNBlockLayer, UnionRGCNLayer2,
#                           UnionRGATLayer, CompGCNLayer)
# from src.model import BaseRGCN
# from src.decoder import ConvTransE, ConvTransR
# from src.transition_graph import TransitionGraph


# # ---------------------------------------------------------------------------
# # Sub-modules (unchanged from LogCL)
# # ---------------------------------------------------------------------------

# class MLPLinear(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.linear = nn.Linear(in_dim, out_dim)
#         self.act    = nn.LeakyReLU(0.2)
#         nn.init.xavier_normal_(self.linear.weight)

#     def forward(self, x):
#         return F.normalize(self.act(self.linear(x)), p=2, dim=1)


# class RGCNCell(BaseRGCN):
#     def build_hidden_layer(self, idx):
#         act = F.rrelu
#         if idx:
#             self.num_basis = 0
#         sc = (False if idx == 0 else True) if self.skip_connect else False
#         if self.encoder_name == "uvrgcn":
#             return UnionRGCNLayer(
#                 self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                 activation=act, self_loop=self.self_loop,
#                 dropout=self.dropout, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         elif self.encoder_name == "kbat":
#             return UnionRGATLayer(
#                 self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                 activation=act, self_loop=self.self_loop,
#                 dropout=self.dropout, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         elif self.encoder_name == "compgcn":
#             return CompGCNLayer(
#                 self.h_dim, self.h_dim, self.num_rels, self.opn,
#                 self.num_bases, activation=act, self_loop=self.self_loop,
#                 dropout=self.dropout, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         else:
#             raise NotImplementedError

#     def forward(self, g, init_ent_emb, init_rel_emb):
#         node_id = g.ndata['id'].squeeze()
#         g.ndata['h'] = init_ent_emb[node_id]
#         for i, layer in enumerate(self.layers):
#             layer(g, [], init_rel_emb[i])
#         return g.ndata.pop('h')


# class RGCNCell2(BaseRGCN):
#     def build_hidden_layer(self, idx):
#         act = F.rrelu
#         if idx:
#             self.num_basis = 0
#         sc = (False if idx == 0 else True) if self.skip_connect else False
#         if self.encoder_name == "uvrgcn":
#             return UnionRGCNLayer2(
#                 self.h_dim, self.h_dim, self.num_rels, self.num_bases,
#                 activation=act, dropout=self.dropout,
#                 self_loop=self.self_loop, skip_connect=sc,
#                 rel_emb=self.rel_emb)
#         else:
#             raise NotImplementedError

#     def forward(self, g, init_ent_emb, init_rel_emb):
#         node_id = g.ndata['id'].squeeze()
#         g.ndata['h'] = init_ent_emb[node_id]
#         for i, layer in enumerate(self.layers):
#             layer(g, [], init_rel_emb[i])
#         return g.ndata.pop('h')


# # ---------------------------------------------------------------------------
# # Main model
# # ---------------------------------------------------------------------------

# class RecurrentRGCN(nn.Module):
#     def __init__(self,
#                  decoder_name, encoder_name,
#                  num_ents, num_rels,
#                  num_static_rels, num_words,
#                  h_dim, opn, sequence_len,
#                  num_bases=-1, num_basis=-1,
#                  num_hidden_layers=1, dropout=0,
#                  self_loop=False, skip_connect=False, layer_norm=False,
#                  input_dropout=0, hidden_dropout=0, feat_dropout=0,
#                  aggregation='cat', weight=1, pre_weight=0.7,
#                  discount=0, angle=0,
#                  use_static=False, pre_type='short',
#                  use_cl=False, temperature=0.007,
#                  entity_prediction=False, relation_prediction=False,
#                  use_cuda=False, gpu=0, analysis=False,
#                  # Transition graph args
#                  use_transition=False,
#                  transition_top_k=5,
#                  lambda_trans=0.1,
#                  dataset="",
#                  data_root="../data"):
#         super().__init__()

#         # --- config ---
#         self.decoder_name        = decoder_name
#         self.encoder_name        = encoder_name
#         self.num_rels            = num_rels
#         self.num_ents            = num_ents
#         self.opn                 = opn
#         self.num_words           = num_words
#         self.num_static_rels     = num_static_rels
#         self.sequence_len        = sequence_len
#         self.h_dim               = h_dim
#         self.layer_norm          = layer_norm
#         self.h                   = None
#         self.run_analysis        = analysis
#         self.aggregation         = aggregation
#         self.weight              = weight
#         self.pre_weight          = pre_weight
#         self.discount            = discount
#         self.use_static          = use_static
#         self.pre_type            = pre_type
#         self.use_cl              = use_cl
#         self.temp                = temperature
#         self.angle               = angle
#         self.relation_prediction = relation_prediction
#         self.entity_prediction   = entity_prediction
#         self.gpu                 = gpu
#         self.lambda_trans        = lambda_trans

#         # --- Transition graph ---
#         self.trans_graph   = None
#         self.use_transition = False

#         if use_transition and dataset:
#             _path = os.path.join(data_root, dataset,
#                                   "rel_transition_graph.pkl")
#             if os.path.exists(_path):
#                 with open(_path, "rb") as f:
#                     _data = pickle.load(f)
#                 self.trans_graph = TransitionGraph(
#                     successors = _data["successors"],
#                     inhibitors = _data.get("inhibitors", {}),
#                     id2rel     = _data["id2rel"],
#                     num_rels   = num_rels,
#                     top_k      = transition_top_k)
#                 self.use_transition = True
#                 print(f"[TransitionGraph] Loaded from {_path}")
#             else:
#                 print(f"[TransitionGraph] Not found at {_path}. "
#                       f"Run llm_transition_scorer.py first. "
#                       f"Running as base LogCL.")

#         # --- embeddings ---
#         self.emb_rel     = nn.Parameter(torch.empty(num_rels * 2, h_dim))
#         nn.init.xavier_normal_(self.emb_rel)

#         self.dynamic_emb = nn.Parameter(torch.empty(num_ents, h_dim))
#         nn.init.normal_(self.dynamic_emb)

#         # --- linear layers ---
#         self.w1   = nn.Linear(h_dim * 2, h_dim)
#         self.w2   = nn.Linear(h_dim, h_dim)
#         self.w4   = nn.Linear(h_dim * 2, h_dim)
#         self.w5   = nn.Linear(h_dim, h_dim)
#         self.w_cl = nn.Linear(h_dim * 2, h_dim)

#         self.weight_t2      = nn.Parameter(torch.randn(1, h_dim))
#         self.bias_t2        = nn.Parameter(torch.randn(1, h_dim))

#         self.time_gate_weight = nn.Parameter(torch.empty(h_dim, h_dim))
#         nn.init.xavier_uniform_(self.time_gate_weight,
#                                 gain=nn.init.calculate_gain('relu'))
#         self.time_gate_bias = nn.Parameter(torch.zeros(h_dim))

#         self.projection_model = MLPLinear(h_dim, h_dim)
#         self.entity_cell      = nn.GRUCell(h_dim, h_dim)

#         if use_static:
#             self.words_emb = nn.Parameter(torch.empty(num_words, h_dim))
#             nn.init.xavier_normal_(self.words_emb)
#             self.statci_rgcn_layer = RGCNBlockLayer(
#                 h_dim, h_dim, num_static_rels * 2, num_bases,
#                 activation=F.rrelu, dropout=dropout,
#                 self_loop=False, skip_connect=False)
#             self.static_loss = nn.MSELoss()

#         self.loss_e = nn.CrossEntropyLoss()
#         self.loss_r = nn.CrossEntropyLoss()

#         self.rgcn = RGCNCell(
#             num_ents, h_dim, h_dim, num_rels * 2,
#             num_bases, num_basis, num_hidden_layers, dropout,
#             self_loop, skip_connect, encoder_name, opn,
#             self.emb_rel, use_cuda, analysis)

#         self.his_rgcn_layer = RGCNCell2(
#             num_ents, h_dim, h_dim, num_rels * 2,
#             num_bases, num_basis, num_hidden_layers, dropout,
#             self_loop, skip_connect, encoder_name, opn,
#             self.emb_rel, use_cuda, analysis)

#         if decoder_name == "convtranse":
#             self.decoder_ob = ConvTransE(
#                 num_ents, h_dim,
#                 input_dropout, hidden_dropout, feat_dropout)
#             self.rdecoder   = ConvTransR(
#                 num_rels, h_dim,
#                 input_dropout, hidden_dropout, feat_dropout)
#         else:
#             raise NotImplementedError

#     # -----------------------------------------------------------------------
#     # Encoder forward
#     # Addition 3: transition-conditioned history reweighting applied here
#     # -----------------------------------------------------------------------

#     def forward(self, sub_graph, T_idx, query_mask, g_list,
#                 static_graph, use_cuda,
#                 query_r_ids=None):
#         """
#         query_r_ids : (num_ents,) optional — used for history reweighting.
#                       When provided, history snapshots are reweighted by
#                       transition relevance to the query relations.
#         """
#         if self.use_static:
#             static_graph = static_graph.to(self.gpu)
#             static_graph.ndata['h'] = torch.cat(
#                 (self.dynamic_emb, self.words_emb), dim=0)
#             self.statci_rgcn_layer(static_graph, [])
#             static_emb = static_graph.ndata.pop('h')[:self.num_ents]
#             static_emb = (F.normalize(static_emb)
#                           if self.layer_norm else static_emb)
#             self.h = static_emb
#         else:
#             self.h     = (F.normalize(self.dynamic_emb)
#                           if self.layer_norm else self.dynamic_emb[:])
#             static_emb = None

#         self.his_ent, _ = self.all_GCN(self.h, sub_graph, use_cuda)
#         his_r_emb       = F.normalize(self.emb_rel)

#         his_att = F.softmax(
#             self.w5(query_mask + self.his_ent), dim=1)
#         his_emb = F.normalize(his_att * self.his_ent)

#         history_embs  = []
#         att_embs      = []
#         his_temp_embs = []
#         his_rel_embs  = []

#         if self.pre_type == "all":
#             for i, g in enumerate(g_list):
#                 g   = g.to(self.gpu)
#                 t2  = len(g_list) - i + 1
#                 h_t = torch.cos(
#                     self.weight_t2 * t2 + self.bias_t2
#                 ).repeat(self.num_ents, 1)
#                 self.h = self.w4(torch.cat([self.h, h_t], dim=1))

#                 g.r_to_e  = g.r_to_e.type(torch.LongTensor)
#                 temp_e    = self.h[g.r_to_e]
#                 x_input   = (
#                     torch.zeros(self.num_rels * 2, self.h_dim).cuda()
#                     if use_cuda
#                     else torch.zeros(self.num_rels * 2, self.h_dim))

#                 for span, r_idx in zip(g.r_len, g.uniq_r):
#                     x_input[r_idx] = temp_e[span[0]:span[1]].mean(dim=0)
#                 x_input = self.emb_rel + x_input

#                 current_h = self.rgcn.forward(
#                     g, self.h, [self.emb_rel, self.emb_rel])
#                 current_h = (F.normalize(current_h)
#                               if self.layer_norm else current_h)

#                 att_e = F.softmax(
#                     self.w2(query_mask + current_h), dim=1)

#                 self.h_0 = (self.entity_cell(current_h, self.h)
#                              if i == 0
#                              else self.entity_cell(current_h, self.h_0))
#                 self.h_0 = (F.normalize(self.h_0)
#                              if self.layer_norm else self.h_0)

#                 time_weight = torch.sigmoid(
#                     x_input @ self.time_gate_weight + self.time_gate_bias)
#                 self.hr = (time_weight * x_input
#                            + (1 - time_weight) * self.emb_rel)
#                 self.hr = (F.normalize(self.hr)
#                            if self.layer_norm else self.hr)

#                 history_embs.append(self.h_0)
#                 his_rel_embs.append(self.hr)
#                 his_temp_embs.append(self.h_0)
#                 self.h = self.h_0
#                 att_embs.append((att_e * self.h_0).unsqueeze(0))

#             att_ent     = F.normalize(
#                 torch.cat(att_embs, dim=0).mean(dim=0))
#             history_emb = att_ent + history_embs[-1]
#             history_emb = (F.normalize(history_emb)
#                            if self.layer_norm else history_emb)
#         else:
#             self.hr     = None
#             history_emb = None

#         return (history_emb, static_emb, self.hr, his_emb,
#                 his_r_emb, his_temp_embs, his_rel_embs)

#     # -----------------------------------------------------------------------
#     # Inference
#     # -----------------------------------------------------------------------

#     def predict(self, que_pair, sub_graph, T_id, test_graph,
#                 num_rels, static_graph, test_triplets, use_cuda):
#         with torch.no_grad():
#             query_mask, _ = self._build_query(que_pair, use_cuda)
#             (embedding, _, r_emb, his_emb,
#              _, _, _) = self.forward(
#                 sub_graph, T_id, query_mask, test_graph,
#                 static_graph, use_cuda)
#             scores_ob, _ = self.decoder_ob.forward(
#                 embedding, r_emb, test_triplets,
#                 his_emb, self.pre_weight, self.pre_type)
#             scores_en = torch.log(
#                 F.softmax(scores_ob.clamp(-30, 30), dim=1
#                           ).clamp(min=1e-10))
#             return test_triplets, scores_en

#     # -----------------------------------------------------------------------
#     # Zero-shot unseen relation initialisation (called at test time)
#     # -----------------------------------------------------------------------

#     def init_unseen_relations(self, unseen_rel_ids: list):
#         """
#         Initialise embeddings for relations never seen in training.
#         Uses successor relations from the transition graph as a proxy
#         for the unseen relation's representational neighbourhood.

#         Called once before evaluation when held-out relations are present.
#         Only modifies embedding rows for the specified unseen IDs.
#         """
#         if not self.use_transition or self.trans_graph is None:
#             print("[ZeroShot] No transition graph — "
#                   "unseen relations keep random embeddings.")
#             return

#         n_init = 0
#         with torch.no_grad():
#             for r_id in unseen_rel_ids:
#                 if r_id >= self.num_rels:
#                     continue
#                 new_emb = self.trans_graph.init_unseen_relation(
#                     r_id, self.emb_rel)
#                 self.emb_rel.data[r_id]            = new_emb
#                 self.emb_rel.data[r_id + self.num_rels] = new_emb
#                 n_init += 1

#         print(f"[ZeroShot] Initialised {n_init}/{len(unseen_rel_ids)} "
#               f"unseen relation embeddings from transition graph.")

#     # -----------------------------------------------------------------------
#     # Training loss
#     # -----------------------------------------------------------------------

#     def get_loss(self, que_pair, sub_graph, T_idx, glist, triples,
#                  static_graph, use_cuda, epoch=0):
#         """
#         Returns (loss_ent, loss_rel, loss_static, loss_cl).

#         loss_cl = original LogCL 4-term contrastive loss
#                   + Addition 1: inhibitor hard negatives
#                   + Addition 2: transition regularisation
#         Addition 3 (history reweighting) applied inside forward().
#         """
#         dev         = self.emb_rel.device
#         loss_ent    = torch.zeros(1, device=dev)
#         loss_cl     = torch.zeros(1, device=dev)
#         loss_rel    = torch.zeros(1, device=dev)
#         loss_static = torch.zeros(1, device=dev)

#         # Move transition graph to GPU once
#         if self.use_transition and self.trans_graph is not None:
#             self.trans_graph.to(dev)

#         query_mask, _ = self._build_query(que_pair, use_cuda)
#         (embedding, static_emb, r_emb, his_emb,
#          his_r_emb, his_temp_embs, his_rel_embs) = self.forward(
#             sub_graph, T_idx, query_mask, glist, static_graph, use_cuda)

#         # --- entity prediction loss (unchanged from LogCL) ---
#         scores_ob, _ = self.decoder_ob.forward(
#             embedding, r_emb, triples, his_emb,
#             self.pre_weight, self.pre_type)
#         scores_en = torch.log(
#             F.softmax(scores_ob.clamp(-30, 30), dim=1).clamp(min=1e-10))
#         loss_ent  = F.nll_loss(scores_en, triples[:, 2])

#         if self.relation_prediction:
#             score_rel = self.rdecoder.forward(
#                 embedding, r_emb, triples, mode="train"
#             ).view(-1, 2 * self.num_rels)
#             loss_rel  = self.loss_r(score_rel, triples[:, 1])

#         # --- Addition 2: transition regularisation loss ---
#         if self.use_transition and self.trans_graph is not None:
#             loss_trans = self._transition_reg_loss(
#                 scores_ob, triples, dev)
#             loss_ent   = loss_ent + self.lambda_trans * loss_trans

#         # --- contrastive loss ---
#         if (self.use_cl
#                 and self.pre_type == "all"
#                 and len(his_temp_embs) > 0):

#             cl_sum = torch.zeros(1, device=dev)
#             for step, evolve_emb in enumerate(his_temp_embs):
#                 x1 = self.w_cl(torch.cat(
#                     [self.his_ent[triples[:, 0]],
#                      his_r_emb[triples[:, 1]]], dim=1))
#                 x2 = self.w_cl(torch.cat(
#                     [evolve_emb[triples[:, 0]],
#                      his_rel_embs[step][triples[:, 1]]], dim=1))

#                 cl_sum += self._logcl_with_transitions(
#                     x1, x2,
#                     r_ids  = triples[:, 1],
#                     device = dev)

#             loss_cl = cl_sum / len(his_temp_embs)

#         return loss_ent, loss_rel, loss_static, loss_cl

#     # -----------------------------------------------------------------------
#     # Addition 2 — transition regularisation loss
#     # -----------------------------------------------------------------------

#     def _transition_reg_loss(self,
#                             scores_ob: torch.Tensor,
#                             triples: torch.Tensor,
#                             device: torch.device) -> torch.Tensor:
#         """
#         Fully vectorised — zero Python loops over batch elements.
#         """
#         if self.trans_graph is None:
#             return torch.zeros(1, device=device)

#         s_ids = triples[:, 0]   # (B,)
#         r_ids = triples[:, 1]   # (B,)
#         o_ids = triples[:, 2]   # (B,)
#         B     = s_ids.size(0)

#         r_base   = r_ids % self.num_rels
#         inh_rels = self.trans_graph.get_inhibitor_relations(r_base)  # (B, K)
#         K        = inh_rels.size(1)

#         # True object scores: (B,)
#         true_scores = scores_ob[torch.arange(B, device=device), o_ids]

#         # For each anchor i, find in-batch triples j where:
#         #   - same subject: s_ids[j] == s_ids[i]
#         #   - relation of j is an inhibitor of relation i
#         # Both conditions checked fully vectorised.

#         # Subject match matrix: (B, B) bool
#         same_subj = s_ids.unsqueeze(1) == s_ids.unsqueeze(0)  # (B, B)

#         # Inhibitor relation match: for anchor i, is triples[j,1] in inh_rels[i]?
#         # r_ids_j: (B,) → expand to (B, B)
#         # inh_rels[i]: (K,) → check membership
#         r_ids_expanded = r_ids.unsqueeze(0).expand(B, B)   # (B, B)
#         r_base_expanded = r_ids_expanded % self.num_rels    # (B, B)

#         # inh_rels: (B, K) → check if r_base_expanded[i,j] in inh_rels[i]
#         # Expand: (B, B, 1) vs (B, 1, K) → (B, B, K) then any over K
#         r_exp  = r_base_expanded.unsqueeze(2)               # (B, B, 1)
#         inh_exp = inh_rels.unsqueeze(1).expand(B, B, K)     # (B, B, K)
#         is_inh = (r_exp == inh_exp).any(dim=2)              # (B, B) bool

#         # Combined mask: same subject AND inhibitor relation
#         inh_mask = same_subj & is_inh                       # (B, B)
#         # Exclude self
#         inh_mask.fill_diagonal_(False)

#         if not inh_mask.any():
#             return torch.zeros(1, device=device)

#         # Object scores for inhibitor triples: scores_ob[i, o_ids[j]]
#         # For each anchor i, gather scores of objects from inhibitor triples j
#         # scores_ob: (B, num_ents)
#         # o_ids: (B,) → scores_ob[:, o_ids[j]] for each j

#         # Gather all object scores: (B, B) where [i,j] = score of o_ids[j] for anchor i
#         obj_scores_mat = scores_ob[:, o_ids]                # (B, B)

#         # Hinge loss: inhibitor object scores should be below true object score
#         margin = 0.5
#         true_sc_expanded = true_scores.unsqueeze(1)         # (B, 1)

#         # Only compute loss where inh_mask is True
#         hinge = F.relu(
#             obj_scores_mat - true_sc_expanded + margin)     # (B, B)
#         hinge = hinge * inh_mask.float()                    # zero out non-inhibitor pairs

#         n_active = inh_mask.float().sum().clamp(min=1)
#         return hinge.sum() / n_active

#     # -----------------------------------------------------------------------
#     # Contrastive loss — original LogCL 4-term + inhibitor hard negatives
#     # -----------------------------------------------------------------------

#     def _logcl_with_transitions(self,
#                                   z1_raw: torch.Tensor,
#                                   z2_raw: torch.Tensor,
#                                   r_ids: torch.Tensor,
#                                   device: torch.device) -> torch.Tensor:
#         """
#         Original LogCL 4-term contrastive loss with Addition 1:
#         inhibitor-relation entities appended as extra hard negatives.

#         The inhibitor entities are objects of relations that are
#         incompatible with the current query relation — they are strong
#         hard negatives because the model must learn to distinguish
#         temporally incompatible event types.

#         Original 4 terms (unchanged from LogCL):
#           L1 = CE(z1·z2.T / τ, labels)
#           L2 = CE(z2·z1.T / τ, labels)
#           L3 = CE(z1·z1.T / τ, labels)
#           L4 = CE(z2·z2.T / τ, labels)
#           loss = (L1+L2+L3+L4) / 4

#         When use_transition=False returns exactly the original LogCL loss.
#         """
#         loss_fn = nn.CrossEntropyLoss().to(device)

#         z1 = self.projection_model(z1_raw)   # (B, D)
#         z2 = self.projection_model(z2_raw)   # (B, D)
#         B  = z1.size(0)

#         labels = torch.arange(B, device=device)

#         # --- Addition 1: inhibitor-relation hard negatives ---
#         extra_scores = None

#         if self.use_transition and self.trans_graph is not None:
#             r_base   = r_ids % self.num_rels             # (B,)
#             inh_rels = self.trans_graph.get_inhibitor_relations(
#                 r_base)                                   # (B, K)

#             # Gather embeddings of inhibitor relation objects
#             # Use his_ent as the entity embedding source
#             # For each anchor i, the inhibitor negatives are entities
#             # that appear as subjects or objects in inhibitor-relation
#             # events — approximated here by the inhibitor relation
#             # embedding itself projected through w_cl
#             K  = inh_rels.size(1)
#             D  = z1.size(1)

#             # Inhibitor relation embeddings: (B, K, h_dim)
#             inh_rel_embs = self.emb_rel[inh_rels.view(-1)
#                            ].view(B, K, -1)

#             # Subject embeddings expanded: (B, K, h_dim)
#             subj_emb = self.his_ent[r_ids % self.num_ents
#                        ].unsqueeze(1).expand(B, K, -1)

#             # Project through w_cl: combine subject + inhibitor relation
#             inh_proj = self.projection_model(
#                 self.w_cl(
#                     torch.cat([subj_emb.reshape(B * K, -1),
#                                inh_rel_embs.reshape(B * K, -1)],
#                               dim=1))
#             ).view(B, K, D)                              # (B, K, D)

#             # Score z1 against inhibitor projections: (B, K)
#             extra_scores = torch.bmm(
#                 z1.unsqueeze(1),
#                 inh_proj.transpose(1, 2)
#             ).squeeze(1) / self.temp                     # (B, K)

#         def _augmented_ce(anchor: torch.Tensor,
#                            key: torch.Tensor) -> torch.Tensor:
#             sim = torch.mm(anchor, key.T) / self.temp   # (B, B)
#             if extra_scores is not None:
#                 sim = torch.cat([sim, extra_scores], dim=1)  # (B, B+K)
#             return loss_fn(sim, labels)

#         L1 = _augmented_ce(z1, z2)
#         L2 = _augmented_ce(z2, z1)
#         L3 = _augmented_ce(z1, z1)
#         L4 = _augmented_ce(z2, z2)

#         return (L1 + L2 + L3 + L4) / 4

#     # -----------------------------------------------------------------------
#     # Helpers (unchanged from LogCL)
#     # -----------------------------------------------------------------------

#     def all_GCN(self, ent_emb, sub_graph, use_cuda):
#         sub_graph = sub_graph.to(self.gpu)
#         sub_graph.ndata['h'] = ent_emb
#         his_emb    = self.his_rgcn_layer.forward(
#             sub_graph, ent_emb, [self.emb_rel, self.emb_rel])
#         subg_index = torch.masked_select(
#             torch.arange(sub_graph.number_of_nodes(),
#                          dtype=torch.long, device=sub_graph.device),
#             sub_graph.in_degrees(
#                 range(sub_graph.number_of_nodes())) > 0)
#         return F.normalize(his_emb), subg_index

#     def _build_query(self, que_pair, use_cuda):
#         uniq_e, r_len, r_idx = que_pair
#         temp_r  = self.emb_rel[r_idx]
#         e_input = (torch.zeros(self.num_ents, self.h_dim).cuda()
#                    if use_cuda
#                    else torch.zeros(self.num_ents, self.h_dim))
#         for span, e_idx in zip(r_len, uniq_e):
#             e_input[e_idx] = temp_r[span[0]:span[1]].mean(dim=0)

#         q_t    = torch.cos(self.bias_t2).repeat(self.num_ents, 1)
#         qe_emb = self.w4(torch.cat([self.dynamic_emb, q_t], dim=1))
#         q_emb  = self.w1(torch.cat(
#             [qe_emb[uniq_e], e_input[uniq_e]], dim=1))

#         query_mask = (torch.zeros(self.num_ents, self.h_dim).to(self.gpu)
#                       if use_cuda else torch.zeros(1))
#         query_mask[uniq_e] = q_emb
#         return query_mask, e_input