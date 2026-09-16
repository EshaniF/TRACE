"""
rrgcntr.py — Recurrent RGCN with Transition Graph Extensions

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
from src.frequency_rerank import FrequencyReranker


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
                 data_root="../data",
                 transition_graph_path=None,
                 rerank_mode="transition"):
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
        self.rerank_mode         = rerank_mode
        self.use_transition      = False

        # --- Transition graph ---
        # transition_graph_path lets you point at an alternative graph
        # (empirical / random / shuffled baselines from
        # build_baseline_graphs.py) without touching the dataset's
        # default rel_transition_graph.pkl.
        self.trans_graph = None
        if use_transition and dataset:
            _path = (transition_graph_path if transition_graph_path
                      else os.path.join(data_root, dataset,
                                         "rel_transition_graph.pkl"))
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
                      f"Relation reg (lambda={lambda_trans}) active. "
                      f"Re-ranking mode = '{rerank_mode}' (alpha={rerank_alpha}).")
            else:
                print(f"[TransitionGraph] Not found at {_path}. "
                      f"Run llm_transition_scorer.py / "
                      f"build_baseline_graphs.py first. "
                      f"Running as base LogCL.")

        # --- Graph-free re-ranking baseline (Mechanism B only) ---
        # Active independently of use_transition/trans_graph: you can
        # run --rerank-mode frequency without ever loading a transition
        # graph at all, or combine --use-transition (for Mechanism A
        # training) with --rerank-mode frequency to isolate Mechanism A
        # from Mechanism B at evaluation time.
        self.freq_reranker = None
        if rerank_mode in ("frequency", "recency"):
            self.freq_reranker = FrequencyReranker(mode=rerank_mode)
            print(f"[Rerank] Using graph-free '{rerank_mode}' baseline "
                  f"reranker (alpha={rerank_alpha}).")

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


    def move_transition_graph_to_device(self, device):
        if self.use_transition and self.trans_graph is not None:
            self.trans_graph.to(device)
        if self.freq_reranker is not None:
            self.freq_reranker.to(device)

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

    def predict(self, que_pair, sub_graph, T_id, test_graph,
                num_rels, static_graph, test_triplets, use_cuda,
                recent_events=None):
        """
        Parameters
        ----------
        recent_events : list of (s, r, o) int tuples from the most recent
                        history snapshots.  Passed in from test() in
                        maintr.py.  Used only when self.rerank_mode is
                        not "none" and recent_events is non-empty.
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

            # --- B. Inference-time re-ranking (mode-dispatched) ----------
            if (recent_events is not None
                    and len(recent_events) > 0
                    and self.rerank_mode != "none"):

                if (self.rerank_mode == "transition"
                        and self.use_transition
                        and self.trans_graph is not None):
                    # Pass test_triplets directly so rerank_scores derives
                    # s_i and r_i per row, keeping loop index i aligned
                    # with the N rows of scores_en.
                    scores_en = self.trans_graph.rerank_scores(
                        log_scores    = scores_en,
                        test_triplets = test_triplets,
                        recent_events = recent_events,
                        rerank_alpha  = self.rerank_alpha)

                elif self.rerank_mode in ("frequency", "recency") \
                        and self.freq_reranker is not None:
                    scores_en = self.freq_reranker.rerank_scores(
                        log_scores    = scores_en,
                        test_triplets = test_triplets,
                        recent_events = recent_events,
                        rerank_alpha  = self.rerank_alpha)
                # else: rerank_mode == "transition" but no graph was
                # loaded (e.g. --use-transition not passed, or the
                # path was missing) — falls through to the base scores
                # unchanged, same as the original behaviour.

            return test_triplets, scores_en

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
        # in the current batch — dense gradient signal. Unaffected by
        # rerank_mode: this only depends on trans_graph being loaded, so
        # you can hold Mechanism A fixed while sweeping --rerank-mode at
        # evaluation time.
        if self.use_transition and self.trans_graph is not None:
            loss_reg  = self.trans_graph.relation_reg_loss(self.emb_rel)
            loss_ent  = loss_ent + self.lambda_trans * loss_reg


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
