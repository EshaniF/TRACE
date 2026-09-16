"""
maintr.py — Training script for LogCL + Temporal Transition Extensions
=========================================================================
Base model : LogCL
Extension  : LLM-Augmented Temporal Transition Rules

Two mechanisms replace the original four additions:

  A. Relation embedding regularisation  [training]
     Constrains the relation embedding matrix geometrically using the
     transition graph.  Controlled by --lambda-trans.

  B. Inference-time re-ranking  [evaluation]
     Post-hoc log-score correction using recent event history and the
     transition graph.  Controlled by --rerank-alpha and --rerank-mode.

Usage
-----
  # Step 1: build the transition graph (once per dataset)
  python llm_transition_scorer.py -d ICEWS14

  # Step 1b (optional): build baseline graphs for comparison
  python build_baseline_graphs.py -d ICEWS14 --mode all --seed 0

  # Step 2: train
  python maintr.py -d ICEWS14 --use-transition

  # Base LogCL (no transition graph):
  python maintr.py -d ICEWS14 --rerank-mode none

  # Evaluate only:
  python maintr.py -d ICEWS14 --use-transition --test

  # Ablate re-ranking only (training regularisation only):
  python maintr.py -d ICEWS14 --use-transition --rerank-alpha 0.0

  # Ablate regularisation only (re-ranking only):
  python maintr.py -d ICEWS14 --use-transition --lambda-trans 0.0

  # Train and Evaluate against the empirical baseline graph, saving ranks:
  python maintr.py -d ICEWS14 --use-transition \\
      --transition-graph-path ../data/ICEWS14/rel_transition_graph_empirical.pkl \\
      --save-ranks --save-ranks-tag icews14_empirical

  # Evaluate against random / shuffled baseline graphs:
  python maintr.py -d ICEWS14 --use-transition \\
      --transition-graph-path ../data/ICEWS14/rel_transition_graph_random_seed0.pkl \\
      --save-ranks --save-ranks-tag icews14_random_seed0

  python maintr.py -d ICEWS14 --use-transition \\
      --transition-graph-path ../data/ICEWS14/rel_transition_graph_shuffled_seed0.pkl \\
      --save-ranks --save-ranks-tag icews14_shuffled_seed0


"""

import csv
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import argparse
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings('ignore')

sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph, build_graph
from src.rrgcntr import RecurrentRGCN
from rgcn.knowledge_graph import _read_triplets_as_list



def e2r(triplets, num_rels, use_cuda=True):
    src, rel, dst = triplets.transpose()
    uniq_e  = np.unique(src)
    e_to_r  = defaultdict(set)
    for s, r, d in zip(src, rel, dst):
        e_to_r[s].add(r)
    r_len, r_idx, idx = [], [], 0
    for e in uniq_e:
        r_len.append((idx, idx + len(e_to_r[e])))
        r_idx.extend(list(e_to_r[e]))
        idx += len(e_to_r[e])
    ue = torch.from_numpy(uniq_e).long()
    rl = torch.from_numpy(np.array(r_len)).long()
    ri = torch.from_numpy(np.array(r_idx)).long()
    if use_cuda:
        ue, rl, ri = ue.cuda(), rl.cuda(), ri.cuda()
    return [ue, rl, ri]


def get_sample_from_history_graph3(subg_arr, sr_to_sro, triples,
                                    num_nodes, num_rels, use_cuda, gpu):
    import pandas as pd
    inverse_triples = triples[:, [2, 1, 0]].copy()
    inverse_triples[:, 1] += num_rels

    inv_subg    = subg_arr[:, [2, 1, 0]].copy()
    inv_subg[:, 1] += num_rels
    subg_all    = np.concatenate([subg_arr, inv_subg])
    df          = pd.DataFrame(subg_all, columns=['src', 'rel', 'dst'])
    subg_df     = df.groupby(df.columns.tolist()).size().reset_index(
    ).rename(columns={0: 'freq'})

    df_dic = pd.DataFrame({'sr':  list(sr_to_sro.keys()),
                            'dst': list(sr_to_sro.values())})

    def _collect(ent_set, er):
        rows  = df_dic.query('sr in @er')
        two   = set().union(*rows['dst'].values) if len(rows) else set()
        all_e = list(ent_set | two)
        return subg_df.query('src in @all_e').to_numpy()

    src_set     = set(triples[:, 0])
    dst_set     = set(triples[:, 2])
    er_list     = list(set((t[0], t[1]) for t in triples))
    er_list_inv = list(set((t[0], t[1]) for t in inverse_triples))

    q_tri     = _collect(src_set, er_list)
    q_tri_inv = _collect(dst_set, er_list_inv)

    return (build_graph(num_nodes, num_rels, q_tri,     use_cuda, gpu),
            build_graph(num_nodes, num_rels, q_tri_inv, use_cuda, gpu))


# ---------------------------------------------------------------------------
# Build recent_events list from history snapshots
# ---------------------------------------------------------------------------

def build_recent_events(input_list: list) -> list:
    """
    Flatten the most recent history snapshots into a list of
    (s, r, o) integer tuples for use by rerank_scores().

    Parameters
    ----------
    input_list : list of np.ndarray, each (N, 3+) — recent history windows
                 in chronological order; last entry is the most recent.

    Returns
    -------
    list of (int, int, int) tuples — all (s, r, o) events across windows.
    Events from more recent windows appear later in the list but all are
    treated equally by rerank_scores (no temporal decay applied here).
    """
    events = []
    for snap in input_list:
        for row in snap:
            events.append((int(row[0]), int(row[1]), int(row[2])))
    return events



def test(model, history_list, test_list, num_rels, num_nodes, use_cuda,
         all_ans_list, all_ans_r_list, model_name, static_graph,
         mode, args):

    ranks_raw,     ranks_filter     = [], []
    ranks_raw_inv, ranks_filter_inv = [], []

    if mode == "test":
        print("Loading model from:", model_name)
        ckpt = torch.load(model_name,
                          map_location=torch.device(
                              args.gpu if use_cuda else 'cpu'))
        print(f"  Best epoch: {ckpt['epoch']}")
        model.load_state_dict(ckpt['state_dict'])
        model.eval()

    input_list = list(history_list[-args.test_history_len:])
    subg_arr   = np.concatenate(history_list[:])
    sr_to_sro  = np.load(
        f'../data/{args.dataset}/his_dict_new/train_s_r.npy',
        allow_pickle=True).item()

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g,
                                          use_cuda, args.gpu)
                         for g in input_list]

        inv_snap = test_snap[:, [2, 1, 0]].copy()
        inv_snap[:, 1] += num_rels

        sub, sub_inv = get_sample_from_history_graph3(
            subg_arr, sr_to_sro, test_snap,
            num_nodes, num_rels, use_cuda, args.gpu)

        test_t = (torch.LongTensor(test_snap).cuda() if use_cuda
                  else torch.LongTensor(test_snap))
        inv_t  = (torch.LongTensor(inv_snap).cuda()  if use_cuda
                  else torch.LongTensor(inv_snap))

        # Build recent_events from current input_list for re-ranking.
        # Both forward and inverse passes share the same recent history.
        recent_events = build_recent_events(input_list)

        triples,     scores     = model.predict(
            e2r(test_snap, num_rels, use_cuda), sub, time_idx,
            history_glist, num_rels, static_graph, test_t, use_cuda,
            recent_events=recent_events)
        inv_triples, inv_scores = model.predict(
            e2r(inv_snap,  num_rels, use_cuda), sub_inv, time_idx,
            history_glist, num_rels, static_graph, inv_t, use_cuda,
            recent_events=recent_events)

        _, _, rr,  rf  = utils.get_total_rank(
            triples,     scores,     all_ans_list[time_idx],
            eval_bz=1000, rel_predict=0)
        _, _, rri, rfi = utils.get_total_rank(
            inv_triples, inv_scores, all_ans_list[time_idx],
            eval_bz=1000, rel_predict=0)

        ranks_raw.append(rr);      ranks_filter.append(rf)
        ranks_raw_inv.append(rri); ranks_filter_inv.append(rfi)

        if args.multi_step:
            pred = utils.construct_snap(
                triples, num_nodes, num_rels, scores, args.topk)
            if len(pred):
                input_list.pop(0); input_list.append(pred)
        else:
            input_list.pop(0); input_list.append(test_snap)

    mrr_raw,    hit_raw    = utils.stat_ranks(ranks_raw,        "raw")
    mrr_filter, hit_filter = utils.stat_ranks(ranks_filter,     "filter")
    mrr_raw_i,  hit_raw_i  = utils.stat_ranks(ranks_raw_inv,    "raw_inv")
    mrr_filt_i, hit_filt_i = utils.stat_ranks(ranks_filter_inv, "filter_inv")

    all_mrr_raw    = (mrr_raw    + mrr_raw_i)  / 2
    all_mrr_filter = (mrr_filter + mrr_filt_i) / 2
    all_hit_raw    = [(hit_raw[i]    + hit_raw_i[i])    / 2
                      for i in range(len(hit_raw))]
    all_hit_filter = [(hit_filter[i] + hit_filt_i[i]) / 2
                      for i in range(len(hit_filter))]

    print("(all_raw)    MRR, H@1,3,10: {:.4f} {:.4f} {:.4f} {:.4f}".format(
        all_mrr_raw.item(), *all_hit_raw[:3]))
    print("(all_filter) MRR, H@1,3,10: {:.4f} {:.4f} {:.4f} {:.4f}".format(
        all_mrr_filter.item(), *all_hit_filter[:3]))


    if getattr(args, "save_ranks", False):
        all_filtered_ranks = torch.cat(
            [torch.cat(ranks_filter), torch.cat(ranks_filter_inv)]
        ).cpu().numpy()
        os.makedirs('../result/ranks', exist_ok=True)
        tag = args.save_ranks_tag or f"{args.dataset}_{args.rerank_mode}"
        out_path = f'../result/ranks/{tag}_ranks.npy'
        np.save(out_path, all_filtered_ranks)
        print(f"[SaveRanks] Saved {len(all_filtered_ranks)} per-query "
              f"filtered ranks to {out_path}")
    # ------------------------------------------------------------------------

    if mode == "test":
        fname        = f'../result/{args.dataset}.csv'
        write_header = not os.path.isfile(fname)
        os.makedirs('../result', exist_ok=True)
        with open(fname, 'a', newline='') as f:
            cols = ['encoder', 'opn', 'pre_type', 'use_static',
                    'use_cl', 'use_transition',
                    'lambda_trans', 'rerank_alpha', 'rerank_mode',
                    'transition_graph_path',
                    'transition_top_k', 'gpu', 'datetime',
                    'pre_weight', 'train_len', 'test_len',
                    'temperature', 'lr', 'n_hidden',
                    'filter_MRR',     'filter_H@1',
                    'filter_H@3',     'filter_H@10',
                    'filter_inv_MRR', 'filter_inv_H@1',
                    'filter_inv_H@3', 'filter_inv_H@10',
                    'all_MRR',        'all_H@1',
                    'all_H@3',        'all_H@10',
                    'filter_all_MRR', 'filter_all_H@1',
                    'filter_all_H@3', 'filter_all_H@10']
            w = csv.DictWriter(f, fieldnames=cols)
            if write_header:
                w.writeheader()
            w.writerow({
                'encoder':        args.encoder,
                'opn':            args.opn,
                'pre_type':       args.pre_type,
                'use_static':     args.add_static_graph,
                'use_cl':         args.use_cl,
                'use_transition': args.use_transition,
                'lambda_trans':   args.lambda_trans,
                'rerank_alpha':   args.rerank_alpha,
                'rerank_mode':    args.rerank_mode,
                'transition_graph_path': args.transition_graph_path or 'default',
                'transition_top_k': args.transition_top_k,
                'gpu':            args.gpu,
                'datetime':       datetime.now(),
                'pre_weight':     args.pre_weight,
                'train_len':      args.train_history_len,
                'test_len':       args.test_history_len,
                'temperature':    args.temperature,
                'lr':             args.lr,
                'n_hidden':       args.n_hidden,
                'filter_MRR':         float(mrr_filter),
                'filter_H@1':         hit_filter[0],
                'filter_H@3':         hit_filter[1],
                'filter_H@10':        hit_filter[2],
                'filter_inv_MRR':     float(mrr_filt_i),
                'filter_inv_H@1':     hit_filt_i[0],
                'filter_inv_H@3':     hit_filt_i[1],
                'filter_inv_H@10':    hit_filt_i[2],
                'all_MRR':            all_mrr_raw.item(),
                'all_H@1':            all_hit_raw[0],
                'all_H@3':            all_hit_raw[1],
                'all_H@10':           all_hit_raw[2],
                'filter_all_MRR':     all_mrr_filter.item(),
                'filter_all_H@1':     all_hit_filter[0],
                'filter_all_H@3':     all_hit_filter[1],
                'filter_all_H@10':    all_hit_filter[2],
            })

    return all_mrr_raw, all_mrr_filter


def run_experiment(args, n_hidden=None, n_layers=None,
                   dropout=None, n_bases=None):

    if n_hidden: args.n_hidden = n_hidden
    if n_layers: args.n_layers = n_layers
    if dropout:  args.dropout  = dropout
    if n_bases:  args.n_bases  = n_bases

    print("Loading graph data...")
    data       = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list  = utils.split_by_time(data.test)
    num_nodes, num_rels = data.num_nodes, data.num_rels

    all_ans_test    = utils.load_all_answers_for_time_filter(
        data.test,  num_rels, num_nodes, False)
    all_ans_r_test  = utils.load_all_answers_for_time_filter(
        data.test,  num_rels, num_nodes, True)
    all_ans_valid   = utils.load_all_answers_for_time_filter(
        data.valid, num_rels, num_nodes, False)
    all_ans_r_valid = utils.load_all_answers_for_time_filter(
        data.valid, num_rels, num_nodes, True)

    model_state_file = f'../models/ctstkg_{args.dataset}_{args.lambda_trans}'
    use_cuda         = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(_read_triplets_as_list(
            f"../data/{args.dataset}/e-w-graph.txt", {}, {},
            load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words       = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] += num_nodes
        static_node_id = (
            torch.from_numpy(np.arange(num_words + num_nodes))
            .view(-1, 1).long().cuda(args.gpu)
            if use_cuda
            else torch.from_numpy(np.arange(num_words + num_nodes))
            .view(-1, 1).long())
    else:
        num_static_rels, num_words, static_triples = 0, 0, []
        static_graph = None

    # --- Build model ---
    model = RecurrentRGCN(
        decoder_name     = args.decoder,
        encoder_name     = args.encoder,
        num_ents         = num_nodes,
        num_rels         = num_rels,
        num_static_rels  = num_static_rels,
        num_words        = num_words,
        h_dim            = args.n_hidden,
        opn              = args.opn,
        sequence_len     = args.train_history_len,
        num_bases        = args.n_bases,
        num_basis        = args.n_basis,
        num_hidden_layers= args.n_layers,
        dropout          = args.dropout,
        self_loop        = args.self_loop,
        skip_connect     = args.skip_connect,
        layer_norm       = args.layer_norm,
        input_dropout    = args.input_dropout,
        hidden_dropout   = args.hidden_dropout,
        feat_dropout     = args.feat_dropout,
        aggregation      = args.aggregation,
        weight           = args.weight,
        pre_weight       = args.pre_weight,
        discount         = args.discount,
        angle            = args.angle,
        use_static       = args.add_static_graph,
        pre_type         = args.pre_type,
        use_cl           = args.use_cl,
        temperature      = args.temperature,
        entity_prediction    = args.entity_prediction,
        relation_prediction  = args.relation_prediction,
        use_cuda         = use_cuda,
        gpu              = args.gpu,
        analysis         = args.run_analysis,
        use_transition   = args.use_transition,
        transition_top_k = args.transition_top_k,
        lambda_trans     = args.lambda_trans,
        rerank_alpha     = args.rerank_alpha,
        dataset          = args.dataset,
        data_root        = args.data_root,
        transition_graph_path = args.transition_graph_path,
        rerank_mode           = args.rerank_mode,
    )

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()
        model.move_transition_graph_to_device(
            torch.device(f'cuda:{args.gpu}'))
    else:
        model.move_transition_graph_to_device(torch.device('cpu'))

    if args.add_static_graph:
        static_graph = build_sub_graph(
            len(static_node_id), num_static_rels,
            static_triples, use_cuda, args.gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5)

    print(f"\nTransition extensions: "
          f"{'ON' if args.use_transition else 'OFF (base LogCL)'}")
    if args.use_transition:
        print(f"  Relation reg   : lambda_trans = {args.lambda_trans}")
    print(f"  Re-rank mode   : {args.rerank_mode} "
          f"(alpha = {args.rerank_alpha})")
    if args.transition_graph_path:
        print(f"  Graph path override : {args.transition_graph_path}")

    # --- Test mode ---
    if args.test and os.path.exists(model_state_file):
        return test(model, train_list + valid_list, test_list,
                    num_rels, num_nodes, use_cuda,
                    all_ans_test, all_ans_r_test,
                    model_state_file, static_graph, "test", args)

    if args.test:
        print(f"{model_state_file} not found — switching to train mode.")

    # --- Training ---
    print("--- Training ---")
    best_mrr   = 0.0
    no_improve = 0
    ema_mrr    = None
    ema_alpha  = 0.5
    avgloss    = []

    for epoch in range(args.n_epochs):
        model.train()
        losses, losses_e, losses_r = [], [], []

        for train_sample_num in tqdm(range(1, len(train_list))):
            output     = train_list[train_sample_num:train_sample_num + 1]
            lo         = max(0, train_sample_num - args.train_history_len)
            input_list = train_list[lo:train_sample_num]

            subg_arr = np.load(
                f'../data/{args.dataset}/his_graph_for_new/'
                f'train_s_r_{train_sample_num}.npy',
                allow_pickle=True)
            subg_arr_inv = np.load(
                f'../data/{args.dataset}/his_graph_inv_new/'
                f'train_o_r_{train_sample_num}.npy',
                allow_pickle=True)
            subg     = build_graph(num_nodes, num_rels, subg_arr,
                                    use_cuda, args.gpu)
            subg_inv = build_graph(num_nodes, num_rels, subg_arr_inv,
                                    use_cuda, args.gpu)

            inv = output[0][:, [2, 1, 0]].copy()
            inv[:, 1] += num_rels

            history_glist = [build_sub_graph(num_nodes, num_rels, s,
                                              use_cuda, args.gpu)
                             for s in input_list]

            t_col      = np.full((output[0].shape[0], 1),
                                 train_sample_num, dtype=np.int64)
            triples_np = np.concatenate([output[0], t_col], axis=1)
            inv_t_col  = np.full((inv.shape[0], 1),
                                 train_sample_num, dtype=np.int64)
            inv_np     = np.concatenate([inv, inv_t_col], axis=1)

            triples_t  = torch.from_numpy(triples_np).long()
            inv_t      = torch.from_numpy(inv_np).long()
            if use_cuda:
                triples_t = triples_t.cuda()
                inv_t     = inv_t.cuda()

            for pass_id in range(2):
                if pass_id == 0:
                    qp   = e2r(output[0], num_rels, use_cuda)
                    sg   = subg
                    trip = triples_t
                else:
                    qp   = e2r(inv, num_rels, use_cuda)
                    sg   = subg_inv
                    trip = inv_t

                le, lr, ls, lcl = model.get_loss(
                    qp, sg, train_sample_num,
                    history_glist, trip,
                    static_graph, use_cuda)

                loss = le + lr + ls + lcl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.item())
                losses_e.append(le.item())
                losses_r.append(lr.item())

        avgloss.append(np.mean(losses))
        print("Epoch {:04d} | Loss {:.4f} | ent {:.4f} | rel {:.4f} | "
              "best MRR {:.4f}".format(
                  epoch, np.mean(losses), np.mean(losses_e),
                  np.mean(losses_r), best_mrr))

        # --- Validation ---
        if epoch and epoch % args.evaluate_every == 0:
            mrr_raw, mrr_filter = test(
                model, train_list, valid_list,
                num_rels, num_nodes, use_cuda,
                all_ans_valid, all_ans_r_valid,
                model_state_file, static_graph,
                mode="train", args=args)

            if not args.relation_evaluation:
                filt    = float(mrr_filter)
                ema_mrr = filt if ema_mrr is None else (
                    ema_alpha * filt + (1 - ema_alpha) * ema_mrr)

                if ema_mrr > best_mrr:
                    best_mrr   = ema_mrr
                    no_improve = 0
                    torch.save({'state_dict': model.state_dict(),
                                'epoch': epoch},
                               model_state_file)
                else:
                    no_improve += 1

                if (epoch >= args.early_stop_min_epochs
                        and no_improve >= args.patience):
                    print(f"Early stopping at epoch {epoch}.")
                    break

        torch.cuda.empty_cache()

    np.savetxt('lossval.txt', avgloss)
    plt.plot(avgloss)
    plt.title('Training loss')
    plt.savefig('lossfig.png')

    return test(model, train_list + valid_list, test_list,
                num_rels, num_nodes, use_cuda,
                all_ans_test, all_ans_r_test,
                model_state_file, static_graph,
                mode="test", args=args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='LogCL + Temporal Transition Extensions')

    # hardware
    parser.add_argument("--gpu",               type=int,   default=0)

    # dataset
    parser.add_argument("-d", "--dataset",     type=str,   default="ICEWS14")
    parser.add_argument("--data-root",         type=str,   default="../data")

    # mode
    parser.add_argument("--test",              action='store_true', default=False)
    parser.add_argument("--run-analysis",      action='store_true', default=False)
    parser.add_argument("--multi-step",        action='store_true', default=False)
    parser.add_argument("--topk",              type=int,   default=10)

    # graph
    parser.add_argument("--add-static-graph",  action='store_true', default=True)
    parser.add_argument("--relation-evaluation", action='store_true', default=False)

    # model
    parser.add_argument("--encoder",           type=str,   default="uvrgcn")
    parser.add_argument("--decoder",           type=str,   default="convtranse")
    parser.add_argument("--opn",               type=str,   default="sub")
    parser.add_argument("--aggregation",       type=str,   default="none")
    parser.add_argument("--n-hidden",          type=int,   default=200)
    parser.add_argument("--n-bases",           type=int,   default=100)
    parser.add_argument("--n-basis",           type=int,   default=100)
    parser.add_argument("--n-layers",          type=int,   default=2)
    parser.add_argument("--self-loop",         action='store_true', default=True)
    parser.add_argument("--skip-connect",      action='store_true', default=False)
    parser.add_argument("--layer-norm",        action='store_true', default=False)
    parser.add_argument("--relation-prediction", action='store_true', default=False)
    parser.add_argument("--entity-prediction",   action='store_true', default=True)

    # LogCL
    parser.add_argument("--pre-type",          type=str,   default="all")
    parser.add_argument("--use-cl",            action='store_true', default=True)
    parser.add_argument("--temperature",       type=float, default=0.03)
    parser.add_argument("--weight",            type=float, default=1)
    parser.add_argument("--pre-weight",        type=float, default=0.9)
    parser.add_argument("--discount",          type=float, default=1)
    parser.add_argument("--angle",             type=int,   default=10)

    # Transition graph
    parser.add_argument("--use-transition",    action='store_true', default=True,
                        help="Enable temporal transition extensions.")
    parser.add_argument("--transition-top-k",  type=int,   default=10)
    parser.add_argument("--lambda-trans",      type=float, default=0.4,
                        help="Weight for relation embedding regularisation "
                             "loss (A). Set 0.0 to ablate.")
    parser.add_argument("--rerank-alpha",      type=float, default=0.3,
                        help="Strength of inference-time re-ranking "
                             "correction (B). Set 0.0 to ablate.")

    # NEW — baseline graph / re-ranking-mode selection
    parser.add_argument("--transition-graph-path", type=str, default=None,
                        help="Override path to the transition graph "
                             "pickle. Defaults to <data-root>/<dataset>/"
                             "rel_transition_graph.pkl. Use this to "
                             "point at the empirical / random / "
                             "shuffled baselines produced by "
                             "build_baseline_graphs.py.")
    parser.add_argument("--rerank-mode", type=str, default="transition",
                        choices=["transition", "frequency", "recency", "none"],
                        help="Which re-ranking mechanism to apply at "
                             "inference time. 'transition' uses the "
                             "loaded TransitionGraph (default). "
                             "'frequency'/'recency' use "
                             "frequency_rerank.py's graph-free "
                             "baselines. 'none' disables re-ranking "
                             "entirely regardless of --rerank-alpha.")

    # NEW — per-query rank saving for significance testing
    parser.add_argument("--save-ranks",     action='store_true', default=False,
                        help="Save per-query filtered ranks to "
                             "../result/ranks/<tag>_ranks.npy for later "
                             "significance testing (significance_test.py).")
    parser.add_argument("--save-ranks-tag", type=str,   default=None,
                        help="Filename tag for --save-ranks output. "
                             "Defaults to '<dataset>_<rerank_mode>'.")

    # dropout
    parser.add_argument("--dropout",           type=float, default=0.2)
    parser.add_argument("--input-dropout",     type=float, default=0.2)
    parser.add_argument("--hidden-dropout",    type=float, default=0.2)
    parser.add_argument("--feat-dropout",      type=float, default=0.2)

    # training
    parser.add_argument("--n-epochs",          type=int,   default=500)
    parser.add_argument("--lr",                type=float, default=0.001)
    parser.add_argument("--grad-norm",         type=float, default=1.0)
    parser.add_argument("--batch-size",        type=int,   default=1)
    parser.add_argument("--evaluate-every",    type=int,   default=1)
    parser.add_argument("--train-history-len", type=int,   default=7)
    parser.add_argument("--test-history-len",  type=int,   default=7)
    parser.add_argument("--dilate-len",        type=int,   default=1)

    # early stopping
    parser.add_argument("--patience",          type=int,   default=5)
    parser.add_argument("--early-stop-min-epochs", type=int, default=0)

    # misc
    parser.add_argument("--run-statistic",     action='store_true', default=False)
    parser.add_argument("--add-rel-word",      action='store_true', default=False)
    parser.add_argument("--split-by-relation", action='store_true', default=False)

    args = parser.parse_args()
    args.test_history_len = args.train_history_len
    print(args)
    run_experiment(args)


