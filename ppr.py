"""
Personalized PageRank (PPR) computation for edge pruning.

Mirrors KUCNet's PPR approach:
  - For each user, compute PPR scores over all nodes in the KG.
  - PPR teleport vector is biased toward the user's known items.
  - Only top-k scores per node are retained (sparse), saving memory and disk.
  - Results are cached to disk so they only need to be computed once per dataset/topk.
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm


def compute_ppr(loader, alpha=0.85, beta=0.8, n_iter=20, batch_size=128):
    """Compute per-user PPR scores over the full knowledge graph.

    Returns the full dense [n_users, n_nodes] tensor (before top-k truncation).
    """
    tkg = torch.LongTensor(loader.tKG).cuda()
    n_nodes = loader.n_nodes
    n_users = loader.n_users

    # degree (out-degree per node)
    uni, count = torch.unique(tkg[:, 0], return_counts=True)
    id_c = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)]).cuda()
    val_c = torch.zeros(n_nodes).cuda()
    val_c[uni] = 1.0 / count.float()
    cnt = torch.sparse_coo_tensor(id_c, val_c, (n_nodes, n_nodes)).cuda()

    # adjacency (head -> tail)
    index = torch.stack([tkg[:, 0], tkg[:, 2]]).cuda()
    value = torch.ones(len(tkg)).cuda()
    Mkg = torch.sparse_coo_tensor(index, value, (n_nodes, n_nodes)).cuda()

    # M = Mkg * diag(1/degree) -> row-normalized transition
    M = torch.sparse.mm(Mkg, cnt).cuda()

    print('PPR: transition matrix ready, starting power iteration ...')
    s_time = time.time()

    n_batch = n_users // batch_size + (n_users % batch_size > 0)
    final_rank = torch.zeros(n_nodes, n_users)

    for i in tqdm(range(n_batch), desc='PPR'):
        start = i * batch_size
        tbs = min(batch_size, n_users - start)
        u_list = torch.arange(start, start + tbs)

        # initial rank: one-hot on user nodes
        u_index = torch.stack([u_list.cuda(), torch.arange(tbs).cuda()])
        u_value = torch.ones(tbs).cuda()
        rank = torch.sparse_coo_tensor(u_index, u_value, (n_nodes, tbs)).cuda()

        # preference / teleport vector P (biased toward known items)
        p_indices_list = []
        p_values_list = []
        for j in range(tbs):
            uid = u_list[j].item()
            known = loader.known_user_set.get(uid, [])
            n_known = len(known)

            node_ids = torch.arange(n_nodes)
            col_ids = torch.full((n_nodes,), j, dtype=torch.long)

            vals = torch.full((n_nodes,), (1 - beta) / max(n_nodes - n_known, 1))
            if n_known > 0:
                known_t = torch.LongTensor(known)
                vals[known_t] = beta / n_known

            p_indices_list.append(torch.stack([node_ids, col_ids]))
            p_values_list.append(vals)

        p_index = torch.cat(p_indices_list, dim=1).cuda()
        p_value = torch.cat(p_values_list).cuda()
        P = torch.sparse_coo_tensor(p_index, p_value, (n_nodes, tbs)).coalesce().cuda()

        # power iteration
        for _ in range(n_iter):
            rank = (1 - alpha) * P + alpha * torch.sparse.mm(M, rank)

        final_rank[:, start:start + tbs] = rank.to_dense().cpu()

    ppr = final_rank.T  # [n_users, n_nodes]
    print(f'PPR done. time: {time.time() - s_time:.1f}s')
    return ppr


def truncate_topk(ppr, topk):
    """Keep only top-k scores per user (row).

    Parameters
    ----------
    ppr  : Tensor [n_users, n_nodes]  dense
    topk : int

    Returns
    -------
    topk_indices : LongTensor [n_users, topk]   node ids
    topk_values  : Tensor     [n_users, topk]   PPR scores
    """
    k = min(topk, ppr.shape[1])
    values, indices = torch.topk(ppr, k, dim=1)  # both [n_users, k]
    return indices, values


def build_lookup(topk_indices, topk_values, n_nodes):
    """Build a fast lookup dict: ppr_lookup[user] -> {node_id: score}.

    This is compact: each user stores at most topk entries.
    """
    n_users = topk_indices.shape[0]
    lookup = {}
    for u in range(n_users):
        idx = topk_indices[u]
        val = topk_values[u]
        lookup[u] = dict(zip(idx.tolist(), val.tolist()))
    return lookup


def get_ppr_cached(loader, topk, cache_dir=None):
    """Load top-k PPR from cache, or compute, truncate, and save.

    Cache file: ``<cache_dir>/ppr_topk{topk}.pt``
    Stores only (topk_indices, topk_values) — sparse per-user top-k.

    Returns
    -------
    topk_indices : LongTensor [n_users, topk]
    topk_values  : Tensor     [n_users, topk]
    """
    if cache_dir is None:
        cache_dir = os.path.join(loader.task_dir, 'ppr_cache')
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f'ppr_topk{topk}.pt')
    if os.path.exists(cache_path):
        print(f'PPR: loading cached top-{topk} scores from {cache_path}')
        data = torch.load(cache_path, map_location='cpu')
        ti, tv = data['topk_indices'], data['topk_values']
        if ti.shape[0] == loader.n_users and ti.shape[1] == min(topk, loader.n_nodes):
            return ti, tv
        print('PPR: cache shape mismatch, recomputing ...')

    ppr = compute_ppr(loader)
    topk_indices, topk_values = truncate_topk(ppr, topk)

    torch.save({'topk_indices': topk_indices, 'topk_values': topk_values}, cache_path)
    print(f'PPR: cached top-{topk} to {cache_path}  '
          f'(size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)')
    return topk_indices, topk_values
