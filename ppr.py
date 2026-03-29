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


def compute_ppr(loader, topk, alpha=0.85, beta=0.8, n_iter=20, batch_size=128):
    """Compute per-user PPR scores on GPU, keep top-k per user, return CPU tensors.

    The entire pipeline (transition matrix, power iteration, top-k truncation)
    runs on GPU.  Only the final sparse top-k results are moved to CPU.

    Returns
    -------
    topk_indices : LongTensor [n_users, topk]  (CPU)  node ids
    topk_values  : Tensor     [n_users, topk]  (CPU)  PPR scores
    """
    device = torch.device('cuda')

    tkg = torch.LongTensor(loader.tKG).to(device)
    n_nodes = loader.n_nodes
    n_users = loader.n_users
    k = min(topk, n_nodes)

    # --- build row-normalized transition matrix M on GPU ---
    uni, count = torch.unique(tkg[:, 0], return_counts=True)
    id_c = torch.stack([torch.arange(n_nodes, device=device),
                        torch.arange(n_nodes, device=device)])
    val_c = torch.zeros(n_nodes, device=device)
    val_c[uni] = 1.0 / count.float()
    cnt = torch.sparse_coo_tensor(id_c, val_c, (n_nodes, n_nodes), device=device)

    index = torch.stack([tkg[:, 0], tkg[:, 2]])
    value = torch.ones(len(tkg), device=device)
    Mkg = torch.sparse_coo_tensor(index, value, (n_nodes, n_nodes), device=device)

    M = torch.sparse.mm(Mkg, cnt)

    print('PPR: transition matrix ready (GPU), starting power iteration ...')
    s_time = time.time()

    n_batch = n_users // batch_size + (n_users % batch_size > 0)
    # accumulate sparse top-k results on CPU
    all_indices = torch.zeros(n_users, k, dtype=torch.long)
    all_values  = torch.zeros(n_users, k)

    for i in tqdm(range(n_batch), desc='PPR (GPU)'):
        start = i * batch_size
        tbs = min(batch_size, n_users - start)
        u_list = torch.arange(start, start + tbs, device=device)

        # initial rank: one-hot on user nodes (GPU)
        u_index = torch.stack([u_list, torch.arange(tbs, device=device)])
        u_value = torch.ones(tbs, device=device)
        rank = torch.sparse_coo_tensor(u_index, u_value, (n_nodes, tbs),
                                       device=device)

        # preference / teleport vector P on GPU
        node_ids = torch.arange(n_nodes, device=device)
        p_indices_list = []
        p_values_list = []
        for j in range(tbs):
            uid = (start + j)
            known = loader.known_user_set.get(uid, [])
            n_known = len(known)

            col_ids = torch.full((n_nodes,), j, dtype=torch.long, device=device)

            vals = torch.full((n_nodes,), (1 - beta) / max(n_nodes - n_known, 1),
                              device=device)
            if n_known > 0:
                known_t = torch.LongTensor(known).to(device)
                vals[known_t] = beta / n_known

            p_indices_list.append(torch.stack([node_ids, col_ids]))
            p_values_list.append(vals)

        p_index = torch.cat(p_indices_list, dim=1)
        p_value = torch.cat(p_values_list)
        P = torch.sparse_coo_tensor(p_index, p_value, (n_nodes, tbs),
                                    device=device).coalesce()

        # power iteration (all on GPU)
        for _ in range(n_iter):
            rank = (1 - alpha) * P + alpha * torch.sparse.mm(M, rank)

        # top-k per user on GPU, then move to CPU
        rank_dense = rank.to_dense().T  # [tbs, n_nodes], GPU
        batch_vals, batch_idx = torch.topk(rank_dense, k, dim=1)
        all_indices[start:start + tbs] = batch_idx.cpu()
        all_values[start:start + tbs]  = batch_vals.cpu()

    print(f'PPR done (GPU). time: {time.time() - s_time:.1f}s')
    return all_indices, all_values


def get_ppr_cached(loader, topk, cache_dir=None):
    """Load top-k PPR from cache, or compute on GPU and save CPU tensors.

    Cache file: ``<cache_dir>/ppr_topk{topk}.pt``
    Stores only (topk_indices, topk_values) on CPU — sparse per-user top-k.

    Returns
    -------
    topk_indices : LongTensor [n_users, topk]  (CPU)
    topk_values  : Tensor     [n_users, topk]  (CPU)
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

    topk_indices, topk_values = compute_ppr(loader, topk=topk)

    torch.save({'topk_indices': topk_indices, 'topk_values': topk_values}, cache_path)
    print(f'PPR: cached top-{topk} (CPU) to {cache_path}  '
          f'(size: {os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)')
    return topk_indices, topk_values
