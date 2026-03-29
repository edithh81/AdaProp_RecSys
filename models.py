"""
AdaProp GNN model adapted for Recommendation.

Core change vs. the original transductive AdaProp:
  - The graph now contains *users* (0 … n_users-1) **and** *items/KG entities*
    (n_users … n_nodes-1).
  - Scoring is over **items only** (n_items dimensional output).
  - The last GNN layer disables node-sampling and instead filters the expanded
    node set to *item* nodes, so the final readout scores only items.
  - Relation embedding size accounts for the interact / inv-interact rels and
    the self-loop relation added by the data loader.

Everything else (Gumbel-softmax adaptive sampling, GRU gating, attention-based
message passing) is identical to the original AdaProp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


# ---------------------------------------------------------------------------
# PPR edge-pruning helpers
# ---------------------------------------------------------------------------
def _variadic_topk_mask(scores, counts, k):
    """Vectorised per-group top-k over a *flat* scores tensor.

    Parameters
    ----------
    scores : Tensor [N]          flat, groups arranged consecutively
    counts : LongTensor [G]      size of each group (sums to N)
    k      : int                 top-k budget per group

    Returns
    -------
    mask : BoolTensor [N]  True for the kept elements
    """
    n_groups = len(counts)
    max_count = counts.max().item()
    actual_k = min(k, max_count)

    # pad to [G, max_count] for batched topk
    padded = torch.full((n_groups, max_count), float('-inf'), device=scores.device)
    group_idx = torch.repeat_interleave(
        torch.arange(n_groups, device=scores.device), counts,
    )
    offsets = torch.zeros(n_groups, dtype=torch.long, device=scores.device)
    if n_groups > 1:
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)
    pos_in_group = torch.arange(len(scores), device=scores.device) - \
        torch.repeat_interleave(offsets, counts)
    padded[group_idx, pos_in_group] = scores

    _, topk_cols = torch.topk(padded, actual_k, dim=1)

    # discard positions beyond actual group length
    valid = topk_cols < counts.unsqueeze(1)
    flat_offsets = offsets.unsqueeze(1).expand_as(topk_cols)
    flat_idx = (topk_cols + flat_offsets)[valid]

    mask = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    mask[flat_idx] = True
    return mask


# ---------------------------------------------------------------------------
# GNN Layer  (one hop of message passing + optional adaptive node sampling)
# ---------------------------------------------------------------------------
class GNNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        attn_dim,
        n_rel,
        n_nodes,
        n_users,
        n_items,
        n_node_topk=-1,
        n_edge_topk=-1,
        tau=1.0,
        act=lambda x: x,
    ):
        super().__init__()
        self.n_rel       = n_rel
        self.n_nodes     = n_nodes
        self.n_users     = n_users
        self.n_items     = n_items
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.attn_dim    = attn_dim
        self.act         = act
        self.n_node_topk = n_node_topk
        self.n_edge_topk = n_edge_topk
        self.tau         = tau

        # +1 for the self-loop relation
        n_rel_total = 2 * n_rel + 1 + 2  # interact(0), inv-interact(1), KG rels shifted by 2, inverses, self-loop
        self.rela_embed = nn.Embedding(n_rel_total + 2, in_dim)   # extra padding

        self.Ws_attn  = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn  = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h      = nn.Linear(in_dim, out_dim, bias=False)
        self.W_samp   = nn.Linear(in_dim, 1, bias=False)

    # -- train / eval mode switch for the softmax variant --
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if self.training and self.tau > 0:
            self.softmax = lambda x: F.gumbel_softmax(x, tau=self.tau, hard=False)
        else:
            self.softmax = lambda x: F.softmax(x, dim=1)
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, q_sub, q_rel, hidden, edges, nodes, old_nodes_new_idx, batchsize):
        """
        Parameters
        ----------
        q_sub : LongTensor [B]          query-subject (user) ids
        q_rel : LongTensor [B]          query-relation ids
        hidden : Tensor [N_prev, dim]   hidden states of previous-layer nodes
        edges : LongTensor [E, 6]       (batch, head, rel, tail, head_idx, tail_idx)
        nodes : LongTensor [N_cur, 2]   (batch, node_id)
        old_nodes_new_idx : LongTensor  mapping from previous-layer to current indices
        batchsize : int

        Returns
        -------
        hidden_new, new_nodes, sampled_mask   (when sampling is active)
        hidden_new                            (when sampling is disabled, n_node_topk <= 0)
        """
        sub  = edges[:, 4]
        rel  = edges[:, 2]
        obj  = edges[:, 5]
        hs   = hidden[sub]
        hr   = self.rela_embed(rel)
        r_idx = edges[:, 0]
        h_qr  = self.rela_embed(q_rel)[r_idx]

        n_node  = nodes.shape[0]
        message = hs + hr

        # --- optional edge sampling ---
        if self.n_edge_topk > 0:
            alpha = self.w_alpha(
                nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
            ).squeeze(-1)
            edge_prob      = F.gumbel_softmax(alpha, tau=1, hard=False)
            topk_idx       = torch.argsort(edge_prob, descending=True)[:self.n_edge_topk]
            edge_prob_hard = torch.zeros_like(alpha)
            edge_prob_hard[topk_idx] = 1
            alpha *= (edge_prob_hard - edge_prob.detach() + edge_prob)
            alpha  = torch.sigmoid(alpha).unsqueeze(-1)
        else:
            alpha = torch.sigmoid(
                self.w_alpha(
                    nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))
                )
            )

        message     = alpha * message
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum')
        hidden_new  = self.act(self.W_h(message_agg))
        hidden_new  = hidden_new.clone()

        # ---------- no sampling ----------
        if self.n_node_topk <= 0:
            return hidden_new

        # ---------- adaptive node sampling ----------
        tmp_diff = torch.ones(n_node)
        tmp_diff[old_nodes_new_idx] = 0
        bool_diff = tmp_diff.bool()
        diff_node = nodes[bool_diff]

        diff_logit = self.W_samp(hidden_new[bool_diff]).squeeze(-1)

        node_scores = torch.ones((batchsize, self.n_nodes)).cuda() * float('-inf')
        node_scores[diff_node[:, 0], diff_node[:, 1]] = diff_logit

        node_scores = self.softmax(node_scores)
        topk_idx    = torch.topk(node_scores, self.n_node_topk, dim=1).indices.reshape(-1)
        topk_batch  = torch.arange(batchsize).repeat(self.n_node_topk, 1).T.reshape(-1)
        batch_topk  = torch.zeros((batchsize, self.n_nodes)).cuda()
        batch_topk[topk_batch, topk_idx] = 1

        bool_sampled_diff = batch_topk[diff_node[:, 0], diff_node[:, 1]].bool()
        bool_same = ~bool_diff.cuda()
        bool_same[bool_diff] = bool_sampled_diff

        # straight-through gradient for the hard selection
        prob_hard = batch_topk[diff_node[:, 0], diff_node[:, 1]]
        prob_soft = node_scores[diff_node[:, 0], diff_node[:, 1]]
        hidden_new[bool_diff] *= (prob_hard - prob_soft.detach() + prob_soft).unsqueeze(-1)

        new_nodes  = nodes[bool_same]
        hidden_new = hidden_new[bool_same]

        return hidden_new, new_nodes, bool_same


# ---------------------------------------------------------------------------
# Full model  (stacked GNN layers + readout)
# ---------------------------------------------------------------------------
class AdaPropRecSys(nn.Module):
    def __init__(self, params, loader):
        super().__init__()
        self.n_layer     = params.n_layer
        self.hidden_dim  = params.hidden_dim
        self.attn_dim    = params.attn_dim
        self.n_rel       = params.n_rel
        self.n_nodes     = params.n_nodes
        self.n_users     = params.n_users
        self.n_items     = params.n_items
        self.n_node_topk = params.n_node_topk   # list[int] per layer
        self.n_edge_topk = params.n_edge_topk
        self.loader      = loader

        # PPR edge-pruning budget (0 = disabled)
        self.ppr_k = getattr(params, 'ppr_topk', 0)

        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act  = acts[params.act]

        self.gnn_layers = nn.ModuleList()
        for i in range(self.n_layer):
            i_topk = (self.n_node_topk
                       if isinstance(self.n_node_topk, int)
                       else self.n_node_topk[i])
            # Disable sampling at the *last* layer so all expanded item-nodes
            # survive to the readout.  Intermediate layers use adaptive sampling.
            if i == self.n_layer - 1:
                i_topk = -1
            self.gnn_layers.append(
                GNNLayer(
                    self.hidden_dim, self.hidden_dim, self.attn_dim,
                    self.n_rel, self.n_nodes, self.n_users, self.n_items,
                    n_node_topk=i_topk, n_edge_topk=self.n_edge_topk,
                    tau=params.tau, act=act,
                )
            )

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate    = nn.GRU(self.hidden_dim, self.hidden_dim)

    # ---- PPR helpers ----------------------------------------------------
    def set_ppr(self, ppr_indices, ppr_values):
        """Register cached top-k PPR data as model buffers (auto-move to CUDA)."""
        self.register_buffer('ppr_indices', ppr_indices.long())
        self.register_buffer('ppr_values', ppr_values.float())

    def _lookup_ppr(self, q_users, tail_nodes):
        """Sparse lookup: PPR[user, node] using cached top-k per user.

        Returns 0 for (user, node) pairs not in the cached top-k.
        """
        user_topk_nodes = self.ppr_indices[q_users]   # [E, topk]
        user_topk_vals  = self.ppr_values[q_users]     # [E, topk]
        match = (user_topk_nodes == tail_nodes.unsqueeze(1))  # [E, topk]
        return (user_topk_vals * match.float()).sum(dim=1)     # [E]

    def _ppr_prune_edges(self, q_sub, edges, nodes, old_nodes_new_idx):
        """Prune edges with PPR: each source node keeps top-k neighbours.

        Self-loop edges are never pruned.
        """
        self_rel_id = 2 * self.n_rel + 2

        is_selfloop = (edges[:, 2] == self_rel_id)
        nsl_idx = torch.where(~is_selfloop)[0]
        sl_idx  = torch.where(is_selfloop)[0]

        if nsl_idx.numel() == 0:
            return edges, nodes, old_nodes_new_idx

        nsl_edges = edges[nsl_idx]

        # PPR score for every non-self-loop edge
        q_users   = q_sub[nsl_edges[:, 0]]
        tails     = nsl_edges[:, 3]
        ppr_scores = self._lookup_ppr(q_users, tails)

        # group by (batch_idx, head_node) — one group per source node
        group_keys = nsl_edges[:, 0] * self.n_nodes + nsl_edges[:, 1]
        sort_perm  = torch.argsort(group_keys)
        sorted_keys   = group_keys[sort_perm]
        sorted_scores = ppr_scores[sort_perm]

        _, counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        keep_mask = _variadic_topk_mask(sorted_scores, counts, self.ppr_k)

        # map kept positions back to original edge ordering
        kept_nsl_original = nsl_idx[sort_perm[keep_mask]]

        # recombine with self-loops
        kept_idx = torch.cat([kept_nsl_original, sl_idx]).sort().values
        edges = edges[kept_idx]

        # recompute tail nodes & tail index
        new_tail_nodes, new_tail_index = torch.unique(
            edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True,
        )
        edges = torch.cat([edges[:, :5], new_tail_index.unsqueeze(1)], dim=1)

        # recompute old_nodes_new_idx via self-loop edges
        idd_mask = (edges[:, 2] == self_rel_id)
        head_idx_sl = edges[idd_mask, 4]
        tail_idx_sl = edges[idd_mask, 5]
        _, sort_old = head_idx_sl.sort()
        new_old_nodes_new_idx = tail_idx_sl[sort_old]

        return edges, new_tail_nodes, new_old_nodes_new_idx

    def updateTopkNums(self, topk_list):
        assert len(topk_list) == self.n_layer
        for idx in range(self.n_layer):
            # keep last-layer topk=-1 (no sampling)
            if idx == self.n_layer - 1:
                self.gnn_layers[idx].n_node_topk = -1
            else:
                self.gnn_layers[idx].n_node_topk = topk_list[idx]

    # -----------------------------------------------------------
    def forward(self, subs, rels, mode='train'):
        n     = len(subs)
        q_sub = torch.LongTensor(subs).cuda()
        q_rel = torch.LongTensor(rels).cuda()
        h0    = torch.zeros((1, n, self.hidden_dim)).cuda()
        nodes = torch.cat([torch.arange(n).unsqueeze(1).cuda(),
                           q_sub.unsqueeze(1)], dim=1)
        hidden = torch.zeros(n, self.hidden_dim).cuda()

        for i in range(self.n_layer):
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(
                nodes.data.cpu().numpy(), n, mode=mode,
            )

            # PPR edge pruning for middle layers (layer 1 … n_layer-2)
            if self.ppr_k > 0 and 0 < i < self.n_layer - 1:
                edges, nodes, old_nodes_new_idx = self._ppr_prune_edges(
                    q_sub, edges, nodes, old_nodes_new_idx,
                )

            n_node = nodes.size(0)

            layer_out = self.gnn_layers[i](
                q_sub, q_rel, hidden, edges, nodes,
                old_nodes_new_idx, n,
            )

            if isinstance(layer_out, tuple):
                # sampling was active → (hidden, new_nodes, mask)
                hidden, nodes, sampled_mask = layer_out
            else:
                # no sampling (last layer) → keep all nodes
                hidden = layer_out
                sampled_mask = torch.ones(n_node, dtype=torch.bool).cuda()

            # re-index h0 to match current node set
            h0 = torch.zeros(1, n_node, hidden.size(1)).cuda() \
                      .index_copy_(1, old_nodes_new_idx, h0)
            h0 = h0[0, sampled_mask, :].unsqueeze(0)

            hidden     = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden     = hidden.squeeze(0)

        # ---------- readout: score items only ----------
        scores = self.W_final(hidden).squeeze(-1)

        # filter to item nodes  (user-ids in [0, n_users), items in [n_users, n_users+n_items))
        item_mask = (nodes[:, 1] >= self.n_users) & (nodes[:, 1] < self.n_users + self.n_items)

        scores_all = torch.zeros((n, self.n_items)).cuda()
        if item_mask.any():
            item_nodes  = nodes[item_mask]
            item_scores = scores[item_mask]
            scores_all[item_nodes[:, 0], item_nodes[:, 1] - self.n_users] = item_scores

        return scores_all
