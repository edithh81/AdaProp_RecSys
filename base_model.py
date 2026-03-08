"""
Training / evaluation harness for AdaProp on RecSys datasets.

Training uses BPR loss (like KUCNet).
Evaluation uses Recall@K and NDCG@K (K=20 by default).
"""

import torch
import numpy as np
import time
import os
import heapq
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from models import AdaPropRecSys
from utils import cal_bpr_loss, ndcg_k


class BaseModel:
    def __init__(self, args, loader):
        self.model  = AdaPropRecSys(args, loader)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.loader   = loader
        self.n_ent    = loader.n_ent
        self.n_rel    = loader.n_rel
        self.n_users  = loader.n_users
        self.n_items  = loader.n_items
        self.n_nodes  = loader.n_nodes
        self.n_batch  = args.n_batch
        self.n_tbatch = args.n_tbatch
        self.n_train  = loader.n_train
        self.n_test   = loader.n_test
        self.n_layer  = args.n_layer
        self.args     = args

        self.known_user_set = loader.known_user_set
        self.test_user_set  = loader.test_user_set

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)

        if getattr(args, 'scheduler', 'exp') == 'exp':
            self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        else:
            self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)

        self.t_time = 0
        self.lastSavePath = None

        # build a human-readable model name
        self.modelName = f'{args.n_layer}-layers'
        for i in range(args.n_layer):
            topk_i = (args.n_node_topk
                       if isinstance(args.n_node_topk, int)
                       else args.n_node_topk[i])
            self.modelName += f'-{topk_i}'
        print(f'==> model: {self.modelName}')

    # ------------------------------------------------------------------
    # checkpoint helpers
    # ------------------------------------------------------------------
    def saveModel(self, tag, delete_last=True):
        save_dir = os.path.join(self.loader.task_dir, 'saveModel')
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{self.modelName}-{tag}.pt')
        print(f'Save checkpoint → {path}')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tag': tag,
        }, path)
        if delete_last and self.lastSavePath and os.path.exists(self.lastSavePath):
            print(f'Remove old checkpoint: {self.lastSavePath}')
            os.remove(self.lastSavePath)
        self.lastSavePath = path

    def loadModel(self, path):
        print(f'Load weights from {path}')
        assert os.path.exists(path), f'{path} not found'
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def _update(self):
        """Re-create optimizer (e.g. after loading weights with different lr)."""
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr,
                              weight_decay=self.args.lamb)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def train_batch(self):
        epoch_loss = 0
        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)

        t0 = time.time()
        self.model.train()

        for i in tqdm(range(n_batch), desc='train'):
            start = i * batch_size
            end   = min(self.loader.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)

            subs, rels, pos, neg = self.loader.get_batch(batch_idx, data='train')

            self.model.zero_grad()
            scores = self.model(subs, rels, mode='train')

            loss = cal_bpr_loss(self.n_users, pos, neg, scores)
            loss.backward()
            self.optimizer.step()

            # NaN guard
            for p in self.model.parameters():
                X = p.data.clone()
                mask = X != X
                X[mask] = np.random.random()
                p.data.copy_(X)

            epoch_loss += loss.item()

            if i % 500 == 0:
                print(f'  batch {i}  loss={loss.item():.4f}')

        self.t_time += time.time() - t0

        if getattr(self.args, 'scheduler', 'exp') == 'exp':
            self.scheduler.step()

        self.loader.shuffle_train()
        print(f'epoch_loss = {epoch_loss:.4f}')

        # run evaluation after each epoch
        recall, ndcg, out_str = self.test_batch()
        print(out_str)
        return recall, ndcg, out_str

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def test_one_user(self, u, score, K=20):
        try:
            training_items = self.known_user_set[u]
        except KeyError:
            training_items = []
        user_pos_test = self.test_user_set[u]

        all_items = set(range(self.n_users, self.n_users + self.n_items))
        test_items = list(all_items - set(training_items))

        item_score = {}
        for it in test_items:
            item_score[it] = score[it - self.n_users]

        top_items = heapq.nlargest(K, item_score, key=item_score.get)

        r = [1 if it in user_pos_test else 0 for it in top_items]
        ndcg = ndcg_k(r, K, len(user_pos_test))
        recall = np.sum(r) / max(len(user_pos_test), 1)
        return recall, ndcg

    def test_batch(self, K=20):
        batch_size = self.n_tbatch
        n_data  = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        self.model.eval()
        t0 = time.time()
        recall_sum, ndcg_sum = 0.0, 0.0

        with torch.no_grad():
            for bid in tqdm(range(n_batch), desc='eval'):
                start = bid * batch_size
                end   = min(n_data, (bid + 1) * batch_size)
                batch_idx = np.arange(start, end)

                subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
                scores = self.model(subs, rels, mode='test').data.cpu().numpy()

                for j in range(len(subs)):
                    u = subs[j]
                    one_r, one_n = self.test_one_user(u, scores[j], K=K)
                    recall_sum += one_r
                    ndcg_sum   += one_n

                if bid % 500 == 0:
                    print(f'  eval batch {bid}  recall(batch)='
                          f'{recall_sum / max((bid + 1) * batch_size, 1):.4f}')

        recall = recall_sum / n_data
        ndcg   = ndcg_sum   / n_data
        i_time = time.time() - t0
        out_str = (f'[TEST] Recall@{K}: {recall:.4f}  NDCG@{K}: {ndcg:.4f}  '
                   f'[TIME] train: {self.t_time:.2f}s  eval: {i_time:.2f}s\n')
        return recall, ndcg, out_str
