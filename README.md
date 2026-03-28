# AdaProp for Recommendation

Adaptation of AdaProp (Adaptive Propagation) from KG completion to recommendation tasks.

## What Is AdaProp

AdaProp is a GNN-based reasoning model that uses **Gumbel-softmax adaptive node sampling** to learn which entities to explore at each propagation step, instead of using fixed top-K or random sampling. This makes reasoning efficient and query-aware.

## Adaptation from KG Completion to Recommendation

### Key Changes

| Aspect | KG Completion (Original) | Recommendation (This) |
|--------|--------------------------|----------------------|
| **Task** | Predict missing entity in `(h, r, ?)` | Predict items a user will interact with |
| **Graph** | KG entities only `[0, n_ent)` | Users `[0, n_users)` + KG entities `[n_users, n_nodes)` |
| **Relations** | `2*n_rel + 1` (original + inverse + self-loop) | `2*n_rel + 3` (+ interact, inv-interact) |
| **Loss** | Ranking loss (softmax CE over all entities) | BPR loss: `-log(σ(pos - neg))` |
| **Output** | Scores for all entities `(batch, n_ent)` | Scores for items only `(batch, n_items)` |
| **Metrics** | MRR, Hit@1, Hit@10 | Recall@20, NDCG@20 |

### What Was Preserved

- Gumbel-softmax adaptive node sampling (intermediate GNN layers)
- Attention-based message passing (`α = σ(W·ReLU(Ws·hs + Wr·hr + Wqr·hqr))`)
- GRU gating across layers
- Straight-through gradient estimator for hard node selection
- Last layer disables sampling so all item nodes survive to readout

### Relation Mapping

```
0           : interact     (user → item)
1           : inv-interact (item → user)
2 ... n_rel+1    : original KG relations (shifted +2)
n_rel+2 ... 2*n_rel+1 : inverse KG relations
2*n_rel+2   : self-loop
```

### Entity Mapping

```
[0, n_users)                    : user nodes
[n_users, n_users + n_items)    : item nodes (subset of KG entities)
[n_users, n_users + n_ent)      : all KG entities (items + non-item entities)
```

## File Structure

```
AdaProp_RecSys/
├── train.py        # Entry point, per-dataset hyperparameter presets
├── base_model.py   # Training loop (BPR loss), evaluation (Recall@K, NDCG@K)
├── models.py       # AdaPropRecSys model: GNNLayer + adaptive sampling + GRU + item readout
├── load_data.py    # Data loader: user-item CF + KG → unified graph, negative sampling
├── utils.py        # BPR loss, NDCG/DCG computation
└── data/           # Symlink to datasets (last-fm, amazon-book, alibaba-fashion, ...)
```

### File Details

- **`train.py`** — CLI entry point. Contains per-dataset hyperparameter presets (lr, decay, batch size, etc.) for `last-fm`, `amazon-book`, `alibaba-fashion`, `Dis_5fold_*`, `new_*` variants. Key AdaProp args: `--topk` (sampling budget per layer), `--tau` (Gumbel temperature), `--layers`.

- **`models.py`** — `AdaPropRecSys` model. Stacks `GNNLayer` modules. Each layer does attention-based message passing + optional Gumbel-softmax node sampling. Last layer disables sampling. Final readout filters to item nodes only, producing `(batch, n_items)` scores.

- **`base_model.py`** — Training harness. `train_batch()` uses BPR loss. `test_batch()` evaluates Recall@20 and NDCG@20 by ranking uninteracted items per user.

- **`load_data.py`** — Builds unified user+entity graph. Reads `train.txt`/`test.txt` (user-item format: `user_id item1 item2 ...`) and `kg.txt` (numeric triples). Creates interact/inv-interact edges. Splits CF into facts (6/7) and train (1/7). Provides `get_neighbors()` via sparse matrix multiplication.

- **`utils.py`** — `cal_bpr_loss(n_users, pos, neg, scores)` for BPR training. `ndcg_k(r, k, len_pos)` and `dcg_k(r, k)` for evaluation.

## Dependencies

```
torch>=2.0          # with CUDA
torch_scatter        # must match torch+CUDA version
scipy
numpy
tqdm
```

### Install (torch 2.8.0 + CUDA 12.8)

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install scipy numpy tqdm
```

## Usage

### Training

```bash
# last-fm
python train.py --data_path data/last-fm/ --topk 50 --layers 3 --gpu 0

# amazon-book
python train.py --data_path data/amazon-book/ --topk 120 --layers 3 --gpu 0

# alibaba-fashion
python train.py --data_path data/alibaba-fashion/ --topk 70 --layers 5 --gpu 0
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `data/last-fm/` | Path to dataset |
| `--topk` | 50 | Node sampling budget per GNN layer |
| `--layers` | 3 | Number of GNN propagation layers |
| `--tau` | 1.0 | Gumbel-softmax temperature |
| `--epoch` | 40 | Training epochs |
| `--gpu` | 0 | GPU device ID |

## Data Format

Each dataset folder should contain:

```
dataset/
├── train.txt       # user_id item1 item2 item3 ...  (one line per user)
├── test.txt        # same format
└── kg.txt          # head_entity_id relation_id tail_entity_id  (one triple per line)
```

## Evaluation

After each training epoch, the model evaluates on the test set:
- **Recall@20**: Fraction of ground-truth items appearing in top-20 predictions
- **NDCG@20**: Position-aware ranking quality (higher = relevant items ranked higher)

Results are logged to `results/<dataset>_perf.txt`.
