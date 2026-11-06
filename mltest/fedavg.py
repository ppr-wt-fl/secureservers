"""FedAvg on poisoning-eval's AdFlush setup: N_CLIENTS clients on the iid partition, all of them every round, no attacks.

data/ holds poisoning-eval's caches (robustness-evaluation/poisoning-eval/.train.parquet, .test.parquet, .model.pt):
the site-wise 70/30 split with the train rows already shuffled, and the Keras model converted to torch layer specs.
Run: uv run --python 3.13 --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    --with-requirements requirements.txt python fedavg.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = Path(__file__).with_name('data')
N_CLIENTS = 200
N_ROUNDS = 50
LEARNING_RATE = 0.1
# per-batch gradient clipping, only a guard against divergence (some standardized inputs reach |z| > 1000)
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 64
# every client takes exactly this many steps per round: 2 epochs of an iid client (1687 rows -> 27 batches of 64)
LOCAL_STEPS = 54
SEED = 0

FEATURE_COLUMNS = [
    "content_policy_type",
    "url_length",
    "brackettodot",
    "is_third_party",
    # FIXME (from poisoning-eval): model expects keyword_char_present (normalization stats match it, raw is a superset)
    "keyword_raw_present",
    "num_get_storage",
    "num_set_storage",
    "num_get_cookie",
    # FIXME (from poisoning-eval): model expects num_requests_sent, stats match num_requests_sent_webgraph
    "num_requests_sent_static",
    "req_url_33",
    "req_url_135",
    "req_url_179",
    "fqdn_4",
    "fqdn_13",
    "fqdn_14",
    "fqdn_15",
    "fqdn_23",
    "fqdn_26",
    "fqdn_27",
    "ng_0_0_2",
    "ng_0_15_15",
    "ng_2_13_2",
    "ng_15_0_3",
    "ng_15_0_15",
    "ng_15_15_15",
    "avg_ident",
    "avg_charperline",
]

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def load(name):
    dataframe = pd.read_parquet(DATA / f'{name}.parquet')
    return (torch.from_numpy(dataframe[FEATURE_COLUMNS].to_numpy(np.float32)),
            torch.from_numpy(dataframe['CLASS'].to_numpy(np.float32)))


train_features, train_labels = load('train')
test_features, test_labels = load('test')
# what flwr's IidPartitioner does: contiguous, near-equal shards of the (already shuffled) train rows
partitions = np.array_split(np.arange(len(train_labels)), N_CLIENTS)


# Keras Rescaling: x * scale + offset (the standardization AutoKeras' Normalization was swapped for)
class Rescale(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

    def forward(self, x):
        return x * self.scale + self.offset


def build_model(specs):
    """poisoning-eval's layer specs to torch; the final sigmoid is dropped, the model outputs logits."""
    layers = []
    for kind, *values in specs:
        if kind == 'rescale':
            layers.append(Rescale(*values))
        elif kind == 'linear':
            weight, bias = values
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data, linear.bias.data = weight, bias
            layers.append(linear)
        elif kind == 'relu':
            layers.append(nn.ReLU())
        elif kind == 'dropout':
            layers.append(nn.Dropout(values[0]))
    return nn.Sequential(*layers)


model = build_model(torch.load(DATA / 'model.pt')['layers'])
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


def get_flat():
    """The shared weights (Dense kernels and biases, not the Rescale buffers) as one vector."""
    return torch.cat([p.detach().flatten() for p in model.parameters()])


@torch.no_grad()
def set_flat(flat):
    for p, chunk in zip(model.parameters(), flat.split([p.numel() for p in model.parameters()])):
        p.copy_(chunk.view_as(p))


def local_train(global_weights, rows):
    """LOCAL_STEPS batches of SGD from the global weights over shuffles of the client's rows; returns the update."""
    set_flat(global_weights)
    model.train()
    n_needed = LOCAL_STEPS * BATCH_SIZE
    order = np.concatenate([rng.permutation(rows) for _ in range(-(-n_needed // len(rows)))])[:n_needed]
    for batch in torch.from_numpy(order).split(BATCH_SIZE):
        optimizer.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(train_features[batch])[:, 0],
                                                       train_labels[batch]).backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
    update = get_flat() - global_weights
    assert torch.isfinite(update).all(), 'local training diverged'
    return update


@torch.no_grad()
def evaluate(weights):
    set_flat(weights)
    model.eval()
    predictions = torch.cat([model(chunk)[:, 0] for chunk in test_features.split(65536)]) > 0
    labels = test_labels.bool()
    tp = (predictions & labels).sum().item()
    fp = (predictions & ~labels).sum().item()
    fn = (~predictions & labels).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {'accuracy': (predictions == labels).float().mean().item(), 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def fmt(metrics):
    return ' '.join(f'{name} {value:.4f}' for name, value in metrics.items())


if __name__ == '__main__':
    weights = get_flat()
    print(f'{len(weights)} weights, {N_CLIENTS} clients of {len(partitions[0])} rows')
    print(f'round 0: {fmt(evaluate(weights))}', flush=True)
    for round_number in range(1, N_ROUNDS + 1):
        total = torch.zeros_like(weights)
        for rows in partitions:
            total += local_train(weights, rows)
        weights = weights + total / N_CLIENTS
        print(f'round {round_number}: {fmt(evaluate(weights))}', flush=True)
