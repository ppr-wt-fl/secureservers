"""Numbers and figure for the thesis from compare.py's results (one JSON line per round).

Usage: python analyze.py [results/n200-a80.jsonl] [--plots DIR]
Prints the numbers (redirect to results/analysis.txt) and writes DIR/secure_cost_vs_clipped.{pdf,png}.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 1e-3  # compare.py's mismatch tolerance on max_abs_diff

parser = argparse.ArgumentParser()
parser.add_argument('results', nargs='?', default=os.path.join(HERE, 'results', 'n200-a80.jsonl'))
parser.add_argument('--plots', default=os.path.join(HERE, 'results', 'plots'))
arguments = parser.parse_args()

with open(arguments.results) as f:
    d = pd.DataFrame([json.loads(line) for line in f])
for side in ('plain', 'secure'):
    for metric in ('accuracy', 'precision', 'recall', 'f1'):
        d[f'{metric}_{side}'] = d[side].map(lambda m: m[metric])
d['counts_differ'] = (d.benign_plain != d.benign_secure) | (d.clipped_plain != d.clipped_secure)
# compare.py's own verdict must follow from the fields
assert (d.mismatch == (d.counts_differ | (d.max_abs_diff > TOLERANCE))).all()
assert (d['round'] == np.arange(1, len(d) + 1)).all()
n, attackers, p = d.n[0], d.attackers[0], d.p[0]
later = d[d['round'] > 1]  # round 1 includes the compile and has no m_h yet


def stats(s):
    return f'median {s.median():.4g}, min {s.min():.4g}, max {s.max():.4g}'


print(f'== setting: {len(d)} rounds, {n} clients, {attackers} attackers, P = {p:,}')

print('\n== correctness against the oracle')
print(f'rounds that differ: {d.mismatch.sum()} of {len(d)}: {d["round"][d.mismatch].tolist()}')
print(f'  of those, counts differ: {d.counts_differ.sum()}, max_abs_diff > {TOLERANCE}: {(d.max_abs_diff > TOLERANCE).sum()}')
for _, r in d[d.mismatch].iterrows():
    print(f'  round {r["round"]}: benign {r.benign_plain}/{r.benign_secure}, clipped {r.clipped_plain}/{r.clipped_secure}'
          f' (plain/secure), max_abs_diff {r.max_abs_diff:.3g}')
print(f'max_abs_diff: {stats(d.max_abs_diff)}')
for metric in ('accuracy', 'precision', 'recall', 'f1'):
    diff = (d[f'{metric}_secure'] - d[f'{metric}_plain']).abs()
    print(f'|{metric} secure - plain|: max {diff.max():.3g}, rounds with a difference {(diff > 0).sum()}')
for metric in ('accuracy', 'precision', 'recall', 'f1'):
    print(f'{metric}: round 1 {d[f"{metric}_secure"].iloc[0]:.4f}, round {len(d)} secure {d[f"{metric}_secure"].iloc[-1]:.4f}'
          f' / plain {d[f"{metric}_plain"].iloc[-1]:.4f}')

print('\n== filter')
print(f'benign (secure): {stats(d.benign_secure)}; attackers accepted (oracle): {stats(d.attackers_accepted)}')
print(f'clipped (secure): round 1 {d.clipped_secure[0]}, rounds 2+ {stats(later.clipped_secure)}')

print('\n== server cost per round')
for col in ('server_round_s', 'party_wall_s', 'mpc_s', 'glue_s', 'sent_mb', 'global_sent_mb', 'mpc_rounds'):
    print(f'{col}: round 1 {d[col][0]:.4g}, rounds 2+ {stats(later[col])}')
print(f'compile_s: round 1 {d.compile_s[0]:.3g}, rounds 2+ max {later.compile_s.max():.3g} (cached)')
print(f'mpc_s share of server_round_s: {stats(d.mpc_s / d.server_round_s)}')
print(f'plain_aggregate_s: {stats(d.plain_aggregate_s)}; secure/plain: {stats(d.server_round_s / d.plain_aggregate_s)}')
print(f'whole run: servers {d.server_round_s.sum() / 3600:.2f} h')

print('\n== cost against clipped clients (least squares over all rounds)')
fits = {}
for col, unit in (('server_round_s', 's'), ('sent_mb', 'MB'), ('mpc_rounds', 'rounds')):
    slope, intercept = np.polyfit(d.clipped_secure, d[col], 1)
    r2 = np.corrcoef(d.clipped_secure, d[col])[0, 1] ** 2
    fits[col] = slope, intercept
    print(f'{col} = {intercept:.4g} + {slope:.4g} {unit} per clipped client, R^2 {r2:.5f}')
print(f'sent_mb per clipped client / client upload: {fits["sent_mb"][0] / d.client_up_mb[0]:.1f}')

print('\n== client traffic per round (MB, from P)')
print(f'down {d.client_down_mb[0]:.3f}, up {d.client_up_mb[0]:.3f}, plaintext FL up {d.plain_client_up_mb[0]:.3f}')

# figure: the per-round cost is set by the number of clipped clients
sns.set_theme(style='whitegrid', context='notebook')
fig, ax = plt.subplots(figsize=(7, 2.8))
x = np.array([0, d.clipped_secure.max() + 5])
slope, intercept = fits['server_round_s']
ax.plot(x, (intercept + slope * x) / 60, color='grey', linewidth=1, linestyle='--', label='least-squares fit')
ax.scatter(later.clipped_secure, later.server_round_s / 60, s=14, label='rounds 2 to 50')
ax.scatter(d.clipped_secure[:1], d.server_round_s[:1] / 60, s=14, marker='s', label='round 1')
ax.set_xlim(0, x[1])
ax.set_ylim(0)
ax.set_xlabel('clipped clients')
ax.set_ylabel('server time per round (min)')
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
os.makedirs(arguments.plots, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(arguments.plots, f'secure_cost_vs_clipped.{ext}'), bbox_inches='tight', dpi=400)
