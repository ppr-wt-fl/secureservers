import numpy as np

from aggregator_class import Aggregator, weighted_mean

# EPPRFL (Li et al., "Efficiently Achieving Privacy Preservation and Poisoning Attack Resistance in Federated
# Learning", IEEE TIFS 19, 2024), Filtering & Clipping (F&C) method of Sec. V-A, here in plaintext:
#   1. initialization: random index set R_idx of eps * d distinct coordinates
#   2. downsample the updates and the historical update D_h to R_idx
#   3. d_i = squared Euclidean distance of the downsampled update i to the downsampled D_h; in the first round
#      (no D_h yet) d_i = squared norm of the downsampled update, and m_h = median of all N d_i
#   4. filtering: client i is benign if (d_i - m_h)^2 < lambda * m_h
#   5. clipping: every benign update scaled by min(1, m_h / d_i)
#   6. aggregate = mean of the clipped benign updates
#   7. history: append the aggregate to U_h, D_h = mean of U_h; m = median of d_i over the benign clients,
#      append m to M_h, m_h = mean of M_h
# eps = 0.06 and lambda = 0.05 as in the paper's evaluation (Sec. VII-B5).
#
# No official code; written from the paper. Choices for what the paper leaves open:
#   - R_idx is drawn once per run (step 1 is "Initialization"), uniformly without replacement, from random_generator
#   - no benign client: zero update (round skipped, as DnC) and the history is left unchanged, since there is no
#     aggregate and no median for this round
#   - d_i = 0: no clipping (factor 1)
# Only the R_idx coordinates of D_h are ever used, so only those are kept (the same numbers as a full D_h).
# Note: d_i is a squared distance, so the filter and the clipping factor are not scale invariant; they are copied
# as written


class EPPRFL(Aggregator):
    name = 'EPPRFL'

    def __init__(self, random_generator, eps=0.06, filter_lambda=0.05):
        self.random_generator = random_generator
        self.eps = eps
        self.filter_lambda = filter_lambda
        self.indices = None
        # sum of the past aggregates (on R_idx); one median per past aggregate, so D_h = sum / len(medians)
        self.history_sum = None
        self.medians = []
        self.m_h = None

    def aggregate(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        updates = np.asarray(updates)
        n, d = updates.shape

        if self.indices is None:
            self.indices = np.sort(self.random_generator.choice(d, int(self.eps * d), replace=False))

        sampled = updates[:, self.indices].astype(np.float64)
        if len(self.medians) == 0:
            distances = (sampled ** 2).sum(1)
        else:
            historical = self.history_sum / len(self.medians)
            distances = ((sampled - historical) ** 2).sum(1)

        if self.m_h is None:
            self.m_h = float(np.median(distances))

        differences = (distances - self.m_h) ** 2
        benign = np.flatnonzero(differences < self.filter_lambda * self.m_h)

        # logged per round by experiment.py
        self.last_info = {
            'epprfl_m_h': self.m_h,
            'epprfl_n_benign': len(benign),
            'epprfl_distances': [float(f'{distance:.6g}') for distance in distances],
        }
        if len(benign) == 0:
            return np.zeros(d, updates.dtype), benign

        factors = np.ones(len(benign))
        for k in range(len(benign)):
            distance = distances[benign[k]]
            if distance > 0:
                factors[k] = min(1.0, self.m_h / distance)
        self.last_info['epprfl_n_clipped'] = int((factors < 1).sum())
        aggregated = weighted_mean(updates, benign, factors.astype(updates.dtype))

        if self.history_sum is None:
            self.history_sum = np.zeros(len(self.indices))
        self.history_sum += aggregated[self.indices]
        self.medians.append(float(np.median(distances[benign])))
        self.m_h = float(np.mean(self.medians))

        return aggregated.astype(updates.dtype), benign
