# Sign flipping (Allen-Zhu et al., "Byzantine-Resilient Non-Convex Stochastic Gradient Descent", ICLR'21 / arXiv 2020),
# ByzFL's SignFlipping (https://github.com/LPD-EPFL/byzfl, byzfl/attacks/attacks.py), which is -mean(vectors),
# written out here so the image needs no byzfl.
# ByzFL feeds it all honest vectors; here the attackers only use their own honest updates (partial knowledge),
# as in the other untargeted attacks. All attackers send the same vector, as ByzFL does ([byz_vector] * f)

import numpy as np
from .attack_class import Attack


# every attacker sends minus the mean of the attackers' honest updates
class SignFlippingAttack(Attack):
    name = 'SignFlipping'

    def __init__(self, n_attackers):
        self.n_attackers = n_attackers

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        malicious = np.arange(self.n_attackers)
        mal_update = -updates[malicious].mean(0)

        # attackers replace their own rows (the first n_attackers)
        updates = updates.copy()
        updates[malicious] = mal_update
        return updates, malicious
