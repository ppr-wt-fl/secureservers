from abc import ABC, abstractmethod

import numpy as np


# base class for attacks as federated_learning_GOO in experiment.py uses them:
# perform_attack swaps in the malicious updates, experiment.py scores the global model after aggregation
class Attack(ABC):
    # shown in the results, e.g. 'MinMax'
    name: str

    # updates: (n_clients, d) honest updates of this round, global_weights: flat global model they were trained from.
    # returns the (n_clients, d) updates the server aggregates and the indices of the clients the attacker
    # controls (the rows it replaced), which the experiment uses as ground truth for the aggregator's filtering
    @abstractmethod
    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


# no-attack baseline: no malicious clients
class NoAttack(Attack):
    name = 'None'

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return updates, np.array([], dtype=int)
