from abc import ABC, abstractmethod

import numpy as np


# base class for aggregators as federated_learning_GOO in experiment.py uses them:
# aggregate turns the (possibly attacked) client updates into the update added to the global model
class Aggregator(ABC):
    # short name for the console output and the results, e.g. 'FedAvg'
    name: str

    # updates: (n_clients, d) updates of this round, global_weights: flat global model they were trained from
    # (unused by purely update-based aggregators, needed e.g. for cosine filters on models).
    # returns (aggregated (d,) update, sorted indices of the updates that went into it). The indices are None
    # if the aggregator doesn't select (e.g. FedAvg) or can't tell which ones it used, empty if it accepted nobody
    @abstractmethod
    def aggregate(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        ...


# wraps a ByzFL aggregator (callable on (n, d) updates), e.g. ByzFLAggregator('FedAvg', Average()).
# ByzFL only returns the aggregate, so no selection is reported, also for filters like Krum / MultiKrum
class ByzFLAggregator(Aggregator):
    def __init__(self, name, aggregation):
        self.name = name
        self.aggregation = aggregation

    def aggregate(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        return self.aggregation(updates), None


# (factors[:, None] * updates[rows]).mean(0), or updates[rows].mean(0) without factors, added up row by row:
# the same numbers (numpy also sums axis 0 row after row) without the (len(rows), d) temporaries
def weighted_mean(updates, rows, factors=None):
    total = None
    for i in range(len(rows)):
        if factors is None:
            row = updates[rows[i]]
        else:
            row = factors[i] * updates[rows[i]]
        if total is None:
            total = row.copy()
        else:
            total += row
    return total / len(rows)
