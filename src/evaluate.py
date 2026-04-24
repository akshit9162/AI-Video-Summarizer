import numpy as np


def evaluate(features, importance, idxs):
    return {
        "importance": float(np.mean(importance[idxs])),
        "coverage": float(np.mean(np.diff(sorted(idxs)))),
    }