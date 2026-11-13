"""Our reference summary statistics for Notebook 2. For us, not for the students.

The solutions notebook derives them step by step. They live here, outside
tutorial_utils.py, so that a student reading the helpers does not find the answer
to Part 2. The pre-trained fallback posteriors need them to reduce a trajectory.
"""

import numpy as np
from scipy import stats


def _wells(x, threshold=0.5):
    """Which well is the particle in? -1 left, +1 right; inside the barrier region, the last one."""
    side = np.sign(x) * (np.abs(x) > threshold)
    last = np.where(side != 0, np.arange(x.shape[1])[None], 0)
    last = np.maximum.accumulate(last, axis=1)
    state = np.take_along_axis(side, last, axis=1)
    state[state == 0] = -1  # every trajectory starts in the left well
    return state


def count_hops(x, threshold=0.5):
    """Number of well-to-well transitions per trajectory, ignoring recrossings at the top."""
    return (np.diff(_wells(x, threshold), axis=1) != 0).sum(axis=1)


def baseline_summary(x):
    """The two-number baseline: mean and standard deviation of x."""
    return np.stack([x.mean(axis=1), x.std(axis=1)], axis=1)


def static_summary(x):
    """Four numbers from the histogram of |x|: no time order involved."""
    a = np.abs(x)
    return np.stack([
        (a < 0.5).mean(axis=1),  # barrier occupancy
        a.mean(axis=1),
        a.std(axis=1),
        stats.kurtosis(a, axis=1),
    ], axis=1)


def dynamic_summary(x):
    """Four numbers that need the time order: displacements at three lags, hop count."""
    return np.stack([
        (x[:, 1:] - x[:, :-1]).std(axis=1),
        (x[:, 10:] - x[:, :-10]).std(axis=1),
        (x[:, 100:] - x[:, :-100]).std(axis=1),
        count_hops(x),
    ], axis=1)


def reference_summary(x):
    """Our eight reference statistics: four static, four dynamic."""
    return np.concatenate([static_summary(x), dynamic_summary(x)], axis=1)
