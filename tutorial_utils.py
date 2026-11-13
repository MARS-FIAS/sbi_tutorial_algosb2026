"""Helpers for the AlgoSB 2026 SBI tutorial.

Everything the two notebooks import lives here: the answer key of the toy problem,
the double-well simulator, the score helper, the plotting helpers and the check
cells. The notebooks stay short; read this file if you want to know what a check does.

Contents
    shared            colours, data loading, the scorecard
    Notebook 1 (toy)  ExactPosterior, plot_cloud, BlackBox, check_by_hand ... check_repair
    Notebook 2 (well) simulate_double_well, summarize, train_and_score,
                      posterior_widths, check_practice ... check_sparse
"""

import contextlib
import io
import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "data"

# One colour per role, the same in every figure of both notebooks.
C_YOURS = "#2a78d6"  # blue: what you built
C_KEY = "#eb6834"  # orange: the answer key, the observation, the truth
C_OTHER = "#1baf7a"  # aqua: a third series
C_MUTED = "#8a8985"  # grey: context (prior box, reference lines)

PASS, FAIL = "✅ PASS", "❌ NOT YET"
RESULTS = {}  # filled by the check cells, read by scorecard()


# --------------------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------------------
def load_benchmarks():
    """The targets of all check cells. Written by the scripts in reference/."""
    with open(DATA_DIR / "benchmarks.json") as f:
        return json.load(f)


def _load(name):
    """Load a tensor file from data/."""
    return torch.load(DATA_DIR / name, weights_only=True)


def _load_pickle(name):
    """Load a pickled sbi posterior from data/ (written with the pinned sbi version)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(DATA_DIR / name, "rb") as f:
                return pickle.load(f)
    except Exception as err:  # noqa: BLE001 - pickles break when library versions differ
        raise RuntimeError(
            f"Could not load the pre-trained network data/{name}. It was saved with other "
            "versions of sbi or PyTorch than the ones installed here. Tell an instructor; "
            "the fix is to run the scripts in reference/ in this environment."
        ) from err


def _report(key, passed, message):
    """Print the verdict of a check cell and remember it for the scorecard."""
    RESULTS[key] = bool(passed)
    print(f"{PASS if passed else FAIL} · {key} · {message}")


def scorecard():
    """Print which challenges of this session have passed."""
    if not RESULTS:
        print("No check cell has run yet.")
        return
    width = max(len(k) for k in RESULTS)
    for key, passed in RESULTS.items():
        print(f"{key:<{width}}  {PASS if passed else FAIL}")
    print(f"\n{sum(RESULTS.values())} of {len(RESULTS)} checks passed.")


def set_plot_style():
    """Quiet axes and readable fonts for every figure."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "font.size": 11,
        "legend.frameon": False,
    })


def plot_posteriors(samples, labels, limits=None, axis_labels=None, truth=None,
                    truth_label="true θ", title=None):
    """Overlay up to three sets of posterior samples: the joint, then one marginal per parameter.

    Args:
        samples: one tensor of shape (n, 2), or a list of up to three.
        labels: one name per set of samples, for the legend.
        limits: [[low_1, high_1], [low_2, high_2]] of the axes. Default: wherever the samples are.
        axis_labels: names of the two parameters.
        truth: optional parameter to mark, shape (2,).
    """
    from matplotlib.lines import Line2D

    if isinstance(samples, (torch.Tensor, np.ndarray)):
        samples, labels = [samples], [labels] if isinstance(labels, str) else labels
    samples = [np.asarray(s, dtype=np.float64).reshape(-1, 2) for s in samples]
    axis_labels = axis_labels or TOY_LABELS
    colors = [C_YOURS, C_KEY, C_OTHER]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), width_ratios=[1.2, 1, 1])
    pooled = np.concatenate(samples)
    for s, color in zip(samples, colors):
        keep = s[:: max(1, len(s) // 1500)]
        axes[0].scatter(keep[:, 0], keep[:, 1], s=6, alpha=0.3, color=color, linewidths=0)
        for d in (0, 1):
            span = limits[d] if limits is not None else (pooled[:, d].min(), pooled[:, d].max())
            inside = pooled[(pooled[:, d] >= span[0]) & (pooled[:, d] <= span[1]), d]
            bins = np.histogram_bin_edges(inside if len(inside) else span, bins=45)
            axes[d + 1].hist(s[:, d], bins=bins, density=True, histtype="step", lw=1.6, color=color)
    handles = [Line2D([], [], color=c, lw=2, label=l) for c, l in zip(colors, labels)]
    if truth is not None:
        truth = np.asarray(truth, dtype=np.float64).reshape(2)
        axes[0].scatter(*truth, marker="x", s=90, color="black", linewidths=2, zorder=5)
        for d in (0, 1):
            axes[d + 1].axvline(truth[d], color="black", lw=1.2, ls="--")
        handles.append(Line2D([], [], color="black", lw=1.2, ls="--", label=truth_label))
    axes[0].set(xlabel=axis_labels[0], ylabel=axis_labels[1])
    for d in (0, 1):
        axes[d + 1].set(xlabel=axis_labels[d], yticks=[])
        axes[d + 1].grid(False)
    if limits is not None:
        axes[0].set(xlim=limits[0], ylim=limits[1])
        axes[1].set_xlim(limits[0])
        axes[2].set_xlim(limits[1])
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.07))
    if title:
        axes[0].set_title(title, loc="left", fontsize=11)
    return axes


def _as_tensor(a, name="value"):
    try:
        return torch.as_tensor(np.asarray(a, dtype=np.float32))
    except Exception as err:  # noqa: BLE001 - tell the student what was wrong
        raise TypeError(f"`{name}` should be a list, array or tensor of numbers.") from err


# --------------------------------------------------------------------------------------
# Notebook 1: the 2D toy problem
# --------------------------------------------------------------------------------------
TOY_LOW, TOY_HIGH, TOY_NOISE = -5.0, 5.0, 0.5
TOY_LABELS = [r"$\theta_1$", r"$\theta_2$"]


def toy_simulator(theta, noise=TOY_NOISE):
    """x = theta + noise * eps: one noisy 2D point per parameter."""
    theta = torch.as_tensor(theta, dtype=torch.float32)
    return theta + noise * torch.randn_like(theta)


def load_toy_observations():
    """The 100 IID observations of Part 5. The first one is the x_o of Parts 1 to 4."""
    return _load("toy_observations.pt")["x_iid"]


def load_toy_training_set():
    """Fallback: 10,000 pre-simulated (theta, x) pairs of the toy problem."""
    d = _load("toy_training_set.pt")
    return d["theta"], d["x"]


class ExactPosterior:
    """The answer key of the toy problem.

    With a uniform prior on a box and Gaussian noise, the posterior of theta given
    n IID points is a Gaussian centred on their mean with standard deviation
    noise / sqrt(n), cut off at the prior box.

    Args:
        x_o: one observation, shape (2,), or n IID observations, shape (n, 2).
        noise: standard deviation of the simulator noise.
        low, high: edges of the prior box, one number or one per parameter.
    """

    def __init__(self, x_o, noise=TOY_NOISE, low=TOY_LOW, high=TOY_HIGH):
        x_o = np.asarray(x_o, dtype=np.float64).reshape(-1, 2)
        self.num_observations = len(x_o)
        self.loc = x_o.mean(axis=0)
        self.scale = noise / np.sqrt(self.num_observations)
        self.low = np.broadcast_to(np.asarray(low, dtype=np.float64), (2,))
        self.high = np.broadcast_to(np.asarray(high, dtype=np.float64), (2,))
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale
        self._dist = stats.truncnorm(a, b, loc=self.loc, scale=self.scale)

    def sample(self, sample_shape=(1000,), seed=None):
        """Draw samples, shape (num_samples, 2)."""
        n = int(np.prod(sample_shape))
        u = np.random.default_rng(seed).uniform(size=(n, 2))
        return torch.as_tensor(self._dist.ppf(u), dtype=torch.float32)

    def log_prob(self, theta):
        """Log density at theta, shape (..., 2) -> (...)."""
        theta = np.asarray(theta, dtype=np.float64)
        return torch.as_tensor(self._dist.logpdf(theta).sum(axis=-1), dtype=torch.float32)

    def interval(self, level=0.9):
        """Central interval of each parameter: tensor [[low_1, low_2], [high_1, high_2]]."""
        tail = (1 - level) / 2
        return torch.as_tensor(
            np.stack([self._dist.ppf(tail), self._dist.ppf(1 - tail)]), dtype=torch.float32
        )

    def width(self, level=0.9):
        """Width of the central interval of each parameter, shape (2,)."""
        low, high = self.interval(level)
        return high - low

    def std(self):
        """Standard deviation of each parameter, shape (2,)."""
        return torch.as_tensor(self._dist.std(), dtype=torch.float32)

    def contains(self, theta, level=0.95):
        """Is theta inside the smallest region that holds `level` of the posterior?"""
        log_probs = self.log_prob(self.sample((100_000,), seed=0))
        threshold = torch.quantile(log_probs, 1 - level)
        return bool(self.log_prob(_as_tensor(theta, "theta").reshape(2)) >= threshold)


def plot_cloud(thetas, x_o, simulator=toy_simulator, labels=None, num_points=500, zoom=None,
               ax=None):
    """Simulate a cloud of points at each theta and show it next to the observation.

    zoom=3 shows only the window x_o ± 3 instead of the whole prior box.
    """
    thetas = _as_tensor(thetas, "thetas").reshape(-1, 2)
    labels = labels or [f"simulations at θ = ({t[0]:.2f}, {t[1]:.2f})" for t in thetas]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))
    for theta, label, color in zip(thetas, labels, [C_YOURS, C_OTHER, C_MUTED]):
        cloud = simulator(theta.repeat(num_points, 1))
        ax.scatter(*cloud.T, s=6, alpha=0.35, color=color, linewidths=0)
        ax.scatter(*theta, marker="+", s=160, color=color, linewidths=2, label=label)
    ax.scatter(*x_o, marker="*", s=260, color=C_KEY, edgecolors="white", linewidths=1,
               zorder=5, label="observation $x_o$")
    x_o = np.asarray(x_o, dtype=np.float64).reshape(2)
    xlim, ylim = ((x_o[0] - zoom, x_o[0] + zoom), (x_o[1] - zoom, x_o[1] + zoom)) if zoom else \
        ((TOY_LOW, TOY_HIGH), (TOY_LOW, TOY_HIGH))
    ax.set(xlabel="$x_1$", ylabel="$x_2$", xlim=xlim, ylim=ylim, aspect="equal")
    ax.legend(loc="upper left", fontsize=9)
    return ax


def interval_width(samples, level=0.9):
    """Width of the central `level` interval of each parameter, from samples (n, d)."""
    samples = _as_tensor(samples, "samples")
    tail = (1 - level) / 2
    q = torch.quantile(samples, torch.tensor([tail, 1 - tail]), dim=0)
    return q[1] - q[0]


def check_by_hand(theta_best, theta_far, x_o):
    """Part 1. Best: its cloud is centred on x_o. Furthest: far away, still plausible."""
    b = load_benchmarks()["toy"]
    theta_best = _as_tensor(theta_best, "theta_best").reshape(2)
    theta_far = _as_tensor(theta_far, "theta_far").reshape(2)
    x_o = _as_tensor(x_o, "x_o").reshape(2)

    gen = torch.Generator().manual_seed(0)
    cloud = theta_best + TOY_NOISE * torch.randn(20_000, 2, generator=gen)
    miss = torch.dist(cloud.mean(0), x_o).item()
    best_ok = miss <= b["best_radius"]
    _report("Part 1 · best θ", best_ok,
            f"cloud centre is {miss:.2f} from x_o (target ≤ {b['best_radius']})")

    distance = torch.dist(theta_far, theta_best).item()
    inside = ExactPosterior(x_o).contains(theta_far, b["region_level"])
    far_ok = best_ok and distance >= b["far_distance"] and inside
    if not best_ok:
        hint = "find the best θ first"
    elif not inside:
        hint = "too far: x_o would be a rare outlier for this θ (outside the 95% region)"
    elif distance < b["far_distance"]:
        hint = f"only {distance:.2f} from your best θ: you can go further (target ≥ {b['far_distance']})"
    else:
        hint = f"{distance:.2f} from your best θ and still inside the 95% region"
    _report("Part 1 · furthest θ", far_ok, hint)


def mismatch_score(samples, reference_samples):
    """Classifier two-sample test: 0.5 = indistinguishable, 1.0 = always told apart."""
    from sbi.utils.metrics import c2st

    n = min(len(samples), len(reference_samples), 2000)
    return c2st(samples[:n], reference_samples[:n], seed=0).item()


def check_match(samples, num_simulations, x_o, low=TOY_LOW, high=TOY_HIGH, key="Part 2 · match"):
    """Part 2. Do samples of your posterior match samples of the answer key?"""
    b = load_benchmarks()["toy"]
    samples = _as_tensor(samples, "samples")
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 1000:
        print(f"`samples` should have shape (N, 2) with N ≥ 1000, got {tuple(samples.shape)}.")
        RESULTS[key] = False
        return
    exact = ExactPosterior(x_o, low=low, high=high)
    score = mismatch_score(samples, exact.sample((2000,), seed=1))
    _report(key, score <= b["c2st_max"],
            f"mismatch score {score:.2f} (target ≤ {b['c2st_max']}) "
            f"with {int(num_simulations):,} simulations")


class BlackBox:
    """A trained network you can ask for samples and nothing else."""

    def __init__(self, posterior):
        self.__posterior = posterior

    def sample(self, sample_shape=(1000,), x=None):
        """Posterior samples for one observation x of shape (2,). Returns shape (num_samples, 2)."""
        import logging

        if x is None:
            raise ValueError("Give the observation: black_box.sample((1000,), x=x_o)")
        x = _as_tensor(x, "x")
        if x.numel() != 2:
            raise ValueError(f"Give one observation of shape (2,) at a time, got shape {tuple(x.shape)}.")
        x = x.reshape(1, 2)
        num_samples = int(np.prod(sample_shape))
        logging.disable(logging.WARNING)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples = self.__posterior.sample(
                    (num_samples,), x=x, show_progress_bars=False,
                    max_sampling_time=2.0, return_partial_on_timeout=True)
        finally:
            logging.disable(logging.NOTSET)
        if len(samples) < num_samples:
            print(f"Note: for x = {x[0].tolist()} the network puts almost no probability inside "
                  f"its own prior. Returning the {len(samples)} samples found in 2 seconds.")
        return samples

    def __repr__(self):
        return "BlackBox(a trained NPE network; its training prior is not yours to see)"


def load_black_box():
    """Part 3: a network trained by someone else, with a prior you are not told."""
    return BlackBox(_load_pickle("toy_posterior_hidden_prior.pkl"))


def check_walls(low, high):
    """Part 3. Did you locate the four edges of the black box's training prior?"""
    b = load_benchmarks()["toy"]
    hidden = _load("toy_hidden.pt")
    guess = torch.stack([_as_tensor(low, "low").reshape(2), _as_tensor(high, "high").reshape(2)])
    truth = torch.stack([hidden["wall_low"], hidden["wall_high"]])
    ok = (guess - truth).abs() <= b["wall_tolerance"]
    names = [["θ₁ lower", "θ₂ lower"], ["θ₁ upper", "θ₂ upper"]]
    marks = ", ".join(f"{names[i][j]} {'✓' if ok[i, j] else '✗'}" for i in range(2) for j in range(2))
    _report("Part 3 · hidden walls", bool(ok.all()),
            f"{int(ok.sum())} of 4 edges within {b['wall_tolerance']} ({marks})")


def load_three_networks():
    """Part 4: three trained networks, {"A": network, "B": network, "C": network}."""
    return {k: BlackBox(p) for k, p in _load_pickle("toy_posteriors_calibration.pkl").items()}


def load_toy_nle():
    """Fallback for Part 5: an NLE posterior trained on 10,000 simulations, sampled with MCMC."""
    return _load_pickle("toy_posterior_nle.pkl")


_CALIBRATION_WORDS = {
    "overconfident": "overconfident", "over": "overconfident", "too narrow": "overconfident",
    "underconfident": "underconfident", "under": "underconfident", "too wide": "underconfident",
    "fine": "fine", "calibrated": "fine", "ok": "fine", "good": "fine", "well calibrated": "fine",
}


def check_calibration(labels):
    """Part 4. labels = {"A": "overconfident", "B": "fine", "C": "underconfident"} (in some order)."""
    hidden = _load("toy_hidden.pt")["calibration"]
    try:
        given = {k.upper(): _CALIBRATION_WORDS[str(v).strip().lower()] for k, v in labels.items()}
    except (AttributeError, KeyError):
        print('Give a dict like {"A": "overconfident", "B": "fine", "C": "underconfident"}.')
        RESULTS["Part 4 · which network lies"] = False
        return
    passed = given == hidden
    _report("Part 4 · which network lies", passed,
            "all three labels are right" if passed else "at least one label is wrong")


def check_iid(n_values, widths, n_needed, key="Part 5 · pin θ down"):
    """Part 5. Do your interval widths follow the answer key, and is n for std 0.1 right?

    Args:
        n_values: the numbers of observations you used, at least five between 1 and 100.
            For each n, use the first n points: x_iid[:n].
        widths: width of the 90% interval of your posterior for each n. Either one
            number per n (for θ₁) or two per n (θ₁ and θ₂).
        n_needed: your reading of how many observations give a standard deviation of 0.1.
    """
    b = load_benchmarks()["toy"]
    x_iid = load_toy_observations()
    n_values = [int(n) for n in np.asarray(n_values).reshape(-1)]
    widths = _as_tensor(widths, "widths")
    if widths.numel() not in (len(n_values), 2 * len(n_values)):
        _report(key, False, f"give one width per n: {len(n_values)} values of n, "
                            f"but {widths.numel()} widths")
        return
    widths = widths.reshape(len(n_values), -1)
    enough = len(set(n_values)) >= 5 and min(n_values) >= 1 and max(n_values) <= len(x_iid)
    if not enough:
        _report(key, False, "use at least five different n between 1 and 100")
        return
    if not isinstance(n_needed, (int, float, np.integer, np.floating)):
        _report(key, False, "set n_needed to the number of observations you read off your plot")
        return
    exact = torch.stack([ExactPosterior(x_iid[:n]).width(0.9) for n in n_values])
    error = (widths / exact[:, : widths.shape[1]] - 1).abs()
    worst = error.max().item()
    worst_n = n_values[int(error.max(dim=1).values.argmax())]
    widths_ok = worst <= b["iid_width_tolerance"]
    n_ok = abs(int(n_needed) - b["iid_n_needed"]) <= b["iid_n_tolerance"]
    message = (f"widths within {worst:.0%} of the answer key, worst at n = {worst_n} "
               f"(target ≤ {b['iid_width_tolerance']:.0%}); "
               f"n for a standard deviation of 0.1 is {'right' if n_ok else 'not right'}")
    _report(key, widths_ok and n_ok, message)


def check_bimodal(samples):
    """Bonus. With x depending on θ₁², both signs of θ₁ explain the data equally well."""
    samples = _as_tensor(samples, "samples")
    fraction = (samples[:, 0] > 0).float().mean().item()
    _report("Bonus · bend the simulator", 0.4 <= fraction <= 0.6,
            f"{fraction:.0%} of the samples have θ₁ > 0 (target: between 40% and 60%)")


def check_repair(samples, num_simulations, x_far, low, high):
    """Bonus. With your wider prior, does the posterior at x_far match its answer key?"""
    x_far = _as_tensor(x_far, "x_far").reshape(2)
    low = np.broadcast_to(np.asarray(low, dtype=np.float64), (2,))
    high = np.broadcast_to(np.asarray(high, dtype=np.float64), (2,))
    margin = 3 * TOY_NOISE
    if (x_far.numpy() - margin < low).any() or (x_far.numpy() + margin > high).any():
        _report("Bonus · repair", False,
                "the prior you report does not reach comfortably beyond x_far")
        return
    check_match(samples, num_simulations, x_far, low=low, high=high, key="Bonus · repair")


# --------------------------------------------------------------------------------------
# Notebook 2: Brownian motion in a double well
# --------------------------------------------------------------------------------------
DW_LOW, DW_HIGH = (2.0, -1.0), (6.0, 0.0)  # barrier height [k_BT], log10 D
DW_LABELS = [r"$\Delta G^{\ddagger}$ [$k_BT$]", r"$\log_{10} D$"]
DW_DT, DW_STRIDE, DW_FRAMES = 0.002, 50, 5000
DW_FRAME_TIME = DW_DT * DW_STRIDE  # time between two saved frames


def dw_prior():
    """Uniform prior: barrier height in [2, 6] k_BT, log10 D in [-1, 0]."""
    from sbi.utils import BoxUniform

    return BoxUniform(low=torch.tensor(DW_LOW), high=torch.tensor(DW_HIGH))


def double_well_potential(x, dG):
    """G_0(x) of the matched-harmonic double well: minima at x = ±1, barrier dG at x = 0."""
    u = np.abs(np.asarray(x, dtype=np.float64))
    return dG * np.where(u <= 0.5, 1 - 2 * u**2, 2 * (u - 1) ** 2)


def double_well_force(x, dG, tilt=0.0):
    """Force -dG/dx. A non-zero tilt adds the term tilt * x to the potential."""
    return -dG * np.where(np.abs(x) <= 0.5, -4 * x, 4 * (x - np.sign(x))) - tilt


def simulate_double_well(theta, seed=None, tilt=0.0):
    """Brownian dynamics in the double well, one trajectory per row of theta.

    Args:
        theta: shape (n, 2), columns barrier height [k_BT] and log10 D.
        seed: seed of the random number generator.
        tilt: slope added to the potential [k_BT per unit length]; 0 in the tutorial model.

    Returns:
        x: positions, shape (n, 5000), one frame every 50 integration steps.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float32))
    dG, D = theta[:, 0], 10 ** theta[:, 1]
    rng = np.random.default_rng(seed)
    drift = D * DW_DT
    kick = np.sqrt(2 * D * DW_DT)
    x_t = np.full(len(theta), -1.0, dtype=np.float32)
    x = np.empty((len(theta), DW_FRAMES), dtype=np.float32)
    for frame in range(DW_FRAMES):
        xi = rng.standard_normal((DW_STRIDE, len(theta)), dtype=np.float32)
        for step in range(DW_STRIDE):
            x_t = x_t + drift * double_well_force(x_t, dG, tilt) + kick * xi[step]
        x[:, frame] = x_t
    return torch.from_numpy(x)


def load_trajectory(name):
    """Hand over a trajectory with hidden parameters: "practice", "blind" or "bonus"."""
    return _load("dw_trajectories.pt")[name]


def load_test_set():
    """The common test set: 200 parameters and their raw trajectories."""
    d = _load("dw_test_set.pt")
    return d["theta"], d["x"]


def plot_trajectories(trajectories, labels, zoom=300, title=None):
    """Show trajectories three ways: the whole trace, its first frames, and the histogram."""
    colors = [C_KEY, C_YOURS, C_OTHER, C_MUTED]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), width_ratios=[3, 2, 1.4])
    for x, label, color in zip(trajectories, labels, colors):
        x = np.asarray(x).reshape(-1)
        t = np.arange(len(x)) * DW_FRAME_TIME
        axes[0].plot(t, x, lw=0.5, color=color, alpha=0.85, label=label)
        axes[1].plot(t[:zoom], x[:zoom], lw=1.0, color=color, alpha=0.9)
        axes[2].hist(x, bins=np.linspace(-2, 2, 61), density=True, histtype="step", lw=1.4,
                     color=color)
    axes[0].set(xlabel="time", ylabel="position $x$", title="whole trajectory")
    axes[1].set(xlabel="time", title=f"first {zoom} frames")
    axes[2].set(xlabel="position $x$", title="histogram", yticks=[])
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(labels), fontsize=9)
    if title:
        fig.suptitle(title, y=1.04)
    return axes


def summarize(summary, x):
    """Apply your summary function to trajectories and return a tensor for sbi.

    Your function receives a NumPy array of shape (n_trajectories, n_frames) and
    returns one row of numbers per trajectory, shape (n_trajectories, n_statistics).
    A single trajectory of shape (n_frames,) is fine too.
    """
    x = np.asarray(x, dtype=np.float32)
    x = x[None] if x.ndim == 1 else x
    s = np.asarray(summary(x), dtype=np.float32)
    if s.ndim == 1 and len(s) == len(x):
        s = s[:, None]
    if s.ndim != 2 or len(s) != len(x):
        raise ValueError(
            f"Your summary got {len(x)} trajectories and returned shape {s.shape}. "
            "It should return shape (n_trajectories, n_statistics): one row per trajectory. "
            "Use axis=1 in NumPy calls, e.g. x.mean(axis=1), and np.stack([...], axis=1)."
        )
    if not np.isfinite(s).all():
        bad = np.where(~np.isfinite(s).all(axis=0))[0].tolist()
        raise ValueError(
            f"Statistic(s) number {bad} (counting from 0) are NaN or infinite for some "
            "trajectories. Division by zero or a log of zero, e.g. for a trajectory "
            "that never crosses the barrier?"
        )
    return torch.from_numpy(s)


def train_and_score(summary, theta, x, seed=0, density_estimator=None, verbose=True,
                    **train_kwargs):
    """Train NPE on your summary statistics and score it on the common test set.

    Steps 3 and 4 of the recipe, plus a score:
        s = summary(x)                             reduce each trajectory to a few numbers
        NPE(prior).append_simulations(theta, s)    hand the training set to sbi
        .train()                                   train the density estimator
        score                                      average log posterior density at the
                                                   true parameters of 200 test trajectories

    Args:
        summary: your function, NumPy array (n, 5000) -> NumPy array (n, n_statistics).
        theta, x: the training set, parameters (n, 2) and raw trajectories (n, 5000).
        seed: seed of the network initialisation and the training.
        density_estimator: "maf", "nsf", "mdn", or a builder from sbi.neural_nets.posterior_nn.
            The default is a small neural spline flow: posterior_nn("nsf", num_transforms=2).
        train_kwargs: passed on to sbi's .train(), e.g. training_batch_size=50.
            The learning rate defaults to 1e-3.

    Returns:
        posterior, score. A posterior equal to the prior would score -1.39.
    """
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    if density_estimator is None:
        density_estimator = posterior_nn("nsf", num_transforms=2)
    train_kwargs.setdefault("learning_rate", 1e-3)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    s = summarize(summary, x)
    theta_test, x_test = load_test_set()
    s_test = summarize(summary, x_test)

    torch.manual_seed(seed)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):  # sbi's own chatter
        warnings.simplefilter("ignore")
        inference = NPE(prior=dw_prior(), density_estimator=density_estimator,
                        show_progress_bars=False)
        inference.append_simulations(theta, s).train(**train_kwargs)
        posterior = inference.build_posterior()
        log_probs = posterior.log_prob_batched(theta_test[None], x=s_test, norm_posterior=False)
    score = log_probs.mean().item()
    if verbose:
        print(f"score = {score:.2f}   ({s.shape[1]} statistics, {len(theta):,} simulations, "
              f"{inference.epoch} epochs)")
    return posterior, score


def posterior_widths(posterior, summary, verbose=True):
    """How much does the posterior narrow each parameter, on average over the test set?

    Returns the width of the 90% interval of each parameter as a fraction of the
    width of the prior's 90% interval: 1.0 means nothing learned, 0.1 means ten
    times narrower than the prior.
    """
    theta_test, x_test = load_test_set()
    s_test = summarize(summary, x_test)
    samples = torch.stack([_sample_quietly(posterior, 400, s_o) for s_o in s_test], dim=1)
    q = torch.quantile(samples, torch.tensor([0.05, 0.95]), dim=0)
    prior_width = 0.9 * (torch.tensor(DW_HIGH) - torch.tensor(DW_LOW))
    fractions = ((q[1] - q[0]) / prior_width).mean(dim=0)
    if verbose:
        print(f"90% interval, as a fraction of the prior's: "
              f"ΔG‡ {fractions[0]:.0%}, log D {fractions[1]:.0%}")
    return fractions[0].item(), fractions[1].item()


def _sample_quietly(posterior, num_samples, x, time_limit=2.0):
    """Posterior samples inside the prior box, without warnings or progress bars.

    sbi rejects samples outside the prior. If almost all of them are outside, that
    takes forever, so after `time_limit` seconds the remaining samples are drawn
    without rejection and moved onto the edge of the box.
    """
    import logging

    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = posterior.sample(
                (num_samples,), x=x, show_progress_bars=False,
                max_sampling_time=time_limit, return_partial_on_timeout=True)
            if len(samples) < num_samples:
                extra = posterior.sample((num_samples - len(samples),), x=x,
                                         show_progress_bars=False, reject_outside_prior=False)
                extra = torch.clamp(extra, torch.tensor(DW_LOW), torch.tensor(DW_HIGH))
                samples = torch.cat([samples, extra])
    finally:
        logging.disable(logging.NOTSET)
    return samples


def region_90(samples, truth=None, low=DW_LOW, high=DW_HIGH, level=0.9):
    """The smallest region that holds `level` of the samples.

    Returns (fraction of the prior box the region covers, whether `truth` is inside).
    The density is estimated from the samples with a Gaussian kernel.
    """
    low, high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    u = (np.asarray(samples, dtype=np.float64) - low) / (high - low)  # prior box -> unit square
    rng = np.random.default_rng(0)
    if len(u) > 4000:
        u = u[rng.choice(len(u), 4000, replace=False)]
    kde = stats.gaussian_kde(u.T)
    threshold = np.quantile(kde(u.T), 1 - level)
    axis = (np.arange(120) + 0.5) / 120
    grid = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    area = float((kde(grid.T) >= threshold).mean())
    inside = None
    if truth is not None:
        t = (np.asarray(truth, dtype=np.float64).reshape(2) - low) / (high - low)
        inside = bool(kde(t[:, None])[0] >= threshold)
    return area, inside


def check_practice(tries):
    """Part 1. tries = [(dG, D), ...] in the order you tried them; the last one is your answer."""
    b = load_benchmarks()["double_well"]
    truth = _load("dw_hidden.pt")["practice"]
    if len(tries) == 0:
        _report("Part 1 · by hand", False, "no tries yet")
        return
    dG, D = (float(v) for v in tries[-1])
    dG_true, D_true = truth[0].item(), 10 ** truth[1].item()
    dG_ok = abs(dG - dG_true) <= b["practice_dG_tolerance"]
    D_ok = max(D / D_true, D_true / D) <= b["practice_D_factor"] if D > 0 else False
    in_time = len(tries) <= b["practice_max_tries"]
    message = (f"try {len(tries)} of at most {b['practice_max_tries']}: "
               f"ΔG‡ {'within' if dG_ok else 'not within'} {b['practice_dG_tolerance']} k_BT, "
               f"D {'within' if D_ok else 'not within'} a factor {b['practice_D_factor']}")
    _report("Part 1 · by hand", dG_ok and D_ok and in_time, message)


def check_score(score):
    """Part 2. Bronze beats the baseline, silver reaches our reference, gold beats it."""
    b = load_benchmarks()["double_well"]
    medal = ("🥇 gold" if score > b["score_gold"] else
             "🥈 silver" if score >= b["score_silver"] else
             "🥉 bronze" if score > b["score_bronze"] else None)
    targets = (f"bronze > {b['score_bronze']:.2f}, silver ≥ {b['score_silver']:.2f}, "
               f"gold > {b['score_gold']:.2f}")
    _report("Part 2 · summary statistics", medal is not None,
            f"score {score:.2f}: {medal or 'no medal yet'} ({targets})")


def check_blind(samples):
    """Part 3. Does your 90% region contain the hidden truth, and is it small enough?"""
    b = load_benchmarks()["double_well"]
    truth = _load("dw_hidden.pt")["blind"]
    samples = _as_tensor(samples, "samples")
    if samples.ndim != 2 or samples.shape[1] != 2 or len(samples) < 1000:
        print(f"`samples` should have shape (N, 2) with N ≥ 1000, got {tuple(samples.shape)}.")
        RESULTS["Part 3 · blind trajectory"] = False
        return
    area, inside = region_90(samples, truth)
    small = area < b["blind_area_max"]
    message = (f"the 90% region {'contains' if inside else 'MISSES'} the truth and covers "
               f"{area:.1%} of the prior area (target < {b['blind_area_max']:.0%})")
    _report("Part 3 · blind trajectory", inside and small, message)


def reveal_blind_truth():
    """The hidden parameters of the blind trajectory, for the very end."""
    truth = _load("dw_hidden.pt")["blind"]
    print(f"Blind trajectory: ΔG‡ = {truth[0]:.2f} k_BT, D = {10 ** truth[1]:.3f} "
          f"(log10 D = {truth[1]:.3f})")
    return truth


def check_blind_to_D(posterior, summary):
    """Part 4. Does the posterior learn the barrier height and nothing about D?"""
    b = load_benchmarks()["double_well"]
    width_dG, width_logD = posterior_widths(posterior, summary, verbose=False)
    passed = width_dG < b["static_dG_width_max"] and width_logD > b["static_logD_width_min"]
    message = (f"ΔG‡ interval is {width_dG:.0%} of the prior's (target < "
               f"{b['static_dG_width_max']:.0%}), log D interval is {width_logD:.0%} "
               f"(target > {b['static_logD_width_min']:.0%})")
    _report("Part 4 · blind to D", passed, message)


def check_wrong_simulator(pick):
    """Bonus. Which of the three extra trajectories (0, 1 or 2) is from a tilted well?"""
    truth = int(_load("dw_hidden.pt")["bonus_tilted_index"])
    passed = int(pick) == truth
    _report("Bonus · wrong simulator", passed,
            "that one is from the tilted well" if passed else "that one is from the tutorial model")


def check_sparse(score, num_simulations):
    """Bonus. Back to bronze with 200 simulations."""
    b = load_benchmarks()["double_well"]
    few = int(num_simulations) <= 200
    passed = few and score > b["score_bronze"]
    message = (f"score {score:.2f} with {int(num_simulations)} simulations "
               f"(target > {b['score_bronze']:.2f} with at most 200)")
    _report("Bonus · sparse training data", passed, message)


# Our reference statistics. The solutions notebook derives them step by step.
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


class FallbackPosterior:
    """A pre-trained posterior that takes raw trajectories: it reduces them itself."""

    def __init__(self, posterior, summary):
        self.posterior, self.summary = posterior, summary

    def sample(self, sample_shape=(10_000,), x=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.posterior.sample(tuple(sample_shape), x=summarize(self.summary, x),
                                         show_progress_bars=False)


def load_fallback_posterior(name="reference"):
    """Pre-trained on 10,000 simulations: "baseline", "reference" or "static"."""
    summary = {"baseline": baseline_summary, "reference": reference_summary,
               "static": static_summary}[name]
    return FallbackPosterior(_load_pickle(f"dw_posterior_{name}.pkl"), summary)
