"""Reference data and networks for Notebook 1 (2D toy problem). For us, not for the students.

    python reference/make_reference_toy.py

Run once, commit what it writes to data/, and the notebook loads it from there:

    toy_observations.pt              100 IID observations at theta = (3, -1.5); the first is x_o
    toy_training_set.pt              10,000 pairs (theta, x), the fallback training set
    toy_posterior_hidden_prior.pkl   Part 3: NPE trained with a hidden, narrower prior
    toy_posteriors_calibration.pkl   Part 4: three NPE networks, simulator noise 0.25, 0.5, 1.0
    toy_posterior_nle.pkl            Part 5: NLE posterior, the fallback
    toy_hidden.pt                    the hidden answers of Parts 3 and 4

The posteriors are pickled, so run this with the sbi version the notebooks pin.
The script ends on a printed check of every network against the exact posterior.
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # everything here is small: stay on the CPU

import torch
from sbi.inference import NLE, NPE
from sbi.utils import BoxUniform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tutorial_utils as tu

warnings.filterwarnings("ignore")

SEED = 2041
THETA_TRUE = torch.tensor([3.0, -1.5])
NUM_SIMULATIONS = 10_000
# Part 3: the prior of the black box. Narrower than [-5, 5] and not symmetric, so it cannot be guessed.
WALL_LOW, WALL_HIGH = torch.tensor([-1.4, -3.7]), torch.tensor([4.2, 0.9])
# Part 4: which letter was trained with which simulator noise (the true noise is 0.5).
CALIBRATION = {"A": (1.0, "underconfident"), "B": (0.25, "overconfident"), "C": (0.5, "fine")}
# Accepted coverage of the 90% interval of theta_1, measured with 1,000 tests. The exact posterior
# has 90% whatever the prior. A well-trained network reaches about 88%: near the walls of the
# prior the flow is slightly too narrow. The wide network stays near 96%: the exact posterior of a
# noise-1.0 model would give 98%, because near the walls its central interval misses on one side.
COVERAGE_BANDS = {"overconfident": (0.0, 0.70), "fine": (0.865, 0.915), "underconfident": (0.94, 1.0)}
# Part 5: MCMC settings of the NLE posterior. Cheap enough for a free Colab CPU.
MCMC = dict(num_chains=20, thin=1, warmup_steps=20)


def train_npe(prior, noise=tu.TOY_NOISE):
    """NPE with sbi's default flow (MAF), retrained if the training got stuck.

    With 10,000 simulations about one MAF training in ten ends in a state where the
    network ignores one coordinate of x: that marginal stays as wide as the prior.
    Outside its training prior the MAF behaves well (it piles up at the walls), which
    Part 3 relies on, so we keep it and retrain the rare bad run.
    """
    probe = prior.sample((1,))[0]
    for attempt in range(10):
        theta = prior.sample((NUM_SIMULATIONS,))
        x = tu.toy_simulator(theta, noise=noise)
        inference = NPE(prior=prior, show_progress_bars=False)
        inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior()
        spread = posterior.sample((2000,), x=probe, show_progress_bars=False).std(dim=0)
        if (spread < 2.5 * noise).all():  # a healthy posterior is about as wide as the noise
            return posterior
        print(f"  training got stuck (posterior spread {spread.tolist()}), retraining")
    raise RuntimeError("NPE training failed ten times in a row")


def save_pickle(obj, name):
    with open(tu.DATA_DIR / name, "wb") as f:
        pickle.dump(obj, f)


def coverage(posterior, prior, num_tests=1000):
    """How often do the central 50% and 90% intervals of theta_1 contain the parameter
    that made the data? Returns the two rates."""
    theta_star = prior.sample((num_tests,))
    x_star = tu.toy_simulator(theta_star)
    hits = torch.zeros(2)
    for t, x in zip(theta_star, x_star):
        samples = posterior.sample((500,), x=x, show_progress_bars=False)
        q = torch.quantile(samples[:, 0], torch.tensor([0.05, 0.25, 0.75, 0.95]))
        hits += torch.tensor([float(q[1] <= t[0] <= q[2]), float(q[0] <= t[0] <= q[3])])
    return (hits / num_tests).tolist()


def nle_width_error(posterior_nle, x_iid, n_values=(1, 5, 25, 100)):
    """Largest relative error of the 90% interval widths against the exact posterior."""
    ratios = []
    for n in n_values:
        samples = posterior_nle.sample((2000,), x=x_iid[:n], show_progress_bars=False)
        ratios.append(tu.interval_width(samples) / tu.ExactPosterior(x_iid[:n]).width())
    ratios = torch.stack(ratios)
    return (ratios - 1).abs().max().item(), ratios


def main():
    torch.manual_seed(SEED)
    tu.DATA_DIR.mkdir(exist_ok=True)
    prior = BoxUniform(low=tu.TOY_LOW * torch.ones(2), high=tu.TOY_HIGH * torch.ones(2))

    # observations
    x_iid = tu.toy_simulator(THETA_TRUE.repeat(100, 1))
    x_o = x_iid[0]
    torch.save({"x_iid": x_iid, "theta_true": THETA_TRUE}, tu.DATA_DIR / "toy_observations.pt")
    assert 0.5 < torch.dist(x_o, THETA_TRUE) < 1.0, "x_o should be clearly off the truth, yet typical"
    assert ((x_o - WALL_LOW) > 1.5).all() and ((WALL_HIGH - x_o) > 1.5).all()

    # fallback training set
    theta = prior.sample((NUM_SIMULATIONS,))
    x = tu.toy_simulator(theta)
    torch.save({"theta": theta, "x": x}, tu.DATA_DIR / "toy_training_set.pt")

    # Part 3: hidden, narrower prior
    hidden_prior = BoxUniform(low=WALL_LOW, high=WALL_HIGH)
    posterior_hidden = train_npe(hidden_prior)
    save_pickle(posterior_hidden, "toy_posterior_hidden_prior.pkl")

    # Part 4: one overconfident, one underconfident, one fine. Each is retrained until its
    # coverage is typical for its kind, so that a student's check with 200 tests can tell them apart.
    networks, coverages = {}, {}
    for letter, (noise, label) in CALIBRATION.items():
        for attempt in range(5):
            networks[letter] = train_npe(prior, noise=noise)
            coverages[letter] = coverage(networks[letter], prior)
            if COVERAGE_BANDS[label][0] < coverages[letter][1] < COVERAGE_BANDS[label][1]:
                break
        else:
            raise RuntimeError(f"network {letter} never reached a typical coverage")
    save_pickle(networks, "toy_posteriors_calibration.pkl")

    # Part 5: NLE on the fallback training set. The width of an NLE posterior varies by 5 to 20%
    # from one training run to the next, so the fallback is retrained until it is a good one.
    for attempt in range(10):
        inference = NLE(prior=prior, density_estimator="mdn", show_progress_bars=False)
        inference.append_simulations(theta, x).train()
        posterior_nle = inference.build_posterior(mcmc_method="slice_np_vectorized", mcmc_parameters=MCMC)
        if nle_width_error(posterior_nle, x_iid)[0] < 0.12:
            break
    else:
        raise RuntimeError("no NLE training run came within 12% of the exact widths")
    save_pickle(posterior_nle, "toy_posterior_nle.pkl")

    torch.save({
        "wall_low": WALL_LOW, "wall_high": WALL_HIGH,
        "calibration": {letter: label for letter, (_, label) in CALIBRATION.items()},
    }, tu.DATA_DIR / "toy_hidden.pt")

    # ---- printed check -------------------------------------------------------------
    print(f"x_o = {x_o.tolist()}, {torch.dist(x_o, THETA_TRUE):.2f} from the true theta")
    exact = tu.ExactPosterior(x_o)

    samples = posterior_hidden.sample((2000,), x=x_o, show_progress_bars=False)
    exact_hidden = tu.ExactPosterior(x_o, low=WALL_LOW.numpy(), high=WALL_HIGH.numpy())
    score = tu.mismatch_score(samples, exact_hidden.sample((2000,), seed=1))
    print(f"Part 3  hidden-prior network at x_o: mismatch {score:.2f} (inside its walls: should be ≈ 0.5)")
    assert score < tu.load_benchmarks()["toy"]["c2st_max"]
    for x_far, wall in ((torch.tensor([9.0, -1.0]), WALL_HIGH[0]), (torch.tensor([-9.0, -1.0]), WALL_LOW[0])):
        s = posterior_hidden.sample((1000,), x=x_far, show_progress_bars=False)
        edge = s[:, 0].max() if x_far[0] > 0 else s[:, 0].min()
        print(f"        at x = {x_far.tolist()}: samples pile up at θ₁ = {edge:.2f} (wall at {wall:.1f})")
        assert abs(edge - wall) < 0.05

    for letter, (noise, label) in CALIBRATION.items():
        samples = networks[letter].sample((2000,), x=x_o, show_progress_bars=False)
        score = tu.mismatch_score(samples, exact.sample((2000,), seed=1))
        cov_50, cov_90 = coverages[letter]
        print(f"Part 4  network {letter} (noise {noise}, {label}): mismatch {score:.2f}, "
              f"coverage of the 50% interval {cov_50:.0%}, of the 90% interval {cov_90:.0%}")

    worst, ratios = nle_width_error(posterior_nle, x_iid)
    for n, ratio in zip((1, 5, 25, 100), ratios):
        print(f"Part 5  NLE with n = {n:3d}: 90% width / answer key = {ratio[0]:.2f}, {ratio[1]:.2f}")
    tolerance = tu.load_benchmarks()["toy"]["iid_width_tolerance"]
    print(f"        worst deviation {worst:.0%} (the benchmark allows {tolerance:.0%})")
    assert worst < tolerance


if __name__ == "__main__":
    main()
