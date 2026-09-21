import math

import cvxpy as cp
import numpy as np
import pytest
import torch

from tribe.stage1 import TRUST_REGION_F, compute_rho_star, filter_zero_variance_groups
from tribe.validate_stage1 import _TRUST_REGION_F as TRUST_REGION_F_CVXPY
from tribe.validate_stage1 import solve_stage1_cvxpy


@pytest.mark.parametrize("divergence", ["kl_new_old", "kl_old_new", "chi_squared"])
def test_torch_and_cvxpy_f_divergences_match(divergence):
    # tribe.stage1.TRUST_REGION_F (torch, for cheap monitoring) must stay numerically identical to
    # validate_stage1._TRUST_REGION_F (cvxpy, for solving) — both docstrings promise this.
    rho = np.array([0.3, 0.7, 1.0, 1.5, 2.5])
    torch_vals = TRUST_REGION_F[divergence](torch.tensor(rho)).numpy()

    rho_var = cp.Variable(len(rho))
    rho_var.value = rho
    cvxpy_vals = TRUST_REGION_F_CVXPY[divergence](rho_var).value

    np.testing.assert_allclose(torch_vals, cvxpy_vals, atol=1e-8)


def _reps_reference(A: np.ndarray, group_size: int, eps: float) -> np.ndarray:
    """
    Independent, standalone implementation of the classic REPS/MPO closed form (single KL-to-old
    constraint, no reference term), coded directly against the textbook formula rather than reusing
    `tribe.stage1`, to give a non-tautological ground truth for the c=0 reduction check.
    """
    G = A.shape[0] // group_size
    A_grouped = A.reshape(G, group_size)

    def constraint(eta: float) -> float:
        logits = A_grouped / eta
        log_Z = np.logaddexp.reduce(logits, axis=1, keepdims=True) - math.log(group_size)
        log_rho = logits - log_Z
        return float(np.mean(np.exp(log_rho) * log_rho))

    lo, hi = -10.0, 10.0
    while constraint(math.exp(lo)) < eps:
        lo -= 10.0
    while constraint(math.exp(hi)) > eps:
        hi += 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if constraint(math.exp(mid)) > eps:
            lo = mid
        else:
            hi = mid
    eta = math.exp((lo + hi) / 2)

    logits = A_grouped / eta
    log_Z = np.logaddexp.reduce(logits, axis=1, keepdims=True) - math.log(group_size)
    return np.exp(logits - log_Z).reshape(-1)


@pytest.mark.parametrize("G,K,beta,eps,seed", [(3, 6, 0.1, 0.05, 0), (5, 4, 0.5, 0.02, 1), (2, 8, 0.0, 0.1, 2)])
def test_closed_form_matches_cvxpy(G, K, beta, eps, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=2.0, size=G * K)
    c = rng.normal(scale=0.5, size=G * K)

    rho_closed, lam_closed = compute_rho_star(torch.tensor(A), torch.tensor(c), beta, K, eps)
    rho_cvx, lam_cvx, status = solve_stage1_cvxpy(A, c, beta, K, eps)

    assert status == "optimal"
    np.testing.assert_allclose(rho_closed.numpy(), rho_cvx, atol=1e-3)
    # The objective's -beta*E[rho*log(rho)] term has the same functional form as the trust-region
    # constraint, so at the KKT stationarity point they combine additively: TRIBE's lambda (the
    # coefficient of (A - beta*c) in the closed form) equals beta plus the constraint's own Lagrange
    # multiplier (cvxpy's dual value), not the dual value alone.
    assert lam_closed.item() == pytest.approx(lam_cvx + beta, rel=0.05)


@pytest.mark.parametrize("G,K,eps,seed", [(3, 6, 0.05, 0), (5, 4, 0.02, 1), (4, 8, 0.2, 2), (2, 10, 0.01, 3)])
def test_chi_squared_beta0_closed_form_matches_cvxpy(G, K, eps, seed):
    # Moderate scale/eps chosen so the closed form's implicit rho>=0 assumption holds (checked below) --
    # this is testing the feasible case; the infeasible/fallback case is covered separately.
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=1.0, size=G * K)
    c = np.zeros(G * K)  # matches how compute_rho_star_offpolicy always calls this (c is always 0 there)

    rho_closed, lam_closed = compute_rho_star(
        torch.tensor(A), torch.tensor(c), beta=0.0, group_size=K, eps=eps, divergence="chi_squared",
        use_chi_squared_beta0_closed_form=True,
    )
    rho_cvx, lam_cvx, status = solve_stage1_cvxpy(A, c, 0.0, K, eps, divergence="chi_squared")

    assert status == "optimal"
    assert (rho_closed.numpy() >= 0).all(), "test fixture should stay in the closed form's feasible region"
    np.testing.assert_allclose(rho_closed.numpy(), rho_cvx, atol=1e-3)
    np.testing.assert_allclose(lam_closed.item(), lam_cvx, atol=1e-3)


def test_chi_squared_beta0_closed_form_falls_back_when_infeasible():
    # One extreme outlier in an otherwise tight group forces the unconstrained closed-form rho for that
    # sample well below 0 -- solve_stage1_cvxpy's own rho>=0 constraint would bind here, so the simple
    # formula is no longer exact and compute_rho_star must fall back to cvxpy instead of returning a
    # wrong (negative-rho) answer.
    K, eps = 4, 0.01
    A = np.array([-10.0, 0.1, 0.1, 0.1] * 3)
    c = np.zeros(A.shape)

    rho_fallback, lam_fallback = compute_rho_star(
        torch.tensor(A), torch.tensor(c), beta=0.0, group_size=K, eps=eps, divergence="chi_squared",
        use_chi_squared_beta0_closed_form=True,
    )
    rho_cvx, lam_cvx, status = solve_stage1_cvxpy(A, c, 0.0, K, eps, divergence="chi_squared")

    assert status == "optimal"
    assert (rho_fallback.numpy() >= -1e-6).all()  # cvxpy's own solution respects rho>=0
    np.testing.assert_allclose(rho_fallback.numpy(), rho_cvx, atol=1e-3)
    np.testing.assert_allclose(lam_fallback.item(), lam_cvx, atol=1e-3)


def test_chi_squared_beta0_closed_form_gated_by_flag(monkeypatch):
    # Default (flag omitted -> False): must still go through the general cvxpy path unchanged, not the
    # new closed form -- verified by asserting solve_stage1_cvxpy actually gets called.
    calls = []
    real_solve = solve_stage1_cvxpy

    def spy(*args, **kwargs):
        calls.append(1)
        return real_solve(*args, **kwargs)

    monkeypatch.setattr("tribe.validate_stage1.solve_stage1_cvxpy", spy)
    # compute_rho_star does `from .validate_stage1 import solve_stage1_cvxpy` as a local import inside the
    # function body, so patching the module attribute above is picked up on the next call.

    A = torch.tensor([1.0, -1.0, 0.5, -0.5])
    c = torch.zeros(4)
    compute_rho_star(A, c, beta=0.0, group_size=4, eps=0.05, divergence="chi_squared")

    assert len(calls) == 1, "flag defaults to False -- should use cvxpy, not the closed form"


def test_reps_reduction_when_c_is_zero():
    rng = np.random.default_rng(3)
    G, K, eps = 4, 6, 0.05
    A = rng.normal(scale=2.0, size=G * K)
    c = np.zeros(G * K)

    rho_reps = _reps_reference(A, K, eps)
    # beta is irrelevant once c == 0: the reference term vanishes entirely.
    for beta in (0.0, 0.3, 2.0):
        rho_tribe, _ = compute_rho_star(torch.tensor(A), torch.tensor(c), beta, K, eps)
        np.testing.assert_allclose(rho_tribe.numpy(), rho_reps, atol=1e-4)


@pytest.mark.parametrize("G,K,beta,eps,seed", [(4, 8, 0.2, 0.05, 0), (3, 5, 1.0, 0.1, 7)])
def test_per_context_normalization(G, K, beta, eps, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=2.0, size=G * K)
    c = rng.normal(scale=0.5, size=G * K)

    rho, _ = compute_rho_star(torch.tensor(A), torch.tensor(c), beta, K, eps)
    group_means = rho.view(G, K).mean(dim=1)
    torch.testing.assert_close(group_means, torch.ones(G, dtype=group_means.dtype), atol=1e-4, rtol=0)


def test_zero_variance_group_drops_reward_term():
    K, eps, beta = 6, 0.05, 0.3
    rng = np.random.default_rng(5)
    c = rng.normal(scale=0.5, size=K)

    # A single zero-variance group: same advantage for every sample in the group.
    A_constant = np.full(K, 3.7)

    rho_tribe, lam = compute_rho_star(torch.tensor(A_constant), torch.tensor(c), beta, K, eps)

    # Per TRIBE-plan.tex Section 2.3, rho* should reduce to being driven purely by -beta*c: the
    # constant A term must cancel out of the per-context normalizer (softmax is shift-invariant).
    logits = -beta * c / lam.item()
    log_Z = np.logaddexp.reduce(logits) - math.log(K)
    rho_reference_term_only = np.exp(logits - log_Z)

    np.testing.assert_allclose(rho_tribe.numpy(), rho_reference_term_only, atol=1e-4)


def test_globally_degenerate_batch_gives_rho_one_not_nan():
    # A == 0 (every group zero-variance) AND c == 0 (pi_old == pi_ref, exactly true at the very first
    # training step) together make (A - beta*c) constant at 0 across the WHOLE batch, so f_val is
    # identically 0 for every lambda and can never reach eps. Before the numeric-safety clamp in
    # solve_lambda, the bracket-expansion loop chased this unreachable target down to log_lambda ~ -1010,
    # where exp() underflows to exact 0.0 and 0/0 poisons rho* with NaN — this crashed real training
    # (device-side assert on the very next generate() call, from sampling NaN/inf probabilities after an
    # optimizer step on NaN gradients). Correct behavior: this batch carries no informative Stage-1
    # signal, so rho* should come out as 1 everywhere (a no-op update), not NaN.
    A = torch.zeros(16)
    c = torch.zeros(16)
    rho, lam = compute_rho_star(A, c, beta=0.04, group_size=4, eps=0.05)

    assert not torch.isnan(rho).any()
    assert lam.item() > 0.0
    torch.testing.assert_close(rho, torch.ones_like(rho))


def test_mixed_batch_zero_variance_group_gets_reference_only_rho():
    # No exclusion/backfill in tribe.stage1 itself: a batch with one informative group and one
    # zero-variance group is solved jointly (both groups feed the same lambda), matching how GRPO trains
    # on every group regardless of its own advantage rather than special-casing zero-variance ones.
    K, beta, eps = 6, 0.3, 0.05
    rng = np.random.default_rng(11)
    A_normal = rng.normal(scale=2.0, size=K)
    c_normal = rng.normal(scale=0.5, size=K)
    c_flat_group = rng.normal(scale=0.5, size=K)
    A_flat_group = np.full(K, 3.7)  # zero-variance: constant advantage

    A = np.concatenate([A_normal, A_flat_group])
    c = np.concatenate([c_normal, c_flat_group])
    rho, lam = compute_rho_star(torch.tensor(A), torch.tensor(c), beta, K, eps)
    rho_flat_group = rho[K:]

    # The zero-variance group's rho* still reduces to reference-term-only, at the SAME lambda the joint
    # solve produced (not a separately-solved one).
    logits = -beta * c_flat_group / lam.item()
    log_Z = np.logaddexp.reduce(logits) - math.log(K)
    rho_reference_term_only = np.exp(logits - log_Z)

    np.testing.assert_allclose(rho_flat_group.numpy(), rho_reference_term_only, atol=1e-4)
    # Still normalizes to mean 1 within its own group.
    assert rho_flat_group.mean().item() == pytest.approx(1.0, abs=1e-4)


_F_DIVERGENCES = {
    "kl_new_old": lambda rho: rho * np.log(rho),
    "kl_old_new": lambda rho: -np.log(rho),
    "chi_squared": lambda rho: 0.5 * (rho - 1) ** 2,
}


@pytest.mark.parametrize("divergence", ["kl_old_new", "chi_squared"])
@pytest.mark.parametrize("G,K,beta,eps,seed", [(3, 6, 0.1, 0.05, 0), (4, 5, 0.3, 0.02, 4)])
def test_general_divergence_satisfies_its_own_constraint(divergence, G, K, beta, eps, seed):
    # Non-KL divergences are solved via cvxpy directly (see validate_stage1.py), so this isn't checking
    # the closed form against cvxpy (there is no closed form to check) — it's checking that the returned
    # rho* actually satisfies ITS OWN constraint (using the f formula independently reimplemented here,
    # not cvxpy's), and normalizes correctly, i.e. that the cvxpy dispatch in compute_rho_star is wired
    # up correctly (right f, right eps, right groups) rather than silently solving the wrong problem.
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=2.0, size=G * K)
    c = rng.normal(scale=0.5, size=G * K)

    rho, _ = compute_rho_star(torch.tensor(A), torch.tensor(c), beta, K, eps, divergence=divergence)
    rho = rho.numpy()

    f = _F_DIVERGENCES[divergence]
    assert f(rho).mean() == pytest.approx(eps, rel=1e-3, abs=1e-4)

    group_means = rho.reshape(G, K).mean(axis=1)
    np.testing.assert_allclose(group_means, np.ones(G), atol=1e-4)


def test_unknown_divergence_raises():
    A = torch.randn(12)
    c = torch.randn(12)
    with pytest.raises(ValueError):
        compute_rho_star(A, c, beta=0.1, group_size=4, eps=0.05, divergence="not_a_real_divergence")


def test_filter_zero_variance_groups():
    K = 4
    rewards = torch.tensor(
        [
            1.0, 1.0, 1.0, 1.0,  # zero-variance group
            0.0, 1.0, 0.0, 1.0,  # has variance
            -1.0, -1.0, -1.0, -1.0,  # zero-variance group
        ]
    )
    mask = filter_zero_variance_groups(rewards, K)
    torch.testing.assert_close(mask, torch.tensor([False, True, False]))
