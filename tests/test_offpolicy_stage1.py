import numpy as np
import pytest
import torch

from tribe.offpolicy_stage1 import compute_rho_star_offpolicy
from tribe.stage1 import compute_rho_star


@pytest.mark.parametrize("G,K,beta,eps,seed", [(4, 16, 0.1, 0.05, 0), (3, 32, 0.5, 0.02, 1)])
def test_matches_compute_rho_star_with_c_zero(G, K, beta, eps, seed):
    # compute_rho_star_offpolicy should be exactly compute_rho_star with c hardcoded to 0 — beta must be
    # irrelevant to the result (only affects lambda's bookkeeping, not rho* itself, per
    # test_reps_reduction_when_c_is_zero in test_stage1.py).
    rng = np.random.default_rng(seed)
    rewards = torch.tensor(rng.integers(0, 2, size=G * K), dtype=torch.float64)

    rho_offpolicy, lam_offpolicy = compute_rho_star_offpolicy(rewards, K, beta, eps)

    rewards_grouped = rewards.view(G, K)
    A = (rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)).reshape(-1)
    c = torch.zeros_like(A)
    rho_reference, lam_reference = compute_rho_star(A, c, beta, K, eps)

    torch.testing.assert_close(rho_offpolicy, rho_reference)
    assert lam_offpolicy.item() == pytest.approx(lam_reference.item())


def test_rho_star_mean_one_per_question():
    G, K = 5, 16
    rng = np.random.default_rng(2)
    rewards = torch.tensor(rng.integers(0, 2, size=G * K), dtype=torch.float64)

    rho, _ = compute_rho_star_offpolicy(rewards, K, beta=0.1, eps=0.05)

    group_means = rho.view(G, K).mean(dim=1)
    torch.testing.assert_close(group_means, torch.ones(G, dtype=group_means.dtype), atol=1e-4, rtol=0)


def test_all_zero_variance_questions_give_rho_one_not_nan():
    # Every question's group_size completions get the same reward (e.g. all wrong) -- A == 0 everywhere,
    # and c is always 0 in the off-policy regime, so this hits the same global-degenerate case
    # test_globally_degenerate_batch_gives_rho_one_not_nan already covers for compute_rho_star directly.
    rewards = torch.zeros(4 * 16, dtype=torch.float64)
    rho, lam = compute_rho_star_offpolicy(rewards, group_size=16, beta=0.1, eps=0.05)

    assert not torch.isnan(rho).any()
    assert lam.item() > 0.0
    torch.testing.assert_close(rho, torch.ones_like(rho))
