"""
Ragged-group counterpart to tribe.offpolicy_stage1 — see that module's docstring for why `c=0` throughout
the single-iteration off-policy regime, and tribe/stage1_ragged.py's docstring for why ragged (variable
`group_sizes`) groups need their own solver rather than reusing tribe.stage1's uniform-`group_size` one.
"""

import torch
from torch import Tensor

from .stage1_ragged import compute_rho_star_ragged


def compute_rho_star_offpolicy_ragged(
    rewards: Tensor,
    group_sizes: list[int],
    beta: float,
    eps: float,
    divergence: str = "chi_squared",
) -> tuple[Tensor | None, Tensor | None]:
    """
    Ragged-group counterpart to tribe.offpolicy_stage1.compute_rho_star_offpolicy: `c` is fixed at 0 (same
    reasoning as the uniform-group-size version — pi_old and pi_ref are both the frozen behavioral policy
    for the whole single epoch), so only the reward-derived, per-group-centered advantage `A` drives `rho*`.

    Args:
        rewards (`Tensor` of shape `(N,)`):
            Per-completion reward, flattened over `N = sum(group_sizes)` completions, laid out in
            contiguous per-group blocks matching `group_sizes`'s order (same layout
            `compute_rho_star_ragged` expects for `A`/`c`).
        group_sizes (`list[int]`):
            Size of each context's group, in the same order as the contiguous blocks in `rewards`.
        beta (`float`):
            Weight on the reference term. Has no effect here (`c` is always 0) — kept as an explicit
            argument so callers don't need regime-specific branching to pass it.
        eps (`float`):
            Trust-region budget, same meaning as `TribeConfig.trust_region_eps`.
        divergence (`str`, *optional*, defaults to `"chi_squared"`):
            See `tribe.stage1_ragged.compute_rho_star_ragged`. Defaults to `"chi_squared"` here (not
            `"kl_new_old"`, unlike the uniform-group-size version's default) to match this project's own
            established best off-policy TRIBE config (see results_llama_final.tex).

    Returns:
        Tuple of `(rho, lam)`, same shape/None-on-failure contract as
        `tribe.stage1_ragged.compute_rho_star_ragged`.
    """
    reward_groups = torch.split(rewards, group_sizes)
    A_groups = [g - g.mean() for g in reward_groups]
    A = torch.cat(A_groups)
    c = torch.zeros_like(A)
    return compute_rho_star_ragged(A, c, group_sizes, beta, eps, divergence=divergence)
