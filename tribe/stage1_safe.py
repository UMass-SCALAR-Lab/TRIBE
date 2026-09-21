"""
Safe-TRIBE Stage 1: reward-maximizing rho* subject to an additional expected-cost constraint.

Extends `tribe.validate_stage1.solve_stage1_cvxpy`'s convex program with a second, linear constraint —
`E_old[rho * Cost] <= cost_limit` — mirroring Safe-RLHF's own Lagrangian safety constraint
(https://huggingface.co/papers/2310.12773), but solved as a single constrained convex program rather than
their primal-dual/Lagrange-multiplier iteration. `Cost` is per-sample (e.g. the cost model's score on that
completion, or a cost *advantage* if centered per-group the same way the reward advantage `A` is), one
value per sample, same flattening/grouping convention as `A`/`c` elsewhere in this package.

Adding a second linear constraint keeps the program convex (a linear constraint never breaks
convexity), so this always goes through cvxpy — there is no closed form to fall back to even for
`divergence="kl_new_old"`, unlike `tribe.stage1.compute_rho_star` (whose KL-new-old closed form solves the
*unconstrained-by-cost* problem only). Reuses `validate_stage1._TRUST_REGION_F` for the trust-region term
so the two solvers stay in lockstep if that mapping ever changes.
"""

import cvxpy as cp
import numpy as np
import torch
from torch import Tensor

from .validate_stage1 import _TRUST_REGION_F


def solve_stage1_safe_cvxpy(
    A: np.ndarray,
    c: np.ndarray,
    cost: np.ndarray,
    beta: float,
    group_size: int,
    eps: float,
    cost_limit: float,
    divergence: str = "kl_new_old",
    solver: str | None = None,
) -> tuple[np.ndarray, float, float, str]:
    """
    Solve Safe-TRIBE Stage 1 as a constrained convex program.

    max_rho  sum_i p_i * rho_i * (A_i - beta * c_i) + beta * sum_i p_i * entr(rho_i)
    s.t.     sum_i p_i * f(rho_i) <= eps                                 (trust region, same as unconstrained)
             sum_i p_i * rho_i * Cost_i <= cost_limit                    (expected-cost constraint)
             sum_{i in group(x)} rho_i == group_size, for every group x  (per-context normalization)

    Args:
        A (`np.ndarray` of shape `(N,)`):
            Per-sample reward advantage, flattened over `N = G * group_size` samples.
        c (`np.ndarray` of shape `(N,)`):
            Per-sample reference log-ratio `log(pi_old/pi_ref)`, flattened the same way as `A`.
        cost (`np.ndarray` of shape `(N,)`):
            Per-sample cost signal (e.g. cost-model score, or a group-centered cost advantage), flattened
            the same way as `A`.
        beta (`float`):
            Weight on the reference term.
        group_size (`int`):
            Number of samples `K` per context (prompt group).
        eps (`float`):
            Trust-region budget.
        cost_limit (`float`):
            Upper bound on `E_old[rho * Cost]`. If `cost` is a raw (uncentered) cost-model score, this is
            an absolute budget; if `cost` is a group-centered advantage (zero mean under `rho == 1`), a
            `cost_limit` of `0` recovers Safe-RLHF's own "no worse than the current policy, on average"
            framing.
        divergence (`str`, *optional*, defaults to `"kl_new_old"`):
            Trust-region divergence, one of `"kl_new_old"`, `"kl_old_new"`, `"chi_squared"` (see
            `validate_stage1._TRUST_REGION_F`).
        solver (`str`, *optional*):
            cvxpy solver name (e.g. `"CLARABEL"`); left to cvxpy's default if not given.

    Returns:
        Tuple of `(rho, lam_trust_region, lam_cost, status)`:
            - `rho` (`np.ndarray` of shape `(N,)`): the solved `rho*`.
            - `lam_trust_region` (`float`): dual variable of the trust-region constraint.
            - `lam_cost` (`float`): dual variable of the cost constraint (the Lagrange multiplier
              Safe-RLHF's own dual-ascent procedure would otherwise iterate on).
            - `status` (`str`): cvxpy solver status, e.g. `"optimal"`.
    """
    if divergence not in _TRUST_REGION_F:
        raise ValueError(f"Unknown divergence {divergence!r}. Must be one of {list(_TRUST_REGION_F)}.")

    N = A.shape[0]
    G = N // group_size
    p = 1.0 / N

    rho = cp.Variable(N, nonneg=True)

    objective = cp.sum(p * cp.multiply(rho, A - beta * c)) + beta * cp.sum(p * cp.entr(rho))
    trust_region = cp.sum(p * _TRUST_REGION_F[divergence](rho)) <= eps
    cost_constraint = cp.sum(p * cp.multiply(rho, cost)) <= cost_limit

    group_constraints = [
        cp.sum(rho[g * group_size : (g + 1) * group_size]) == group_size for g in range(G)
    ]

    problem = cp.Problem(cp.Maximize(objective), [trust_region, cost_constraint, *group_constraints])
    problem.solve(solver=solver)

    return rho.value, trust_region.dual_value, cost_constraint.dual_value, problem.status


def compute_rho_star_safe(
    A: Tensor,
    c: Tensor,
    cost: Tensor,
    beta: float,
    group_size: int,
    eps: float,
    cost_limit: float,
    divergence: str = "kl_new_old",
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """
    Torch-facing wrapper around `solve_stage1_safe_cvxpy`, matching `tribe.stage1.compute_rho_star`'s
    calling convention and None-on-failure contract (callers must skip the batch when `rho` is `None`, not
    treat it as a value to use — same as the unconstrained solver).

    Args:
        A, c, cost (`Tensor` of shape `(N,)`):
            See `solve_stage1_safe_cvxpy`.
        beta, group_size, eps, cost_limit, divergence:
            See `solve_stage1_safe_cvxpy`.

    Returns:
        Tuple of `(rho, lam_trust_region, lam_cost)`, all `None` if the solve did not return `"optimal"`
        or `"optimal_inaccurate"`.
    """
    import warnings

    rho, lam_tr, lam_cost, status = solve_stage1_safe_cvxpy(
        A.detach().float().cpu().numpy(),
        c.detach().float().cpu().numpy(),
        cost.detach().float().cpu().numpy(),
        beta,
        group_size,
        eps,
        cost_limit,
        divergence=divergence,
    )
    if status == "optimal_inaccurate":
        warnings.warn(
            f"cvxpy Safe-TRIBE Stage 1 solve for divergence={divergence!r} returned 'optimal_inaccurate' "
            "— using the approximate solution.",
            category=UserWarning,
            stacklevel=2,
        )
    elif status != "optimal":
        warnings.warn(
            f"cvxpy Safe-TRIBE Stage 1 solve for divergence={divergence!r} returned status {status!r} "
            "(not 'optimal' or 'optimal_inaccurate') — skipping this batch.",
            stacklevel=2,
        )
        return None, None, None
    return (
        torch.as_tensor(rho, dtype=A.dtype, device=A.device),
        torch.as_tensor(lam_tr, dtype=A.dtype, device=A.device),
        torch.as_tensor(lam_cost, dtype=A.dtype, device=A.device),
    )
