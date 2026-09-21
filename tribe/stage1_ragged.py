"""
Ragged-group generalization of tribe.stage1 / tribe.validate_stage1: identical Stage 1 program, except
groups are allowed to have DIFFERENT sizes (`group_sizes: list[int]` instead of a single `group_size: int`
assumed uniform across the whole batch). New, additive module — tribe/stage1.py and tribe/validate_stage1.py
are untouched, so every existing trainer's behavior (which relies on the uniform-group-size assumption for
its vectorized `.view(G, group_size)` reshapes / fixed-stride cvxpy slicing) is unaffected.

Why this exists: PRM800K (openai/prm800k) has a genuinely variable number of GPT-4 rollouts per problem (1
to 470, median 5) — the first dataset this project has needed Stage 1 on that isn't laid out in uniform
`group_size`-sized blocks by construction (every scripts/generate_offpolicy_*.py output uses a constant
`--num_generations`). Rather than downsampling every problem to a common group size (discarding real
rollouts either direction — positives, since they're rare, or negatives, since only a handful survive the
cap), this solves the SAME math the uniform-group-size code does, just without requiring the constraint
`len(dataset) % group_size == 0` in the first place.

Math: nothing about TRIBE-plan.tex's Stage 1 derivation requires groups to be the same size — the
per-context constraint is `sum_{i in group(x)} rho_i == |group(x)|` for each context x, which is perfectly
well-defined for `|group(x)|` varying by x. The uniform-group-size code's `.view(G, group_size)` reshapes
and fixed-stride `rho[g*group_size:(g+1)*group_size]` slicing are pure implementation conveniences for the
common case (vectorizes cleanly when every group is the same size), not a mathematical requirement.
"""

import math
import warnings

import cvxpy as cp
import numpy as np
import torch
from torch import Tensor

from .validate_stage1 import _TRUST_REGION_F


def _constraint_value_ragged(
    A_groups: list[Tensor], c_groups: list[Tensor], beta: float, log_lambda: Tensor
) -> tuple[list[Tensor], Tensor]:
    """
    Ragged-group counterpart to tribe.stage1._constraint_value: `A_groups`/`c_groups` are lists of
    variable-length 1-D tensors (one per context), rather than a single `(G, K)` rectangular tensor.

    Returns:
        Tuple of `(log_rho_groups, f_val)`:
            - `log_rho_groups` (`list[Tensor]`): log rho* for each group, same lengths as the inputs.
            - `f_val` (`Tensor` scalar): mean_{x,y}[rho* * log(rho*)] over ALL samples (uniform per-sample
              weight `1/N`, `N = sum of all group sizes`) — this is the same global trust-region quantity
              tribe.stage1._constraint_value computes, just accumulated group-by-group instead of via one
              rectangular reshape.
    """
    lam = torch.exp(log_lambda)
    log_rho_groups = []
    f_val_sum = A_groups[0].new_zeros(())
    total_n = 0
    for A_g, c_g in zip(A_groups, c_groups):
        K = A_g.numel()
        logits = (A_g - beta * c_g) / lam
        log_Z = torch.logsumexp(logits, dim=0) - math.log(K)
        log_rho = logits - log_Z
        log_rho_groups.append(log_rho)
        f_val_sum = f_val_sum + (torch.exp(log_rho) * log_rho).sum()
        total_n += K
    return log_rho_groups, f_val_sum / total_n


def solve_lambda_ragged(
    A_groups: list[Tensor],
    c_groups: list[Tensor],
    beta: float,
    eps: float,
    log_lambda_init: tuple[float, float] = (-10.0, 10.0),
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Tensor:
    """Ragged-group counterpart to tribe.stage1.solve_lambda — same bisection procedure, see that
    function's own docstring for the algorithm and the safety-clamp rationale."""
    lo, hi = log_lambda_init
    device, dtype = A_groups[0].device, A_groups[0].dtype
    lo = torch.tensor(lo, dtype=dtype, device=device)
    hi = torch.tensor(hi, dtype=dtype, device=device)
    safe_log_lambda_bound = 50.0

    for _ in range(max_iter):
        if _constraint_value_ragged(A_groups, c_groups, beta, lo)[1] >= eps or lo <= -safe_log_lambda_bound:
            break
        lo = lo - 10.0
    lo = torch.clamp(lo, min=-safe_log_lambda_bound)
    for _ in range(max_iter):
        if _constraint_value_ragged(A_groups, c_groups, beta, hi)[1] <= eps or hi >= safe_log_lambda_bound:
            break
        hi = hi + 10.0
    hi = torch.clamp(hi, max=safe_log_lambda_bound)

    for _ in range(max_iter):
        if (hi - lo) < tol:
            break
        mid = (lo + hi) / 2
        f_mid = _constraint_value_ragged(A_groups, c_groups, beta, mid)[1]
        if f_mid > eps:
            lo = mid
        else:
            hi = mid

    return torch.exp((lo + hi) / 2)


def solve_stage1_ragged_cvxpy(
    A: np.ndarray,
    c: np.ndarray,
    group_sizes: list[int],
    beta: float,
    eps: float,
    divergence: str = "chi_squared",
    solver: str | None = None,
) -> tuple[np.ndarray, float, str]:
    """
    Ragged-group counterpart to tribe.validate_stage1.solve_stage1_cvxpy — identical program, except the
    per-context normalization constraints use each group's own offset/size (from `group_sizes`) instead of
    a fixed stride. See that function's own docstring for the objective/constraints being solved.

    Args:
        A, c (`np.ndarray` of shape `(N,)`):
            Per-sample advantage / reference log-ratio, flattened over all `N = sum(group_sizes)` samples,
            laid out in contiguous per-group blocks matching the ORDER of `group_sizes` (block `g` has
            `group_sizes[g]` samples).
        group_sizes (`list[int]`):
            Size of each context's group, in the same order as the contiguous blocks in `A`/`c`.
        beta, eps, divergence, solver:
            See `tribe.validate_stage1.solve_stage1_cvxpy`.
    """
    if divergence not in _TRUST_REGION_F:
        raise ValueError(f"Unknown divergence {divergence!r}. Must be one of {list(_TRUST_REGION_F)}.")
    N = A.shape[0]
    if N != sum(group_sizes):
        raise ValueError(f"A has {N} samples but group_sizes sums to {sum(group_sizes)}.")
    p = 1.0 / N

    rho = cp.Variable(N, nonneg=True)
    objective = cp.sum(p * cp.multiply(rho, A - beta * c)) + beta * cp.sum(p * cp.entr(rho))
    trust_region = cp.sum(p * _TRUST_REGION_F[divergence](rho)) <= eps

    offsets = np.cumsum([0] + list(group_sizes))
    group_constraints = [
        cp.sum(rho[offsets[g] : offsets[g + 1]]) == group_sizes[g] for g in range(len(group_sizes))
    ]

    problem = cp.Problem(cp.Maximize(objective), [trust_region, *group_constraints])
    problem.solve(solver=solver)
    return rho.value, trust_region.dual_value, problem.status


def compute_rho_star_ragged(
    A: Tensor,
    c: Tensor,
    group_sizes: list[int],
    beta: float,
    eps: float,
    divergence: str = "chi_squared",
) -> tuple[Tensor | None, Tensor | None]:
    """
    Ragged-group counterpart to tribe.stage1.compute_rho_star. `kl_new_old` uses the ragged closed form
    (`solve_lambda_ragged`, fast, exact — same derivation as tribe.stage1's closed form, generalized to
    per-group sizes); any other divergence goes through cvxpy (`solve_stage1_ragged_cvxpy`), same
    None-on-failure contract as `tribe.stage1.compute_rho_star`.

    Args:
        A, c (`Tensor` of shape `(N,)`):
            Flattened per-sample advantage / reference log-ratio, laid out in contiguous per-group blocks
            matching `group_sizes`'s order.
        group_sizes (`list[int]`):
            Size of each group, in the same order as the contiguous blocks in `A`/`c`.
        beta, eps, divergence:
            See `tribe.stage1.compute_rho_star`.

    Returns:
        Tuple of `(rho, lam)`, same shape/None-on-failure contract as `tribe.stage1.compute_rho_star`.
    """
    if divergence == "kl_new_old":
        A_groups = list(torch.split(A, group_sizes))
        c_groups = list(torch.split(c, group_sizes))
        lam = solve_lambda_ragged(A_groups, c_groups, beta, eps)
        log_rho_groups, _ = _constraint_value_ragged(A_groups, c_groups, beta, torch.log(lam))
        return torch.cat([torch.exp(lr) for lr in log_rho_groups]), lam

    rho, lam, status = solve_stage1_ragged_cvxpy(
        A.detach().float().cpu().numpy(), c.detach().float().cpu().numpy(), group_sizes, beta, eps, divergence=divergence
    )
    if status == "optimal_inaccurate":
        warnings.warn(
            f"cvxpy ragged Stage 1 solve for divergence={divergence!r} returned 'optimal_inaccurate' — "
            "using the approximate solution.",
            category=UserWarning,
            stacklevel=2,
        )
    elif status != "optimal":
        warnings.warn(
            f"cvxpy ragged Stage 1 solve for divergence={divergence!r} returned status {status!r} (not "
            "'optimal' or 'optimal_inaccurate') — skipping this batch.",
            stacklevel=2,
        )
        return None, None
    return (
        torch.as_tensor(rho, dtype=A.dtype, device=A.device),
        torch.as_tensor(lam, dtype=A.dtype, device=A.device),
    )
