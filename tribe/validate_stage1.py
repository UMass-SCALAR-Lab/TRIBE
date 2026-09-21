"""
Brute-force validation of the Stage 1 closed form against a direct convex solve — and, for trust-region
divergences other than KL, the actual solver (not just a validation check).

Poses the batch-level program from TRIBE-plan.tex Section 2.1 directly in cvxpy, with no reference to the
closed-form rho*/lambda derivation, so it can be compared against `tribe.stage1`'s KL closed form on real
rollout batches before that closed form is trusted at full training scale.

For a general trust-region divergence f (`E_old[f(rho)] <= eps` in place of the KL-specific
`E_old[rho*log(rho)] <= eps`), the objective's own `-beta * E_old[rho*log(rho)]` term (from the
KL-to-reference decomposition) keeps its ordinary KL shape regardless of f — only the trust-region term
changes — so the pointwise stationarity condition becomes `lambda*f'(rho) + beta*(log(rho)+1) = ...`, a
transcendental equation in rho with no closed form except when f is itself the KL shape (the one case
where f' and the objective's own log term coincide and combine additively, as `tribe.stage1` exploits).
Rather than hand-deriving and root-finding that mixed equation, `solve_stage1_cvxpy` just hands the whole
convex program to cvxpy for any DCP-representable f, sidestepping the need for a custom nested solve.
"""

import cvxpy as cp
import numpy as np


# Each entry maps a divergence name to a callable building the DCP-compliant cvxpy expression for
# f(rho), elementwise, given the cvxpy `rho` variable. `f(1) = 0` for all of these, as required for an
# f-divergence. Named by explicit argument order (not "forward"/"reverse", which convention flips
# depending on the field) so there is nothing to misremember:
#   - "kl_new_old" = KL(pi || pi_old) = E_old[rho*log(rho)]: TRIBE's own trust region (matches the
#     objective's reference-KL term shape, the one case with a closed form in `tribe.stage1`).
#   - "kl_old_new" = KL(pi_old || pi) = E_old[-log(rho)]: TRPO's own convention (a plain expectation
#     under pi_old, no importance weight needed) — different from TRIBE's choice above.
# Only "kl_new_old" has a closed-form solver in `tribe.stage1`; the others are solved via cvxpy directly.
_TRUST_REGION_F = {
    "kl_new_old": lambda rho: -cp.entr(rho),  # f(rho) = rho*log(rho)
    "kl_old_new": lambda rho: -cp.log(rho),  # f(rho) = -log(rho)
    "chi_squared": lambda rho: 0.5 * cp.square(rho - 1),  # f(rho) = (rho-1)^2 / 2
}


def solve_stage1_cvxpy(
    A: np.ndarray,
    c: np.ndarray,
    beta: float,
    group_size: int,
    eps: float,
    divergence: str = "kl_new_old",
    solver: str | None = None,
) -> tuple[np.ndarray, float, str]:
    """
    Solve TRIBE Stage 1 directly as a convex program (no closed form).

    max_rho  sum_i p_i * rho_i * (A_i - beta * c_i) + beta * sum_i p_i * entr(rho_i)
    s.t.     sum_i p_i * f(rho_i) <= eps                                 (global trust region)
             sum_{i in group(x)} rho_i == group_size, for every group x  (per-context normalization)

    where `entr(x) = -x * log(x)`, `p_i = 1/N` (uniform weight over the batch), and `f` is chosen by
    `divergence` (see `_TRUST_REGION_F`). The objective's own `entr(rho)` term is unaffected by
    `divergence`: it comes from the reference-KL decomposition, not the trust-region constraint.

    Args:
        A (`np.ndarray` of shape `(N,)`):
            Per-sample advantage, flattened over `N = G * group_size` samples.
        c (`np.ndarray` of shape `(N,)`):
            Per-sample reference log-ratio `log(pi_old/pi_ref)`, flattened the same way as `A`.
        beta (`float`):
            Weight on the reference term.
        group_size (`int`):
            Number of samples `K` per context (prompt group).
        eps (`float`):
            Trust-region budget.
        divergence (`str`, *optional*, defaults to `"kl_new_old"`):
            Trust-region divergence, one of `"kl_new_old"`, `"kl_old_new"`, `"chi_squared"` (see
            `_TRUST_REGION_F`).
        solver (`str`, *optional*):
            cvxpy solver name (e.g. `"CLARABEL"`); left to cvxpy's default if not given.

    Returns:
        Tuple of `(rho, lam, status)`:
            - `rho` (`np.ndarray` of shape `(N,)`): the solved `rho*`.
            - `lam` (`float`): the dual variable of the global trust-region constraint. For
              `divergence="kl_new_old"`, this is `lambda_raw`, not `tribe.stage1`'s combined
              `lambda = lambda_raw + beta` (see the module docstring); for any other divergence there is
              no such recombination, so this dual value is the only meaningful `lambda`.
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

    group_constraints = [
        cp.sum(rho[g * group_size : (g + 1) * group_size]) == group_size for g in range(G)
    ]

    problem = cp.Problem(cp.Maximize(objective), [trust_region, *group_constraints])
    problem.solve(solver=solver)

    return rho.value, trust_region.dual_value, problem.status
