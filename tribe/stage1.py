"""
Stage 1 of TRIBE: closed-form rho* and the single scalar root-find for lambda.

See TRIBE-plan.tex, Section 2.1. Given per-sample advantages A(x,y) and reference log-ratios
c(x,y) = log(pi_old(y|x)/pi_ref(y|x)), grouped into G contexts of K samples each (the same K-sample
group already drawn for the GRPO-style baseline), this solves

    rho*(x,y) = exp((A(x,y) - beta * c(x,y)) / lambda) / Z(x)
    Z(x)      = mean_{y in group(x)} exp((A(x,y) - beta * c(x,y)) / lambda)

for the single scalar lambda such that the global trust-region constraint

    mean_{x,y}[rho* * log(rho*)] = eps

holds. The constraint value is monotonically decreasing in lambda (lambda -> inf => rho* -> 1
everywhere => constraint value -> 0; lambda -> 0+ => rho* becomes extreme => constraint value -> inf),
so the root is found by bisection in log(lambda).

This closed form is specific to the "kl_new_old" trust region (f(rho) = rho*log(rho), i.e.
KL(pi || pi_old), in the general `E_old[f(rho)] <= eps` constraint from TRIBE-plan.tex eq. 50): the KKT
stationarity condition is `lambda*f'(rho) + beta*(log(rho)+1) = ...` in general (the objective's own
`-beta*E_old[rho*log(rho)]` term, from the KL-to-reference decomposition, keeps its log shape regardless
of which f the trust region uses), and only for f(rho) = rho*log(rho) does `f'(rho) = log(rho)+1`
coincide with that term and combine additively into the single exp()-invertible coefficient `lambda+beta`
used above. Note this is the opposite argument order from TRPO's own trust region, `KL(pi_old || pi)`
(named "kl_old_new" here) — TRIBE ends up with "kl_new_old" only because it reuses the same shape as the
separate reference-KL term, not by matching TRPO's convention. Any divergence other than "kl_new_old"
(`compute_rho_star(..., divergence=...)`) has no such closed form and is solved via cvxpy instead — see
`validate_stage1.py`.
"""

import math
import warnings

import torch
from torch import Tensor


# Torch-native mirror of validate_stage1._TRUST_REGION_F (that one is cvxpy expressions, for solving;
# this one is plain tensor ops, for cheap monitoring during training — e.g. measuring the realized
# f-divergence of the trained policy against pi_old, without a cvxpy round-trip on every step). Keep in
# sync with validate_stage1._TRUST_REGION_F and tribe_config.VALID_DIVERGENCES if this set ever changes.
TRUST_REGION_F = {
    "kl_new_old": lambda rho: rho * torch.log(rho),
    "kl_old_new": lambda rho: -torch.log(rho),
    "chi_squared": lambda rho: 0.5 * (rho - 1) ** 2,
}


def _constraint_value(A: Tensor, c: Tensor, beta: float, log_lambda: Tensor) -> tuple[Tensor, Tensor]:
    """
    Args:
        A, c: (G, K) per-sample advantage and reference log-ratio.
        log_lambda: scalar tensor, log(lambda).

    Returns:
        Tuple of `(log_rho, f_val)`:
            - `log_rho` (`Tensor` of shape `(G, K)`): log rho* at this lambda.
            - `f_val` (`Tensor` scalar): mean_{x,y}[rho* * log(rho*)], the trust-region constraint value.
    """
    lam = torch.exp(log_lambda)
    K = A.size(1)
    logits = (A - beta * c) / lam  # (G, K)
    log_Z = torch.logsumexp(logits, dim=1, keepdim=True) - math.log(K)  # (G, 1), per-context normalizer
    log_rho = logits - log_Z  # (G, K)
    f_val = (torch.exp(log_rho) * log_rho).mean()
    return log_rho, f_val


def solve_lambda(
    A: Tensor,
    c: Tensor,
    beta: float,
    eps: float,
    log_lambda_init: tuple[float, float] = (-10.0, 10.0),
    tol: float = 1e-8,
    max_iter: int = 100,
) -> Tensor:
    """
    Bisection root-find for lambda in log-space against the global trust-region constraint.

    Args:
        A (`Tensor` of shape `(G, K)`):
            Per-sample advantage, grouped by context.
        c (`Tensor` of shape `(G, K)`):
            Per-sample reference log-ratio `log(pi_old/pi_ref)`, grouped by context.
        beta (`float`):
            Weight on the reference term.
        eps (`float`):
            Trust-region budget (target constraint value).
        log_lambda_init (`tuple[float, float]`, *optional*, defaults to `(-10.0, 10.0)`):
            Initial bracket for `log(lambda)`, expanded outward if it doesn't already bracket the root.
        tol (`float`, *optional*, defaults to `1e-8`):
            Bisection stops once the bracket width in log-space is below this.
        max_iter (`int`, *optional*, defaults to `100`):
            Maximum number of bracket-expansion and bisection steps (each bounded separately).

    Returns:
        `Tensor` scalar: the solved `lambda`.
    """
    lo, hi = log_lambda_init
    lo = torch.tensor(lo, dtype=A.dtype, device=A.device)
    hi = torch.tensor(hi, dtype=A.dtype, device=A.device)

    # Numeric floor/ceiling for the expansion below. If (A - beta*c) is exactly constant across the
    # WHOLE batch (e.g. every group has zero reward variance, which alone already makes A == 0
    # everywhere, AND pi_old == pi_ref, which holds exactly at the very first training step, making
    # c == 0 everywhere too), f_val is identically 0 for every lambda and can never reach eps. Without a
    # bound, the loop below would keep expanding lo forever (up to max_iter), driving log_lambda so
    # negative that exp(lo) underflows to exact 0.0, which turns `logits = (A - beta*c) / lam` into a
    # 0/0 NaN. Clamping keeps lambda in a range where exp() is always finite and nonzero; in the
    # degenerate case this correctly settles on rho* == 1 everywhere (this batch carries no informative
    # Stage-1 signal, so leave the policy where it already is) instead of propagating NaN.
    safe_log_lambda_bound = 50.0

    # f_val is monotonically decreasing in log_lambda: expand the bracket outward until
    # f_val(lo) >= eps >= f_val(hi).
    for _ in range(max_iter):
        if _constraint_value(A, c, beta, lo)[1] >= eps or lo <= -safe_log_lambda_bound:
            break
        lo = lo - 10.0
    lo = torch.clamp(lo, min=-safe_log_lambda_bound)
    for _ in range(max_iter):
        if _constraint_value(A, c, beta, hi)[1] <= eps or hi >= safe_log_lambda_bound:
            break
        hi = hi + 10.0
    hi = torch.clamp(hi, max=safe_log_lambda_bound)

    for _ in range(max_iter):
        if (hi - lo) < tol:
            break
        mid = (lo + hi) / 2
        f_mid = _constraint_value(A, c, beta, mid)[1]
        if f_mid > eps:
            lo = mid  # constraint value too high => lambda too small => raise the lower bound
        else:
            hi = mid

    return torch.exp((lo + hi) / 2)


def _solve_chi_squared_beta0_closed_form(A: Tensor, group_size: int, eps: float) -> tuple[Tensor, Tensor] | None:
    """
    Closed form for `divergence="chi_squared"` in the special case `beta == 0`: with no reference term,
    the objective `sum p_i*rho_i*A_i` is LINEAR in `rho`, so the KKT stationarity condition for
    `f(rho) = 0.5*(rho-1)^2` is `rho_i = 1 + (A_i - b_x) / lambda` for context `x`, where `b_x` is the
    per-group Lagrange multiplier of the `sum_{i in x} rho_i == K` equality constraint. Substituting into
    that constraint gives `b_x = mean_{i in x}(A_i)` exactly (independent of lambda), so this reduces to a
    single global scalar solve:

        rho_i = 1 + (A_i - group_mean(A)_x) / lambda
        lambda = sqrt(mean_i[(A_i - group_mean(A)_x)^2] / (2*eps))

    with the global chi-squared constraint plugged in directly (no iteration/bisection at all, unlike
    every other divergence/beta combination in this module). See tests/test_stage1.py's
    test_chi_squared_beta0_closed_form_matches_cvxpy for the derivation validated numerically against
    `validate_stage1.solve_stage1_cvxpy`.

    `validate_stage1.solve_stage1_cvxpy` additionally constrains `rho >= 0` (`cp.Variable(..., nonneg=True)`)
    -- a constraint this formula doesn't enforce. Returns `None` (caller must fall back to the general
    cvxpy solve) if that constraint would actually bind for this batch (any `rho_i < 0`), since a binding
    nonnegativity constraint changes which KKT system applies and this single formula is no longer exact.

    Args:
        A (`Tensor` of shape `(N,)`):
            Per-sample advantage, flattened over `N = G * group_size` samples (need not already be
            group-mean-centered -- this function centers it itself).
        group_size (`int`):
            Number of samples `K` per context (prompt group).
        eps (`float`):
            Trust-region budget.

    Returns:
        `(rho, lam)` (both matching `A`'s dtype), or `None` if infeasible (see above) or the batch has
        zero advantage variance (nothing for this closed form to do; let the caller's existing
        zero-variance handling apply instead).
    """
    G = A.numel() // group_size
    A_grouped = A.float().view(G, group_size)
    A_centered = A_grouped - A_grouped.mean(dim=1, keepdim=True)
    mean_sq = (A_centered**2).mean()
    if mean_sq <= 0:
        return None
    lam = torch.sqrt(mean_sq / (2 * eps))
    rho = (1 + A_centered / lam).reshape(-1)
    if (rho < 0).any():
        return None
    return rho.to(A.dtype), lam.to(A.dtype)


def compute_rho_star(
    A: Tensor,
    c: Tensor,
    beta: float,
    group_size: int,
    eps: float,
    divergence: str = "kl_new_old",
    use_chi_squared_beta0_closed_form: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Solve TRIBE Stage 1: `rho*` and its single scalar `lambda`.

    Args:
        A (`Tensor` of shape `(N,)`):
            Per-sample advantage, flattened over `N = G * group_size` samples.
        c (`Tensor` of shape `(N,)`):
            Per-sample reference log-ratio `log(pi_old/pi_ref)`, flattened the same way as `A`.
        beta (`float`):
            Weight on the reference term.
        group_size (`int`):
            Number of samples `K` per context (prompt group).
        eps (`float`):
            Trust-region budget.
        divergence (`str`, *optional*, defaults to `"kl_new_old"`):
            Trust-region divergence `f` in `E_old[f(rho)] <= eps`, named by explicit argument order so
            there is nothing to misremember (see `validate_stage1._TRUST_REGION_F`). `"kl_new_old"`
            (`KL(pi || pi_old)`) is TRIBE's own choice and uses the closed form derived in
            TRIBE-plan.tex (fast, exact, validated against cvxpy in `tests/test_stage1.py`). Any other
            divergence — including `"kl_old_new"` (`KL(pi_old || pi)`, TRPO's own convention) and
            `"chi_squared"` — has no closed form here, see this module's docstring for why, and is solved
            directly via cvxpy instead, which is exact but far slower and CPU-only; expect this path to
            be impractical at full training-batch sizes. Exception: `"chi_squared"` with `beta == 0` AND
            `use_chi_squared_beta0_closed_form=True` — see `_solve_chi_squared_beta0_closed_form`.
        use_chi_squared_beta0_closed_form (`bool`, *optional*, defaults to `False`):
            Opt-in fast path for `divergence="chi_squared"`, `beta == 0` batches — skips cvxpy entirely
            (see `_solve_chi_squared_beta0_closed_form`). Falls back to the general cvxpy solve below,
            unchanged, whenever that closed form reports infeasibility (some `rho_i` would be negative) or
            this flag is `False` / the batch doesn't match (`divergence != "chi_squared"` or `beta != 0`).
            Gated behind this flag rather than always-on because it's new, less battle-tested code than
            the existing cvxpy path.

    Returns:
        Tuple of `(rho, lam)`:
            - `rho` (`Tensor` of shape `(N,)`, or `None`): the solved `rho*`, or `None` if the cvxpy path
              (any divergence other than `"kl_new_old"`) failed to solve this batch (status != "optimal";
              see the cvxpy branch below) — callers must skip the batch when this is `None`, not treat it
              as a value to use.
            - `lam` (`Tensor` scalar, or `None`): the solved `lambda`, `None` under the same condition as
              `rho`. For `divergence="kl_new_old"` this already has `beta` folded in (see the module
              docstring's KKT note); for any other divergence it is the trust-region constraint's raw dual
              value, with no such recombination.
    """
    if divergence == "kl_new_old":
        G = A.numel() // group_size
        A_grouped = A.view(G, group_size)
        c_grouped = c.view(G, group_size)
        lam = solve_lambda(A_grouped, c_grouped, beta, eps)
        log_rho, _ = _constraint_value(A_grouped, c_grouped, beta, torch.log(lam))
        return torch.exp(log_rho).reshape(-1), lam

    if use_chi_squared_beta0_closed_form and divergence == "chi_squared" and beta == 0.0:
        result = _solve_chi_squared_beta0_closed_form(A, group_size, eps)
        if result is not None:
            return result
        # Infeasible (nonnegativity would bind) or zero-variance -- fall through to the general cvxpy
        # solve below, same as if this flag were off.

    from .validate_stage1 import solve_stage1_cvxpy  # local import: cvxpy is only needed on this path

    rho, lam, status = solve_stage1_cvxpy(
        A.detach().float().cpu().numpy(), c.detach().float().cpu().numpy(), beta, group_size, eps, divergence=divergence
    )
    if status == "optimal_inaccurate":
        # The solver ran out of iterations/precision before certifying full convergence but still
        # returned a point that approximately satisfies the KKT/trust-region constraints — a usable
        # solution in practice, not a failure to solve (unlike "infeasible"/"unbounded"/"solver_error",
        # which mean the problem itself is degenerate for this batch). Accept it and use the values as
        # normal; only warn (with the status, and the "UsageWarning" category so callers can specifically
        # catch-and-count this case, e.g. OffPolicyTribeTrainer.compute_loss's own inaccurate-count
        # logging) rather than skip a perfectly usable batch over a numerical-tolerance technicality.
        warnings.warn(
            f"cvxpy Stage 1 solve for divergence={divergence!r} returned 'optimal_inaccurate' — using the "
            "approximate solution.",
            category=UserWarning,
            stacklevel=2,
        )
    elif status != "optimal":
        # Genuinely bad status (infeasible/unbounded/solver_error/...) — the problem itself couldn't be
        # solved for this batch's reward configuration, not just imprecisely. Signal this to the caller by
        # returning (None, None) instead of raising, so an otherwise-fine training run can skip this one
        # batch rather than crash (see OffPolicyTribeTrainer.compute_loss's own skip-counting).
        warnings.warn(
            f"cvxpy Stage 1 solve for divergence={divergence!r} returned status {status!r} (not "
            "'optimal' or 'optimal_inaccurate') — skipping this batch.",
            stacklevel=2,
        )
        return None, None
    return (
        torch.as_tensor(rho, dtype=A.dtype, device=A.device),
        torch.as_tensor(lam, dtype=A.dtype, device=A.device),
    )


def filter_zero_variance_groups(rewards: Tensor, group_size: int, atol: float = 1e-8) -> Tensor:
    """
    Identify groups whose reward is constant across all `K` samples (zero-variance groups).

    Per TRIBE-plan.tex Section 2.3: for these groups the reward term collapses out of the Stage-1
    objective via the normalization constraint, leaving an update driven purely by the reference term.
    They don't need to be excluded from `compute_rho_star` for correctness — their `A` is exactly `0`, so
    the reward term washes out of the objective on its own, same as GRPO trains on every group regardless
    of its advantage. This mask is for the *optional* backfill mechanism (replacing them with
    freshly-sampled prompts, trainer-level, so they're less often reward-uninformative in the first
    place) and for monitoring, not for gating the Stage-1 solve itself.

    Args:
        rewards (`Tensor` of shape `(N,)`):
            Per-sample reward, flattened over `N = G * group_size` samples.
        group_size (`int`):
            Number of samples `K` per context (prompt group).
        atol (`float`, *optional*, defaults to `1e-8`):
            Absolute tolerance for treating the group standard deviation as zero.

    Returns:
        `Tensor` of shape `(G,)`: boolean mask, `True` for groups with nonzero reward variance.
    """
    G = rewards.numel() // group_size
    grouped = rewards.view(G, group_size)
    return grouped.std(dim=1) > atol
