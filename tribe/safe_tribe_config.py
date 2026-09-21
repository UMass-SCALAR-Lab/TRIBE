from dataclasses import dataclass, field

from .tribe_config import TribeConfig


@dataclass
class SafeTribeConfig(TribeConfig):
    """
    Configuration for [`SafeTribeTrainer`].

    Adds new fields on top of [`TribeConfig`]'s reward-only Stage 1 fields (`trust_region_eps`,
    `divergence`, `beta`, ...), all inherited unchanged. See `tribe.stage1_safe.solve_stage1_safe_cvxpy`
    for the exact constrained program the default (hard-constraint) mode solves.

    > Parameters for Stage 1 (in addition to TribeConfig's):

    Args:
        cost_limit (`float`, *optional*, defaults to `0.0`):
            Upper bound on `E_old[rho * Cost]`, the expected cost of the reweighted policy under `pi_old`.
            The cost signal itself comes from the LAST entry of `reward_funcs` passed to
            [`SafeTribeTrainer`] (see that class's own docstring for the `reward_funcs`/`reward_weights`
            convention this requires). Defaults to `0.0`, matching Safe-RLHF's own framing of "no worse
            than the current policy, on average" when the cost signal is a group-centered advantage (zero
            mean at `rho == 1`); pass a different value if using a raw (uncentered) cost-model score
            instead.
        soft_cost_constraint (`bool`, *optional*, defaults to `False`):
            If `False` (default): solve Stage 1 as a hard-constrained convex program
            (`tribe.stage1_safe.compute_rho_star_safe`) — if `cost_limit` is infeasible given the current
            trust region, that batch falls back to `rho == 1` (no Stage-1 signal at all that step; see
            [`SafeTribeTrainer`]'s docstring). If `True`: never solve a hard cost constraint at all.
            Instead, fold the cost into the reward advantage via a persistent Lagrange multiplier updated
            by dual ascent every step (`lambda <- clip(lambda + lambda_lr * (batch_cost_mean - cost_limit),
            0, lambda_max)`, matching Safe-RLHF's own PPO-Lag convention), then solve the ordinary
            unconstrained TRIBE Stage 1 (`tribe.stage1.compute_rho_star`) on the adjusted advantage
            `A - lambda * Cost`. This can never be infeasible (the underlying solve is always
            unconstrained-by-cost), at the cost of the cost limit only being enforced on average over
            training, not exactly satisfied every single batch.
        lambda_init (`float`, *optional*, defaults to `1.0`):
            Initial value of the dual-ascent Lagrange multiplier. Only used when `soft_cost_constraint`.
        lambda_lr (`float`, *optional*, defaults to `0.01`):
            Dual-ascent step size for the Lagrange multiplier. Only used when `soft_cost_constraint`.
        lambda_max (`float`, *optional*, defaults to `100.0`):
            Upper clip on the Lagrange multiplier. Only used when `soft_cost_constraint`.
    """

    cost_limit: float = field(
        default=0.0,
        metadata={
            "help": "Upper bound on E_old[rho * Cost] (the last entry of reward_funcs). 0.0 matches "
            "Safe-RLHF's own 'no worse than the current policy, on average' framing for a group-centered "
            "cost advantage; use a different value for a raw, uncentered cost-model score."
        },
    )
    soft_cost_constraint: bool = field(
        default=False,
        metadata={
            "help": "If set, never hard-fail on an infeasible cost constraint. Instead fold cost into the "
            "advantage via a persistent dual-ascent Lagrange multiplier (Safe-RLHF's own PPO-Lag "
            "convention) and solve the ordinary unconstrained TRIBE Stage 1 on A - lambda * Cost. Default "
            "(False) keeps the existing hard-constrained cvxpy solve unchanged."
        },
    )
    lambda_init: float = field(
        default=1.0, metadata={"help": "Initial dual-ascent Lagrange multiplier. Only used with soft_cost_constraint."}
    )
    lambda_lr: float = field(
        default=0.01, metadata={"help": "Dual-ascent step size for the Lagrange multiplier. Only used with soft_cost_constraint."}
    )
    lambda_max: float = field(
        default=100.0, metadata={"help": "Upper clip on the Lagrange multiplier. Only used with soft_cost_constraint."}
    )
