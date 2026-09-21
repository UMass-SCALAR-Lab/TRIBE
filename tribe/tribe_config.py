import math
from dataclasses import dataclass, field

from trl.trainer.grpo_config import GRPOConfig


# Kept as a plain literal (not imported from validate_stage1._TRUST_REGION_F) so constructing a
# TribeConfig with the default divergence doesn't pull in cvxpy, which is only needed by non-default
# divergences (see tribe.stage1's module docstring for why "kl_new_old" is the only one with a closed
# form). Keep in sync with validate_stage1._TRUST_REGION_F if that set ever changes.
VALID_DIVERGENCES = ("kl_new_old", "kl_old_new", "chi_squared")

# Stage 2 loss aggregation over completion tokens:
#   - "sum": full-sequence sum of log-probs, TRIBE-plan.tex Section 2.2's literal M-projection quantity
#     (log pi(y|x) is a sum over tokens by the chain rule). Length-dependent gradient scale.
#   - "grpo": per-sequence token average, matching GRPOTrainer's own default ("grpo") loss_type. Length
#     invariant, but changes each sample's relative weight vs "sum" whenever completions differ in length.
#   - "dapo": batch-level token-count normalization, matching GRPOTrainer's "dapo" loss_type (each *token*
#     gets equal weight across the batch, rather than each *sequence*).
VALID_STAGE2_LOSS_TYPES = ("sum", "grpo", "dapo")


@dataclass
class TribeConfig(GRPOConfig):
    """
    Configuration for [`TribeTrainer`].

    See TRIBE-plan.tex for the derivation. `beta` (inherited from [`~trl.GRPOConfig`]) is reused as the
    weight on the reference term in Stage 1's closed form; `trust_region_eps` and `divergence` are the new
    quantities: `trust_region_eps` is the trust-region budget `epsilon` that Stage 1's `lambda` is
    root-found against, and `divergence` is the `f` in the trust-region constraint `E_old[f(rho)] <= eps`.

    > Parameters for Stage 1:

    Args:
        scale_rewards (`str`, *optional*, defaults to `"none"`):
            Overrides [`~trl.GRPOConfig`]'s own default (`"group"`) to `"none"`, matching Stage 1's own
            derivation of `A(x,y) = R(x,y) - mean_K(R)` as the raw group-relative advantage. `"group"` and
            `"batch"` remain fully valid if passed explicitly — see the field's own `help` text for why
            `"none"` is preferred by default.
        trust_region_eps (`float`, *optional*, defaults to `0.05`):
            Trust-region budget `epsilon`: the target value of `mean[rho* * log(rho*)]` that Stage 1's
            root-find for `lambda` solves against.
        divergence (`str`, *optional*, defaults to `"kl_new_old"`):
            Trust-region divergence `f`, named by explicit argument order (see
            `tribe.stage1.compute_rho_star`'s docstring for why, instead of "forward"/"reverse"): one of
            `"kl_new_old"` (`KL(pi || pi_old)`, TRIBE's own choice, fast closed form), `"kl_old_new"`
            (`KL(pi_old || pi)`, TRPO's own convention), or `"chi_squared"`. Anything other than
            `"kl_new_old"` is solved via cvxpy per batch, which is exact but far slower and CPU-only —
            expect this to be impractical at full training-batch sizes.
        backfill_zero_variance_prompts (`bool`, *optional*, defaults to `False`):
            Whether to replace zero-variance groups (TRIBE-plan.tex Section 2.3) with freshly-sampled
            prompts and regenerate, so fewer groups end up reward-uninformative in the first place. Off
            by default: without it, such groups are simply included in the joint Stage-1 solve like any
            other group (their `A` is exactly `0`, so the reward term washes out on its own, same as
            GRPO trains on every group regardless of its own advantage). See `max_backfill_attempts` for
            how many rounds are allowed once enabled.
        max_backfill_attempts (`int`, *optional*, defaults to `4`):
            Number of extra generation rounds (per training step) to replace zero-variance groups with
            freshly-sampled prompts, when `backfill_zero_variance_prompts` is enabled. Has no effect
            otherwise.
        stage2_loss_type (`str`, *optional*, defaults to `"sum"`):
            How Stage 2's `(rho*-1)`-weighted log-prob loss aggregates over completion tokens: `"sum"`
            (full-sequence sum, TRIBE-plan.tex Section 2.2's literal quantity, but length-dependent
            gradient scale), `"grpo"` (per-sequence token average, matching `GRPOTrainer`'s own default
            loss type), or `"dapo"` (batch-level token-count normalization, matching `GRPOTrainer`'s
            `"dapo"` loss type). `"grpo"` and `"dapo"` depart from the literal M-projection quantity
            whenever completions in a batch have different lengths — see the module-level comment next
            to `VALID_STAGE2_LOSS_TYPES` for why.

    > Parameters for Stage 2:

    Args:
        negative_logp_floor (`float`, *optional*, defaults to `log(1e-8)`):
            Floors per-token log-probabilities at this value before Stage 2's `(rho*-1)`-weighted loss,
            bounding the gradient magnitude on negative-weight (`rho*<1`) tokens instead of unbounded
            `-logp` pressure (see TRIBE-plan.tex's "Open question" paragraph). Matches the off-policy
            trainer's own default. Set to `None` to disable and fall back to the plain unbounded loss —
            on-policy training runs far fewer, more homogeneous cycles than the offline setting this floor
            was validated on, so it's less clearly load-bearing here; kept easy to turn off for ablation.
        negative_fraction (`float`, *optional*, defaults to `1.0`):
            What fraction of each group's negative-weight (`rho*<1`) examples participate in the loss
            (positives are never gated). `1.0` (default) keeps every negative, matching this trainer's
            original behavior. `0.0` drops all negatives (positive-only). Values `>1.0` oversample
            negatives instead. Ported from the off-policy trainer's own `negative_fraction` ablation —
            see [`~scripts.offpolicy_trainer.OffPolicyTrainer._negative_sample_weight_global`]'s docstring
            for the exact subsampling/oversampling mechanics, which this reuses unchanged.
    """

    scale_rewards: str = field(
        default="none",
        metadata={
            "help": "Overrides GRPOConfig's own default ('group') to 'none': Stage 1's closed form is "
            "derived around A(x,y) = R(x,y) - mean_K(R), the raw group-relative advantage (see "
            "TRIBE-plan.tex); 'group' divides each group by its own std, a per-group rescaling that Stage "
            "1's single shared lambda (solved jointly across every group in the batch) can't cleanly "
            "absorb the way it can a single global rescale. Still fully supported if you explicitly pass "
            "'group' or 'batch' — this only changes the default, not what's allowed."
        },
    )
    trust_region_eps: float = field(
        default=0.05,
        metadata={
            "help": "Trust-region budget epsilon: the target value of mean[rho* * log(rho*)] that Stage "
            "1's root-find for lambda solves against."
        },
    )
    divergence: str = field(
        default="kl_new_old",
        metadata={
            "help": "Trust-region divergence f in E_old[f(rho)] <= eps. One of 'kl_new_old' (TRIBE's own "
            "choice, fast closed form), 'kl_old_new' (TRPO's own convention), or 'chi_squared'. Anything "
            "other than 'kl_new_old' is solved via cvxpy per batch (exact but far slower, CPU-only)."
        },
    )
    backfill_zero_variance_prompts: bool = field(
        default=False,
        metadata={
            "help": "Replace zero-variance groups with freshly-sampled prompts and regenerate, so fewer "
            "groups end up reward-uninformative. Off by default."
        },
    )
    max_backfill_attempts: int = field(
        default=4,
        metadata={
            "help": "Number of extra generation rounds per training step to replace zero-variance groups "
            "with freshly-sampled prompts, when backfill_zero_variance_prompts is enabled."
        },
    )
    stage2_loss_type: str = field(
        default="sum",
        metadata={
            "help": "How Stage 2's loss aggregates over completion tokens: 'sum' (full-sequence sum, the "
            "literal M-projection quantity, length-dependent gradient scale), 'grpo' (per-sequence token "
            "average, matching GRPOTrainer's own default), or 'dapo' (batch-level token-count "
            "normalization, matching GRPOTrainer's 'dapo' loss type)."
        },
    )
    negative_logp_floor: float | None = field(
        default=math.log(1e-8),
        metadata={
            "help": "Floor per-token log-probs at this value before Stage 2's (rho*-1)-weighted loss, "
            "bounding gradient magnitude on negative-weight tokens. Matches the off-policy trainer's own "
            "default. Set to None to disable (plain unbounded loss, this trainer's original behavior)."
        },
    )
    negative_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of each group's negative-weight (rho*<1) examples that participate in the "
            "loss (positives are never gated). 1.0 (default) keeps every negative, matching this trainer's "
            "original behavior. 0.0 drops all negatives (positive-only). Values >1.0 oversample negatives."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.divergence not in VALID_DIVERGENCES:
            raise ValueError(f"Unknown divergence {self.divergence!r}. Must be one of {VALID_DIVERGENCES}.")
        if self.stage2_loss_type not in VALID_STAGE2_LOSS_TYPES:
            raise ValueError(
                f"Unknown stage2_loss_type {self.stage2_loss_type!r}. Must be one of {VALID_STAGE2_LOSS_TYPES}."
            )
        if self.multi_objective_aggregation != "sum_then_normalize":
            raise ValueError(
                f"TribeTrainer requires `multi_objective_aggregation='sum_then_normalize'`, got "
                f"{self.multi_objective_aggregation!r}: 'normalize_then_sum' always std-normalizes, which would "
                "depart from the raw advantage Stage 1 expects."
            )
        if self.negative_fraction < 0:
            raise ValueError(f"negative_fraction must be non-negative, got {self.negative_fraction!r}.")
