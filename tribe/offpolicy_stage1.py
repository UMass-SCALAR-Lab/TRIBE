"""
Off-policy adaptation of TRIBE's Stage 1 (tribe.stage1) for the single-iteration, fully-offline regime: a
fixed dataset of `group_size` completions per question, generated once from a frozen behavioral policy mu
(the untrained base model), labeled with reward, and trained for a single epoch with no regeneration —
TOPR's own primary experimental setup (https://huggingface.co/papers/2503.14286, "single-iteration, fully
offline regime").

The only thing that changes from on-policy TribeTrainer is how A and c are obtained — Stage 1's own
closed-form solve (tribe.stage1.compute_rho_star) is dataset-shape-agnostic and reused as-is, unmodified.

c = log(pi_old/pi_ref) is identically 0 here, not just at step 0: pi_old (mu, the behavioral policy) and
pi_ref (the KL anchor) are both the same frozen base-model checkpoint for the entire single epoch — neither
one ever changes, so their log-ratio is exactly 0 for every sample throughout training (unlike on-policy
TRIBE, where pi_old is refreshed every cycle and only coincides with pi_ref at step 0). This is NOT the
degenerate "A == 0 AND c == 0" case tribe.stage1.solve_lambda's numeric safety clamp guards against (that
additionally requires zero reward variance across the WHOLE batch) — with genuine reward variance present
(the normal case, since group_size=16/32 completions per question will usually include a mix of
correct/incorrect), Stage 1 reduces cleanly and stably to rho* = softmax_K(A/lambda), a trust-region
reweighted advantage softmax with beta's reference-anchoring term contributing exactly 0 throughout (not
skipped or special-cased — see tests/test_stage1.py's test_reps_reduction_when_c_is_zero, which already
validates this exact reduction against an independent REPS/MPO reference implementation).
"""

import torch
from torch import Tensor

from .stage1 import compute_rho_star


def compute_rho_star_offpolicy(
    rewards: Tensor,
    group_size: int,
    beta: float,
    eps: float,
    divergence: str = "kl_new_old",
    use_chi_squared_beta0_closed_form: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Stage 1 for the off-policy, single-iteration regime: `c` is fixed at 0 (`pi_old == pi_ref`, both the
    frozen behavioral policy, for the whole single epoch), so only the reward-derived advantage `A` drives
    `rho*` through the `-beta*c` cross term.

    Args:
        rewards (`Tensor` of shape `(N,)`):
            Per-completion reward, flattened over `N = num_questions * group_size` completions, grouped by
            question in contiguous blocks of `group_size` (same layout `compute_rho_star` expects for `A`).
        group_size (`int`):
            Number of completions per question (`n=16` for GSM8K, `n=32` for MATH, per TOPR's own setup).
        beta (`float`):
            Weight on the reference term in Stage 1's closed form. The `-beta*c` cross term vanishes since
            `c` is always 0 here, but Stage 1's objective also has a separate `+beta*entr(rho)` entropy-
            regularization term (see `validate_stage1.solve_stage1_cvxpy`'s own docstring) that does NOT
            depend on `c` at all — for `divergence="kl_new_old"` this term happens to have the same
            functional shape as that divergence's own trust-region penalty and folds into an equivalent
            re-scaling of `lambda`, making `beta` a genuine no-op end to end (see
            `tests/test_stage1.py::test_reps_reduction_when_c_is_zero`); for any other divergence
            (`"chi_squared"` in particular) the two terms have different shapes and do NOT cancel, so
            `beta` actively changes `rho*` even though `c == 0`.
        eps (`float`):
            Trust-region budget, same meaning as `TribeConfig.trust_region_eps`.
        divergence (`str`, *optional*, defaults to `"kl_new_old"`):
            See `tribe.stage1.compute_rho_star` for what this controls.
        use_chi_squared_beta0_closed_form (`bool`, *optional*, defaults to `False`):
            Forwarded to `tribe.stage1.compute_rho_star` as-is — see its own docstring.

    Returns:
        Tuple of `(rho, lam)`, same as `tribe.stage1.compute_rho_star`.
    """
    G = rewards.numel() // group_size
    rewards_grouped = rewards.view(G, group_size)
    A = (rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)).reshape(-1)
    c = torch.zeros_like(A)
    return compute_rho_star(
        A,
        c,
        beta,
        group_size,
        eps,
        divergence=divergence,
        use_chi_squared_beta0_closed_form=use_chi_squared_beta0_closed_form,
    )
