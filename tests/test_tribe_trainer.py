import torch
from datasets import load_dataset

from tribe.tribe_config import TribeConfig
from tribe.tribe_trainer import TribeTrainer


def _parity_reward(completions, **kwargs):
    """Toy verifiable (binary) reward: deterministic given the completion, mimics GSM8K/MATH correctness."""
    return [float(len(c) % 2 == 0) for c in completions]


def test_tribe_trainer_smoke(tmp_path):
    dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

    training_args = TribeConfig(
        output_dir=str(tmp_path),
        learning_rate=0.1,
        per_device_train_batch_size=6,
        num_generations=3,
        max_completion_length=8,
        beta=0.1,
        trust_region_eps=0.05,
        report_to="none",
        max_steps=2,
        logging_steps=1,
    )
    trainer = TribeTrainer(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        reward_funcs=_parity_reward,
        args=training_args,
        train_dataset=dataset,
    )

    previous_trainable_params = {n: p.clone() for n, p in trainer.model.named_parameters()}

    trainer.train()

    loss_history = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    assert len(loss_history) > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in loss_history)

    # `tribe/lambda` can legitimately log as None for a step where every group in the batch was
    # zero-variance (see filter_zero_variance_groups): GRPOTrainer.log() filters NaNs before averaging
    # and reports None when nothing valid remains for that window.
    lambda_history = [
        log["tribe/lambda"] for log in trainer.state.log_history if log.get("tribe/lambda") is not None
    ]
    assert len(lambda_history) > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in lambda_history)

    for n, p in previous_trainable_params.items():
        new_p = trainer.model.get_parameter(n)
        assert not torch.equal(p, new_p), f"Parameter {n} has not changed."
