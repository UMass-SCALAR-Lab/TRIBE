# TRIBE: Trust-Region Iterative Behavior-cloning Estimator

TRIBE is a two-stage RL-from-preferences method: Stage 1 solves a closed-form (or cvxpy) trust-region
reweighting of a group's completions against their reward advantage; Stage 2 trains on the reweighted
completions with a `(rho*-1)`-weighted behavior-cloning loss. This repo implements TRIBE — on-policy,
fully-offline/off-policy, and cost-constrained (Safe-TRIBE) — alongside baselines (GRPO, DAPO, RLOO, RAFT,
TOPR, TIS, DPO, SimPO, PPO-Lag, ...) for direct comparison, on GSM8K, MATH, UltraFeedback, and
BeaverTails/PKU-SafeRLHF.

## Installation

Three separate conda environments are used — don't mix them; installing vLLM/newer transformers into the
training env (or vice versa) has broken things before (see `scripts/overlay_base_config.py`'s docstring on
the RoPE-config writer/reader version-mismatch failure mode this project has hit).

**`tribe`** (training — GRPO/TRIBE/DAPO/RLOO/RAFT/DPO/SimPO/PPO-Lag-style trainers, deepspeed, cvxpy):

```bash
conda create -n tribe python=3.11
conda activate tribe
pip install -e ./trl   # this repo's own TRL fork — training scripts import trl.GRPOTrainer etc. from here
pip install -e .       # installs the `tribe` package (tribe/stage1.py, tribe/tribe_trainer.py, ...)
pip install deepspeed cvxpy
```

**`llm_gen`** (vLLM-based generation/eval — GSM8K/MATH/TACO/KodCode eval scripts, offline-data generation):

```bash
conda create -n llm_gen python=3.10
conda activate llm_gen
pip install vllm==0.6.4.post1 transformers==4.45.2 datasets math-verify latex2sympy2_extended pytest
```

**`llm_gen_new`** (only needed for architectures `llm_gen`'s pinned transformers doesn't recognize yet,
e.g. Gemma-3, Qwen3 — see `scripts/eval_kodcode.py`/`scripts/eval_taco.py`'s usage for when this matters):

```bash
conda create -n llm_gen_new python=3.11
conda activate llm_gen_new
pip install vllm datasets latex2sympy2_extended math_verify pytest
```
If a job hits `RuntimeError: ... libcudart.so.12` under this env, see
`slurm/run_eval_gsm8k_newenv.sbatch`'s comment (torch here is built against CUDA 13; the cluster's
`module load cuda/12.6` doesn't put its lib dir on `LD_LIBRARY_PATH` by default).

## Repository layout

- `tribe/` — the core method: `stage1.py`/`stage1_safe.py`/`offpolicy_stage1.py` (the trust-region solves),
  `tribe_trainer.py`/`safe_tribe_trainer.py` (on-policy trainers, subclass `GRPOTrainer`),
  `safe_tribe_config.py`/`tribe_config.py`.
- `scripts/` — one `train_<task>_<method>.py` (on-policy) or `train_<task>_offpolicy_<method>.py`
  (off-policy) entry point per task/method combination — deliberately duplicated, not shared base classes
  (see `trl/CLAUDE.md`'s "Code duplication and consistency" section for why). Also: `eval_*.py` (GSM8K/
  MATH/TACO/KodCode eval via vLLM), `generate_offpolicy_*.py`/`convert_*_offpolicy.py` (build the fixed
  offline datasets off-policy training reads), `offpolicy_trainer.py`/`offpolicy_split.py` (shared
  off-policy trainer classes + the train/val split every off-policy script uses).
- `configs/` — algorithm hyperparameters as `TrlParser`-loadable YAML (`--config configs/.../<file>.yaml`),
  one file per method; dataset-dependent flags (group_size, batch size, `--dataset_path`, `--output_dir`,
  `--deepspeed`) are passed via CLI at launch, not baked into the YAML — see any file's own header comment.
  Also holds the deepspeed JSON configs (`deepspeed_zero2.json`, `deepspeed_zero3.json`, ...).
- `slurm/` — SLURM launch scripts. `run_experiment.sbatch`/`run_eval_gsm8k.sbatch`/
  `run_eval_gsm8k_newenv.sbatch` are generic one-job-per-`--export=...,TRAIN_CMD=...` launchers shared
  across every experiment; `run_final_*_suite.sh` submit a full method x seed sweep for one task.
- `tests/` — unit tests for the Stage 1 solves (`pytest tests/`, run in the `tribe` env).

## Running an experiment

Every training script takes a config file plus CLI overrides, e.g. on-policy TRIBE on GSM8K:

```bash
conda activate tribe
accelerate launch --num_processes=2 scripts/train_gsm8k.py \
    --config configs/onpolicy/tribe.yaml \
    --dataset_train_split 'train[:6726]' --num_generations 8 \
    --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --max_completion_length 512 \
    --output_dir <your_output_dir> --answer_format boxed --seed 42
```

Off-policy training reads a FIXED, pre-generated dataset (never regenerates mid-epoch) — build it once,
then train for a single epoch over it:

```bash
python scripts/generate_offpolicy_gsm8k.py --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --num_generations 16 --answer_format boxed --output_dir <offline_data_dir>   # llm_gen env

conda activate tribe
accelerate launch --num_processes=2 scripts/train_gsm8k_offpolicy_tribe.py \
    --config configs/offpolicy/tribe.yaml \
    --dataset_path <offline_data_dir> --group_size 16 --per_device_train_batch_size 8 \
    --output_dir <your_output_dir>
```

UltraFeedback's off-policy data doesn't need generating — it already ships 4 scored completions/prompt —
just convert it once (`scripts/convert_ultrafeedback_offpolicy.py`), then train with
`configs/offpolicy/ultrafeedback_tribe.yaml` the same way. `reward` is the average of each completion's 4
fine-grained per-aspect ratings (helpfulness/honesty/instruction_following/truthfulness), NOT the raw
`overall_score` field — the two disagree on which of a prompt's 4 completions is best ~45% of the time (see
the script's own docstring); `overall_score` is the same known-unreliable field HuggingFace's own
`ultrafeedback_binarized` avoids for the same reason.

Two more off-policy UltraFeedback baselines, alongside RAFT/TRIBE:
`scripts/train_ultrafeedback_offpolicy_rloo.py` (RLOO leave-one-out baseline, `OffPolicyReinforceTrainer`,
no importance-sampling correction) and `scripts/train_ultrafeedback_offpolicy_simpo.py` (reference-free
SimPO via `trl.experimental.cpo.CPOTrainer` with `loss_type="simpo"`, on pairs built by
`scripts/build_ultrafeedback_dpo_pairs.py` — chosen/rejected = highest/lowest-reward completion per group of
4). SimPO's `CPOConfig.max_length` (prompt+completion combined, no separate prompt cap) defaults to 1024 —
too short for a meaningful fraction of UltraFeedback's longer prompts, silently truncating some completions
to near-zero length and producing NaN losses within the first few steps; pass `--max_length 2048` (or
larger) explicitly.

Fine-tuning a **base** (non-instruct) model on this off-policy UltraFeedback/DPO-pairs data works
out-of-the-box: every base-model script above falls back to a fixed PKU-Alignment raw-prompt template
(`"BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:"`) wherever `tokenizer.chat_template is None`, both
at train time (`_OffPolicyCollator` in `scripts/offpolicy_trainer.py`) and at eval time
(`scripts/generate_ultrafeedback_eval_completions.py`, `scripts/generate_alpacaeval_completions.py`,
`scripts/generate_mtbench_completions.py`). Evaluating an **instruct** checkpoint with this same fixed
template instead of its own native chat template collapses its output quality dramatically (measured
directly: >3 point mean-reward drop under UltraRM, larger than the entire base-vs-fine-tuned gap under
investigation) — always use each model's own template, never a fixed one, when comparing an instruct model
against anything else.

### TRIBE Stage 1's chi-squared trust region, `beta == 0` closed form

`tribe.stage1.compute_rho_star`'s `beta` parameter is genuinely inert (a provable no-op) only for
`divergence="kl_new_old"`, where the objective's `+beta*entr(rho)` entropy term happens to have the same
functional shape as that trust region's own penalty and folds into an equivalent rescaling of `lambda`. For
any other divergence (`"chi_squared"` in particular, the one every UltraFeedback off-policy TRIBE run in
this project uses) that term does NOT cancel and `beta` actively changes `rho*` even when the reference
log-ratio `c` is exactly 0 (the off-policy regime) — empirically, `beta > 0` acts as real entropy
regularization pulling `rho*` toward uniform, substantially fewer samples end up pinned at the group's
reweighting bounds than at `beta == 0` (checked directly against real training-batch advantages: ~6x fewer
boundary-pinned samples at `beta=0.1` vs `beta=0`).

`chi_squared` normally has no closed form and is solved via cvxpy every training step — slow, CPU-bound, and
the dominant runtime cost of any `chi_squared` TRIBE run. The special case `beta == 0` (only that case) does
have one: with the reference term entirely absent, the objective is linear in `rho`, and because Stage 1's
own advantage `A` is already group-mean-centered, the per-group equality-constraint multiplier collapses to
exactly 0, leaving `rho_i = 1 + A_i/lambda` with `lambda = sqrt(mean(A^2) / (2*eps))` — no iteration at all.
Opt in with `--use_chi_squared_beta0_closed_form True` (`train_ultrafeedback_offpolicy_tribe.py` /
`train_gsm8k_offpolicy_tribe.py`); it falls back to the general cvxpy solve automatically whenever the
closed form's implicit `rho >= 0` assumption doesn't hold for a batch (see
`tribe.stage1._solve_chi_squared_beta0_closed_form`'s own docstring, and
`tests/test_stage1.py::test_chi_squared_beta0_closed_form_matches_cvxpy` /
`test_chi_squared_beta0_closed_form_falls_back_when_infeasible` for the numerical validation against cvxpy).
Off by default since it's newer, less battle-tested code than the existing cvxpy path.

Safe-TRIBE (cost-constrained, BeaverTails/PKU-SafeRLHF) needs Safe-RLHF-style reward/cost model
checkpoints (`AutoModelForScore`, see `tribe/score_model.py`) trained on the same SFT base being aligned:

```bash
conda activate tribe
accelerate launch --num_processes=2 scripts/train_beavertails_safetribe.py \
    --config configs/safety/safetribe_hard.yaml \
    --model_name_or_path <sft_checkpoint> --reward_model_name_or_path <rm_checkpoint> \
    --cost_model_name_or_path <cm_checkpoint> --output_dir <your_output_dir>
```
(`configs/safety/safetribe_soft.yaml` for the soft/dual-ascent cost constraint instead of the default hard
cvxpy one — see `tribe/safe_tribe_config.py`'s docstring for the difference.)

See `slurm/run_final_3b_gsm8k_suite.sh` / `slurm/run_final_3b_math_suite.sh` for the full method x seed
sweep this project's own results were generated from, and `configs/offpolicy/` for every other baseline's
hyperparameters (GRPO, DAPO, RLOO, RAFT, TOPR, TIS, DPO, SimPO).

## Evaluation

**Before evaluating any checkpoint trained in the `tribe` env**, overlay its non-weight files (config,
tokenizer, chat template) from the base model it was fine-tuned from:

```bash
python scripts/overlay_base_config.py --checkpoint_dir <checkpoint> --base_model meta-llama/Llama-3.2-3B-Instruct
```
`transformers` 5.14.1 (the `tribe` env's writer) silently drops Llama-3.2's `rope_theta` on save (merges it
into a `rope_parameters` field `llm_gen`'s older transformers/vLLM doesn't recognize), degrading generation
quality with no error and no weight change — skipping this step is a silent correctness bug, not a
cosmetic one. Not needed for HF Hub base models (already correct) or checkpoints never touched by the
`tribe` env's writer.

```bash
conda activate llm_gen   # or llm_gen_new, see Installation
python scripts/eval_gsm8k.py --model_path <checkpoint_or_hf_id> --name <label> \
    --answer_format boxed --output_file <out>.json
python scripts/eval_math.py --model_path <checkpoint_or_hf_id> --name <label> \
    --num_generations 4 --temperature 1.0 --top_k 50 --output_file <out>.json   # pass@4
```
`--limit N` on either script runs on just the first N examples to sanity-check a checkpoint/env before the
full eval; `--val_fraction 0.1` scores the held-out training-set segment instead of the real test set (use
this for hyperparameter selection, never the actual test split — see `scripts/eval_gsm8k.py`'s docstring).

### UltraFeedback: reward-model scoring, AlpacaEval 2.0, MT-Bench

Reward-model win-rate (quick iteration, UltraRM-13b or Skywork-Reward-V2 as judge):

```bash
conda activate llm_gen_new
python scripts/generate_ultrafeedback_eval_completions.py --model_path <checkpoint_or_hf_id> --name <label> \
    --dataset_path <offpolicy-data dir> --output_file ultrafeedback_eval/<label>_completions.json
python scripts/score_ultrarm.py --completions_file ultrafeedback_eval/<label>_completions.json \
    --output_file ultrafeedback_eval/<label>_scored.json
```
`generate_ultrafeedback_eval_completions.py` also accepts `--temperature`/`--top_p`/`--top_k`/
`--repetition_penalty` (default greedy, `temperature=0.0`) and falls back to the PKU raw-prompt template for
base-model checkpoints (see above). `score_skyrm.py` is the same interface for
`Skywork/Skywork-Reward-V2-Llama-3.1-8B` (standard `AutoModelForSequenceClassification`, no custom class
needed, unlike UltraRM's ported `LlamaRewardModel`).

AlpacaEval 2.0 (`scripts/generate_alpacaeval_completions.py`, judged win-rate vs a fixed GPT-4-turbo
reference — NOT vs this project's own base model) and MT-Bench (`scripts/generate_mtbench_completions.py`,
multi-turn: turn 2's prompt is built from the model's own turn-1 answer) both need a dedicated env, since
`datasets>=4` dropped support for these packages' legacy HF-Hub loading scripts and `alpaca-eval`/`fschat`
aren't needed anywhere else in this project:

```bash
python3 -m venv .venv_evalharness && source .venv_evalharness/bin/activate
pip install "setuptools<81" "datasets<4" alpaca-eval fschat   # setuptools<81: alpaca-eval still imports pkg_resources
```
AlpacaEval's own `alpaca_eval_gpt4_turbo_fn`/`weighted_alpaca_eval_gpt4_turbo` annotator configs hardcode the
now-retired `gpt-4-1106-preview` — copy the config directory and swap in a currently-served model (e.g.
`gpt-4-turbo`) rather than editing the installed package in place. MT-Bench's question set/judge prompts/
reference answers aren't in the `fschat` PyPI package (only in the GitHub repo) — fetch them once:
```bash
mkdir -p mtbench/data/mt_bench/reference_answer
curl -sL -o mtbench/data/mt_bench/question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
curl -sL -o mtbench/data/mt_bench/reference_answer/gpt-4.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl
curl -sL -o mtbench/data/judge_prompts.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/judge_prompts.jsonl
```
Both judging steps call the OpenAI API in a loop — run them as a SLURM job (CPU-only, no GPU needed), never
as a blocking foreground command in an interactive session.
