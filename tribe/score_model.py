"""
Standalone `LlamaForScore` (Safe-RLHF's reward/cost model architecture), ported directly into this repo so
scripts/train_beavertails_safetribe.py doesn't need the `safe_rlhf` package importable from the `tribe`
conda env — only `transformers`/`torch`, already there. Loads checkpoints produced by Safe-RLHF's own
training scripts (https://github.com/PKU-Alignment/safe-rlhf) unchanged: same class hierarchy, same
`state_dict` keys (`model.*` for the base LlamaModel, `score_head.*` for the linear scoring layer,
`normalizer.*` for its running-stats buffers), so `from_pretrained` on one of their checkpoints just works.

Trimmed relative to `safe_rlhf.models.score_model`: only Llama is ported (the only architecture this
project uses) — no `AutoModelForScore` multi-architecture dispatch, no bloom/gemma/gpt2/etc. variants, no
package-level docstring-decorator plumbing tied to the safe_rlhf repo's own doc infra. Everything else
(the `ScoreModelMixin`/`Normalizer` logic itself) is a faithful copy, Apache-2.0 licensed same as the
source (Copyright 2023-2024 PKU-Alignment Team).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.types import Number
from transformers import LlamaModel, LlamaPreTrainedModel, PretrainedConfig
from transformers.utils.generic import ModelOutput


NormalizeFunction = Literal["affine", "scale", "translate", "identity"]
NormalizerType = Literal["RunningMeanStd", "ExponentialMovingAverage"]


class Normalizer(nn.Module):
    """Normalize input to have zero mean and unit variance."""

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.LongTensor
    normalize_function: NormalizeFunction

    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if normalize_function not in {"affine", "scale", "translate", "identity"}:
            raise ValueError(
                f"Invalid normalization function type: {normalize_function}. "
                'Expected one of "affine", "scale", "translate", "identity".'
            )
        self.normalize_function = normalize_function
        self.register_buffer("mean", torch.zeros(shape, device=device))
        self.register_buffer("var", torch.ones(shape, device=device))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long, device=device))

    @abstractmethod
    def update(self, data: torch.Tensor) -> None:
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        return self.var.sqrt()

    def set_mean_var(
        self,
        mean: torch.Tensor | list[float] | tuple[float, ...] | None,
        var: torch.Tensor | list[float] | tuple[float, ...] | None,
    ) -> None:
        mean = torch.as_tensor(mean, dtype=self.mean.dtype, device=self.mean.device) if mean is not None else self.mean
        var = torch.as_tensor(var, dtype=self.var.dtype, device=self.var.device) if var is not None else self.var
        assert mean.shape == self.mean.shape
        assert var.shape == self.var.shape
        self.mean = mean
        self.var = var

    def forward(self, data: torch.Tensor, epsilon: Number = 1e-8) -> torch.Tensor:
        if self.training:
            self.update(data)
        return self.normalize(data, epsilon=epsilon)

    def normalize(self, data: torch.Tensor, epsilon: Number = 1e-8) -> torch.Tensor:
        if self.normalize_function == "affine":
            return (data - self.mean.detach()) / (self.std.detach() + epsilon)
        if self.normalize_function == "scale":
            return data / (self.std.detach() + epsilon)
        if self.normalize_function == "translate":
            return data - self.mean.detach()
        if self.normalize_function == "identity":
            return data
        raise ValueError(f"Invalid normalization function type: {self.normalize_function}.")

    @classmethod
    def instantiate(
        cls,
        normalizer_type: NormalizerType | None,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> "Normalizer":
        if normalizer_type == "RunningMeanStd":
            return RunningMeanStd(normalize_function, shape=shape, device=device)
        if normalizer_type == "ExponentialMovingAverage":
            return ExponentialMovingAverage(normalize_function, shape=shape, device=device, **kwargs)
        if normalizer_type is None:
            return IdentityNormalizer(normalize_function, shape=shape, device=device)
        raise ValueError(f"Invalid normalizer type: {normalizer_type!r}.")


class RunningMeanStd(Normalizer):
    def update(self, data: torch.Tensor) -> None:
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_count = data.size(0)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * (self.count * batch_count / total_count)
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class ExponentialMovingAverage(Normalizer):
    def __init__(
        self,
        normalize_function: NormalizeFunction,
        shape: tuple[int, ...],
        device: torch.device | str | None = None,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(normalize_function, shape=shape, device=device)
        self.momentum = momentum

    def update(self, data: torch.Tensor) -> None:
        batch_mean = data.mean(dim=0)
        batch_var = data.var(dim=0)
        batch_count = data.size(0)
        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * batch_var
        self.count += batch_count


class IdentityNormalizer(Normalizer):
    def update(self, data: torch.Tensor) -> None:
        self.count += data.size(0)


@dataclass
class ScoreModelOutput(ModelOutput):
    """
    Args:
        scores (`torch.FloatTensor` of shape `(batch_size, sequence_length, score_dim)`).
        end_scores (`torch.FloatTensor` of shape `(batch_size, score_dim)`).
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`).
        end_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_dim)`).
        end_index (`torch.LongTensor` of shape `(batch_size,)`).
    """

    scores: torch.FloatTensor | None = None
    end_scores: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    end_last_hidden_state: torch.FloatTensor | None = None
    end_index: torch.LongTensor | None = None


class ScoreModelMixin:
    """Base class for score models — adds a linear score head + normalizer on top of a base LM."""

    score_head: nn.Linear
    normalizer: Normalizer
    do_normalize: bool = False
    normalize_function: NormalizeFunction = "affine"
    _is_score_head_initialized: bool = False

    def init_score_head(self, config: PretrainedConfig, hidden_size: int, **kwargs: Any) -> None:
        if self._is_score_head_initialized:
            return

        self.score_dim = config.score_dim = kwargs.pop("score_dim", getattr(config, "score_dim", 1))
        self.score_bias = config.score_bias = kwargs.pop("score_bias", getattr(config, "score_bias", True))

        self.score_head = nn.Linear(hidden_size, config.score_dim, bias=config.score_bias)
        if config.score_bias:
            nn.init.zeros_(self.score_head.bias)

        config.score_type = kwargs.pop("score_type", getattr(config, "score_type", "reward"))
        if config.score_type == "reward":
            self.normalize_function = "affine"
        elif config.score_type == "cost":
            self.normalize_function = "scale"
        elif config.score_type == "critic":
            self.normalize_function = "identity"
        else:
            raise ValueError(f"Invalid score type: {config.score_type!r}.")

        self.do_normalize = config.do_normalize = kwargs.pop("do_normalize", getattr(config, "do_normalize", False))

        config.normalizer_type = kwargs.pop("normalizer_type", getattr(config, "normalizer_type", None))
        if config.normalizer_type not in {"RunningMeanStd", "ExponentialMovingAverage", None}:
            raise ValueError(f"Invalid normalizer type: {config.normalizer_type!r}.")
        if config.normalizer_type == "ExponentialMovingAverage":
            config.momentum = kwargs.pop("momentum", getattr(config, "momentum", None))
        momentum = getattr(config, "momentum", None)
        self.normalizer = Normalizer.instantiate(
            normalizer_type=config.normalizer_type,
            normalize_function=self.normalize_function,
            shape=(config.score_dim,),
            momentum=momentum,
        )

        mean = getattr(config, "mean", None)
        var = getattr(config, "var", None)
        self.normalizer.set_mean_var(mean, var)

        self._is_score_head_initialized = True

    def get_scores(
        self,
        last_hidden_state: torch.FloatTensor,
        attention_mask: torch.BoolTensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput:
        B, L, E = last_hidden_state.size()

        if attention_mask is None:
            if B > 1:
                raise ValueError("'attention_mask' is required when batch size > 1.")
            attention_mask = last_hidden_state.new_ones(B, L, dtype=torch.bool)

        scores = self.score_head(last_hidden_state).float()

        end_index = torch.cat([m.nonzero()[-1] for m in attention_mask])
        end_last_hidden_state = torch.gather(
            last_hidden_state,
            dim=1,
            index=(end_index.to(last_hidden_state.device).unsqueeze(1).unsqueeze(2).expand(-1, -1, last_hidden_state.size(-1))),
        )
        end_scores = torch.gather(
            scores,
            dim=1,
            index=(end_index.to(scores.device).unsqueeze(1).unsqueeze(2).expand(-1, -1, scores.size(-1))),
        )
        end_last_hidden_state = end_last_hidden_state.squeeze(dim=1)
        end_scores = end_scores.squeeze(dim=1)

        if self.training:
            if dist.is_initialized():
                gathered_end_scores_list = [torch.zeros_like(end_scores) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_end_scores_list, end_scores)
                gathered_end_scores = torch.cat(gathered_end_scores_list, dim=0)
                self.normalizer.update(gathered_end_scores)
            else:
                self.normalizer.update(end_scores)
            self.config.mean = self.normalizer.mean.tolist()
            self.config.var = self.normalizer.var.tolist()

        if self.do_normalize:
            scores = self.normalizer.normalize(scores)
            end_scores = self.normalizer.normalize(end_scores)

        if not return_dict:
            return scores, end_scores

        return ScoreModelOutput(
            scores=scores,
            end_scores=end_scores,
            last_hidden_state=last_hidden_state,
            end_last_hidden_state=end_last_hidden_state,
            end_index=end_index,
        )

    def set_normalize(self, mode: bool = True) -> None:
        if self.do_normalize == mode:
            return
        self.do_normalize = self.config.do_normalize = mode


class LlamaForScore(ScoreModelMixin, LlamaPreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder) -> None:
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | ScoreModelOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state
        return self.get_scores(last_hidden_state, attention_mask=attention_mask, return_dict=return_dict)
