# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SGLang adapters for Qwen3.5 sequence classification."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Type

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType, score_and_pool
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM
from sglang.srt.utils import add_prefix
from torch import nn


class _Qwen3_5ForSequenceClassificationBase(nn.Module):
    """Text-only Qwen3.5 backbone plus a replicated classification head."""

    backbone_cls: Type[Qwen3_5ForCausalLM] = Qwen3_5ForCausalLM
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise ValueError("Qwen3.5 sequence classification currently supports tensor parallelism only; disable PP")

        self.model = self.backbone_cls(
            self.text_config,
            quant_config=quant_config,
            prefix=add_prefix("model.language_model", prefix),
        )
        self.score = nn.Linear(
            self.text_config.hidden_size,
            int(config.num_labels),
            bias=False,
        )
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
        self.eos_token_id = self.text_config.eos_token_id

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        if not get_embedding:
            raise ValueError("Qwen3.5 sequence classification requires SGLang --is-embedding")
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        return score_and_pool(self.score, self.pooler, hidden_states, forward_batch, input_ids)

    @staticmethod
    def _backbone_weight_name(name: str) -> str | None:
        if name.startswith("model.visual.") or name.startswith("visual."):
            return None
        if name.startswith("mtp.") or ".mtp." in name or name.startswith("lm_head."):
            return None
        for prefix in ("model.language_model.", "language_model.", "model."):
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loaded = set()

        def iter_backbone_weights():
            for name, weight in weights:
                if name == "score.weight" or name.endswith(".score.weight"):
                    default_weight_loader(self.score.weight, weight)
                    loaded.add("score.weight")
                    continue
                backbone_name = self._backbone_weight_name(name)
                if backbone_name is not None:
                    yield backbone_name, weight

        loaded.update(self.model.load_weights(iter_backbone_weights()))
        if "score.weight" not in loaded:
            raise ValueError("Qwen3.5 sequence-classification checkpoint is missing score.weight")
        return loaded


class Qwen3_5ForSequenceClassification(_Qwen3_5ForSequenceClassificationBase):
    """Dense Qwen3.5 sequence classifier."""


class Qwen3_5MoeForSequenceClassification(_Qwen3_5ForSequenceClassificationBase):
    """Qwen3.5 MoE sequence classifier."""

    backbone_cls = Qwen3_5MoeForCausalLM
    packed_modules_mapping = Qwen3_5MoeForCausalLM.packed_modules_mapping


EntryClass = [Qwen3_5ForSequenceClassification, Qwen3_5MoeForSequenceClassification]
