"""PyTorch Kmoshi model."""

from collections.abc import Callable

import torch

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..personaplex.configuration_personaplex import PersonaplexConfig, PersonaplexDepthConfig
from ..personaplex.modeling_personaplex import (
    PersonaplexAttention,
    PersonaplexForCausalLM,
    PersonaplexForConditionalGeneration,
    PersonaplexModel,
    PersonaplexRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class KmoshiRMSNorm(PersonaplexRMSNorm):
    pass


class KmoshiDepthConfig(PersonaplexDepthConfig):
    pass


class KmoshiConfig(PersonaplexConfig):
    # Kmoshi swaps the Helium temporal backbone for Qwen3-8B (Korean-capable text LLM).
    # `hidden_size` stays 4096 on purpose: the 16 audio input embeddings and the depth
    # decoder's `input_projections` keep their PersonaPlex shapes (and weights) across
    # the swap. `ffn_dim` is the fused-GLU width: 2 * Qwen3-8B intermediate_size (12288).
    default_theta = 1000000.0

    vocab_size: int = 151936
    num_hidden_layers: int = 36
    num_key_value_heads: int | None = 8
    head_dim: int | None = 128
    ffn_dim: int = 24576
    rms_norm_eps: float = 1e-6


class KmoshiAttention(PersonaplexAttention):
    # Qwen3 normalizes q/k per head (QK-Norm) before RoPE — its weights require it.
    # Only the temporal backbone carries Qwen3 weights, so the norms are created for
    # the non-flexible path only; the depth decoder (use_flexible_linear=True) keeps
    # the PersonaPlex attention computation unchanged.
    def __init__(self, config: KmoshiConfig, layer_idx: int | None = None, use_flexible_linear=False, use_rope=True):
        super().__init__(config, layer_idx, use_flexible_linear, use_rope)
        self.q_norm = None
        self.k_norm = None
        if not use_flexible_linear:
            self.q_norm = KmoshiRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = KmoshiRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        codebook_idx: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states, codebook_idx).view(hidden_shape)
        key_states = self.k_proj(hidden_states, codebook_idx).view(hidden_shape)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states, codebook_idx).view(hidden_shape).transpose(1, 2)

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, codebook_idx)
        return attn_output, attn_weights


class KmoshiModel(PersonaplexModel):
    pass


class KmoshiForCausalLM(PersonaplexForCausalLM):
    pass


class KmoshiForConditionalGeneration(PersonaplexForConditionalGeneration):
    pass


__all__ = [
    "KmoshiConfig",
    "KmoshiDepthConfig",
    "KmoshiModel",
    "KmoshiForCausalLM",
    "KmoshiForConditionalGeneration",
    "KmoshiPreTrainedModel",  # noqa: F822
]
