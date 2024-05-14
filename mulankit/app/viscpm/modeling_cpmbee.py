# coding=utf-8
# Copyright 2022 The OpenBMB Team The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch CpmBee model."""
import copy
import math
from collections import UserDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.generation.beam_search import BeamHypotheses, BeamSearchScorer
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    dist,
    inspect,
    is_deepspeed_zero3_enabled,
    warnings,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmbee import CpmBeeConfig
from .tokenization_viscpmbee import VisCpmBeeTokenizer


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openbmb/cpm-bee-10b"
_CONFIG_FOR_DOC = "CpmBeeConfig"

CPMBEE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openbmb/cpm-bee-10b",
    "openbmb/cpm-bee-5b",
    "openbmb/cpm-bee-2b",
    "openbmb/cpm-bee-1b",
    # See all CPMBee models at https://huggingface.co/models?filter=cpmbee
]


class CpmBeeLinear(nn.Linear):
    def __init__(self, dim_in, dim_out, dtype):
        """
        Construct a linear for CPMBee. It contains a scale operation.
        """
        super().__init__(dim_in, dim_out, bias=False)
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out

        self.weight = torch.nn.parameter.Parameter(torch.empty((dim_out, dim_in), dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (`torch.Tensor` of shape `(batch, seq_len, dim_in)`): The input of linear layer
        Returns:
            `torch.Tensor` of shape `(batch, seq_len, dim_out)`: The output of the linear transform y.
        """
        x = nn.functional.linear(x, self.weight)
        x = x / math.sqrt(self.dim_in)
        return x


class CpmBeeLayerNorm(nn.Module):
    """
    We use Root Mean Square (RMS) Layer Normalization, please see https://arxiv.org/abs/1910.07467 for details."
    """

    def __init__(self, config: CpmBeeConfig):
        super().__init__()

        self.eps = config.eps
        self.dim_norm = config.hidden_size
        self.weight = nn.Parameter(torch.empty(config.hidden_size, dtype=config.torch_dtype))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        if hidden_states.size(-1) != self.dim_norm:
            raise AssertionError("hidden_states.size(-1) != self.dim_norm")
        old_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = (hidden_states * torch.rsqrt(variance + self.eps)).to(old_dtype) * self.weight
        return hidden_states


class CpmBeeAttention(nn.Module):
    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        self.dim_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dim_head = config.dim_head

        self.project_q = CpmBeeLinear(self.dim_model, self.num_heads * self.dim_head, dtype=config.torch_dtype)
        self.project_k = CpmBeeLinear(self.dim_model, self.num_heads * self.dim_head, dtype=config.torch_dtype)
        self.project_v = CpmBeeLinear(self.dim_model, self.num_heads * self.dim_head, dtype=config.torch_dtype)

        self.attention_out = CpmBeeLinear(self.num_heads * self.dim_head, self.dim_model, dtype=config.torch_dtype)

        self.softmax = torch.nn.Softmax(dim=-1)

        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_q: torch.Tensor,
        hidden_kv: torch.Tensor,
        attention_mask: torch.BoolTensor,
        position_bias: torch.Tensor,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_q (`torch.Tensor`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            hidden_kv (`torch.Tensor` of shape `(batch, len_k, dim_model)`)):
                Tensor *key_value* and *query* of shape `(batch, len_k, dim_model)`
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple[torch.Tensor, torch.Tensor]`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        batch_size = hidden_q.size(0)
        len_q = hidden_q.size(1)
        len_k = hidden_kv.size(1)

        query = self.project_q(hidden_q)
        key = self.project_k(hidden_kv)
        value = self.project_v(hidden_kv)

        query = query.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, len_k, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        if past_key_values is not None:
            key = torch.cat([past_key_values[0], key], dim=-2)
            value = torch.cat([past_key_values[1], value], dim=-2)
            len_k = key.size(-2)

        # (batch_size, num_heads, len_q, dim_head) @ (batch_size, num_heads, dim_head, len_k) -> (batch_size, num_heads, len_q, len_k)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim_head)
        score = score + position_bias

        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == torch.tensor(False),
            torch.scalar_tensor(float("-inf"), device=score.device, dtype=score.dtype),
        )
        score = self.softmax(score)

        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k) == torch.tensor(False),
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype),
        )
        if output_attentions:
            attn_weights = score
        else:
            attn_weights = None

        if self.dropout is not None:
            score = self.dropout(score)

        # (batch_size, num_heads, len_q, len_k) @ (batch_size, num_heads, len_k, dim_head) -> (batch_size, num_heads, len_q, dim_head)
        score = torch.matmul(score, value)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3)
        score = score.contiguous().view(batch_size, len_q, self.num_heads * self.dim_head)

        score = self.attention_out(score)

        past_key_values = None
        if use_cache:
            past_key_values = (key, value)

        return score, attn_weights, past_key_values


class CpmBeeSelfAttentionBlock(nn.Module):
    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        self.layernorm_before_attention = CpmBeeLayerNorm(config)
        self.self_attention = CpmBeeAttention(config)
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        outputs = self.layernorm_before_attention(hidden_states)
        outputs = self.self_attention(
            outputs, outputs, attention_mask, position_bias, output_attentions, past_key_values, use_cache
        )

        outputs, attn_weights, current_key_value = outputs

        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = (hidden_states + outputs) / 1.05

        return hidden_states, attn_weights, current_key_value


class CpmBeeDenseGatedACT(nn.Module):
    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        self.w_0 = CpmBeeLinear(config.hidden_size, config.dim_ff, dtype=config.torch_dtype)
        self.w_1 = CpmBeeLinear(config.hidden_size, config.dim_ff, dtype=config.torch_dtype)
        self.act = torch.nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        gate_score = self.act(self.w_0(hidden_states))
        hidden_states = self.w_1(hidden_states)

        hidden_states = gate_score * hidden_states
        return hidden_states


class CpmBeeFeedForward(nn.Module):
    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        self.w_in = CpmBeeDenseGatedACT(config)
        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

        self.w_out = CpmBeeLinear(config.dim_ff, config.hidden_size, dtype=config.torch_dtype)

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        hidden_states = self.w_in(hidden_states)

        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_out(hidden_states)

        return hidden_states


class CpmBeeFFNBlock(nn.Module):
    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        self.layernorm_before_ffn = CpmBeeLayerNorm(config)
        self.ffn = CpmBeeFeedForward(config)
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Hidden states before feed forward layer.
        """
        ln_outputs = self.layernorm_before_ffn(hidden_states)
        outputs = self.ffn(ln_outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = (hidden_states + outputs) / 1.05
        return hidden_states


class CpmBeeTransformerBlock(nn.Module):
    def __init__(self, config: CpmBeeConfig, mask_att: bool = False, mask_ffn: bool = False):
        super().__init__()
        self.mask_att = mask_att
        self.mask_ffn = mask_ffn

        if not self.mask_att:
            self.self_att = CpmBeeSelfAttentionBlock(config)
        if not self.mask_ffn:
            self.ffn = CpmBeeFFNBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input to the layer of shape `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                Cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        if not self.mask_att:
            hidden_states = self.self_att(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            hidden_states, attn_weights, current_key_value = hidden_states
        else:
            attn_weights, current_key_value = None, (None, None)

        if not self.mask_ffn:
            hidden_states = self.ffn(hidden_states)

        return hidden_states, attn_weights, current_key_value


class CpmBeeEncoder(nn.Module):
    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        self.num_layers = config.num_hidden_layers
        if config.mask_modules is not None:
            assert len(config.mask_modules) == self.num_layers, "The total number of masks should equal to num_layers"
            for mask_module in config.mask_modules:
                assert len(mask_module) == 2, "For encoder, each mask should be (mask_att, mask_ffn)"
        else:
            config.mask_modules = [(False, False)] * self.num_layers

        self.layers = nn.ModuleList(
            [
                CpmBeeTransformerBlock(
                    config, mask_att=config.mask_modules[ith][0], mask_ffn=config.mask_modules[ith][1]
                )
                for ith in range(self.num_layers)
            ]
        )

        self.output_layernorm = CpmBeeLayerNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                Input to the layer of shape `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                Avoid invalid areas to participate in the calculation of shape `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                Provides position information to attention mechanism of shape `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                Cached past key and value projection states
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        current_key_values = () if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_bias,
                output_attentions=output_attentions,
                past_key_values=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
            )
            hidden_states, attn_weights, current_key_value = layer_outputs
            if output_attentions:
                all_self_attns += (attn_weights,)
            if current_key_value is not None:
                current_key_values = current_key_values + (current_key_value,)

        hidden_states = self.output_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, current_key_values, all_hidden_states, all_self_attns


class CpmBeeBucketPositionBias(nn.Module):
    def __init__(self, config: CpmBeeConfig) -> None:
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_buckets = config.position_bias_num_buckets
        self.num_segment_bucket = config.position_bias_num_segment_buckets
        self.max_distance = config.position_bias_max_distance

        self.relative_attention_bias = nn.Parameter(
            torch.empty(
                config.position_bias_num_buckets + config.position_bias_num_segment_buckets,
                config.num_attention_heads,
                dtype=config.torch_dtype,
            ),
        )

    def forward(self, query_pos: torch.Tensor, key_pos: torch.Tensor, rel_buckets: torch.Tensor):
        with torch.no_grad():
            batch = key_pos.size(0)
            keylen = key_pos.size(1)
            querylen = query_pos.size(1)

            if key_pos.size(0) != query_pos.size(0):
                raise AssertionError(
                    f"key_pos.size(0) should be equal to query_pos.size(0), but got {key_pos.size(0)} and {query_pos.size(0)}!"
                )
            if rel_buckets.size(0) != batch:
                raise AssertionError(
                    f"rel_buckets.size(0) should be equal to batch, but got {rel_buckets.size(0)} and {batch}!"
                )
            if rel_buckets.size(1) != querylen:
                raise AssertionError(
                    f"rel_buckets.size(1) should be equal to querylen, but got {rel_buckets.size(1)} and {querylen}!"
                )
            if rel_buckets.size(2) != keylen:
                raise AssertionError(
                    f"rel_buckets.size(2) should be equal to keylen, but got {rel_buckets.size(2)} and {keylen}!"
                )

            relative_position_bucket = rel_buckets - 1 + self.num_buckets

            inner_segment_bucket = self._position_bucket(
                key_pos[..., None, :] - query_pos[..., :, None],
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            relative_position_bucket = torch.where(
                rel_buckets == 0,
                inner_segment_bucket,
                relative_position_bucket,
            )

        embeds = nn.functional.embedding(relative_position_bucket, self.relative_attention_bias)
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        return embeds

    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        num_buckets //= 2
        relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        return relative_buckets


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->CPMBee
class CpmBeeOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CpmBeeRotaryEmbedding(nn.Module):
    """
    RotaryEmbedding embeds the unk token and special token. It will embeds the "...<mask>...<mask>...<unk>...<unk>..."
    to "...<mask_0>...<mask_1>...<unk_0>...<unk_1>..."" to help model to specify different special tokens and unk
    tokens.
    """

    def __init__(self, config: CpmBeeConfig):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, config.hidden_size, 2, dtype=torch.float32) / config.hidden_size))
        self.distance_scale = config.distance_scale
        self.dtype = config.torch_dtype
        self.inv_freq = inv_freq.to(config.torch_dtype)

    def forward(self, x: torch.Tensor, x_pos: torch.Tensor):
        inv_freq = self.inv_freq.to(device=x.device, dtype=self.dtype)

        x_pos = x_pos * self.distance_scale
        freqs = x_pos[..., None].to(self.dtype) * inv_freq[None, :]  # (..., dim/2)

        emb = torch.cat((freqs, freqs), dim=-1)  # (..., dim)
        emb_cos = emb.cos()  # (..., dim)
        emb_sin = emb.sin()  # (..., dim)

        rotate_x = torch.cat([-x[..., x.size(-1) // 2 :], x[..., : x.size(-1) // 2]], dim=-1)  # (..., dim)

        return x * emb_cos + rotate_x * emb_sin


class CpmBeeEmbeddingExt(nn.Embedding):
    """
    Contains a RotaryEmbedding.
    """

    def __init__(self, config: CpmBeeConfig):
        super().__init__(config.vocab_size, config.hidden_size, dtype=config.torch_dtype)
        self.dim_model = config.hidden_size
        self.rotary_emb = CpmBeeRotaryEmbedding(config)

    def forward(self, ids: torch.Tensor, ids_sub: torch.Tensor):
        embeds = super().forward(ids) / math.sqrt(self.dim_model)
        return self.rotary_emb(embeds, ids_sub)

    def projection(self, x: torch.Tensor, ext_table: Optional[torch.Tensor] = None):
        logits = nn.functional.linear(x / math.sqrt(self.dim_model), self.weight)
        if ext_table is not None:
            logits_ext = nn.functional.linear(x, ext_table)
            logits = torch.cat([logits, logits_ext], dim=-1)
        return logits


class CpmBeePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CpmBeeConfig
    base_model_prefix = "cpmbee"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        # still needed
        elif isinstance(module, CpmBeeEmbeddingExt):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, CpmBeeLayerNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, CpmBeeBucketPositionBias):
            module.relative_attention_bias.data.normal_(mean=0.0, std=self.config.init_std)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CpmBeeEncoder):
            module.gradient_checkpointing = value


CPMBEE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters
        config ([`~CpmBeeConfig`]): Model configuration class with all the parameters of the
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CPMBEE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`CPMBeeTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_id_sub (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Subscription of input sequence tokens in the vocabulary.

            Subscription of normal text will be zero while the special tokens of each group will be the 0, 1, 2, ...
            <ans_0>, <ans_1>, <ans_2> ... belongs to group <ans>. <mask_0>, <mask_1>, <mask_2> ... belongs to group
            <mask>.
        position (`torch.Tensor` of shape `(batch_size, seq_len)`):
            The position of input sequence tokens in the vocabulary for each segment. if segment1 is 0, 1, 2 and
            segment2 is 0, 1, 2, 3, the position will be 0, 1, 2, 0, 1, 2, 3
        context (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Whether this token id is context or not. If is context, the value is 1. If not, the value is 0. If a token
            id is context, it does not need to be predicted.
        sample_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Give a sample id to every token id. The token ids with same sample ids belongs to the same sample.
        num_segments (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Total number of segments in the current input.
        segment (`torch.Tensor` of shape `(batch_size, seq_len)`):
            Give a segment id to every token id. The token ids with same segment ids belongs to the same sample.

            Generally, a string key or value in input data will be a segment. For example, input {"input": "hello, ",
            "<ans>": ""}, the segments includes: "input", "hello, ", "<ans>" and "".
        segment_rel_offset (`torch.Tensor` of shape `(batch_size, seq_len)`):
            The offset of segment rel.
        segment_rel (`torch.Tensor` of shape `(batch_size, seq_len)`):
            The segment relevance. A relative implementation of measuring the importance of segments.
        past_states (`Dict[str, Union[torch.Tensor, List]]`):
            Store the history information including position, context, sample_ids, num_segments, segment and
            past_key_values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A dummy arguments for CPMBee. The `past_states` contains pre-computed hidden-states (key and values in the
            self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) and
            other history arguments to speed up sequential decoding.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare CPMBee Model outputting raw hidden-states without any specific head on top.",
    CPMBEE_START_DOCSTRING,
)
class CpmBeeModel(CpmBeePreTrainedModel):
    def __init__(self, config: CpmBeeConfig):
        super().__init__(config)
        if config.half:
            config.torch_dtype = torch.half
        else:
            config.torch_dtype = torch.float
        self.encoder = CpmBeeEncoder(config)
        self.input_embedding = CpmBeeEmbeddingExt(config)
        self.position_bias = CpmBeeBucketPositionBias(config)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.input_embedding

    def set_input_embeddings(self, embeddings, **kwargs):
        self.input_embedding = embeddings

    @add_start_docstrings_to_model_forward(CPMBEE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        input_id_sub: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        num_segments: Optional[torch.Tensor] = None,
        segment: Optional[torch.Tensor] = None,
        segment_rel_offset: Optional[torch.Tensor] = None,
        segment_rel: Optional[torch.Tensor] = None,
        past_states: Optional[Dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[List] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # dummy setting for common tests
        if input_id_sub is None:
            dtype, device = input_ids.dtype, input_ids.device
            batch, seq_length = input_ids.size()
            segment = torch.where(input_ids != 0, 2, 0).to(dtype=dtype, device=device)
            context = torch.full((batch, seq_length), 1, dtype=dtype, device=device)
            position = torch.arange(seq_length, dtype=dtype, device=device).repeat(batch, 1)
            input_id_sub = torch.full((batch, seq_length), 0, dtype=dtype, device=device)
            segment_rel_offset = torch.full((batch, seq_length), 0, dtype=dtype, device=device)
            segment_rel = torch.full((batch, seq_length), 0, dtype=dtype, device=device)
            num_segments = torch.full((batch, seq_length), 0, dtype=dtype, device=device)
            sample_ids = torch.zeros_like(input_ids)

        with torch.no_grad():
            if past_states is None:
                present_position = position
                present_context = context
                present_sample_ids = sample_ids
                present_num_segments = num_segments
                present_segments = segment
                present_buffer = None
            else:
                present_position = torch.cat([past_states["buffer_position"], position], dim=-1)
                present_context = torch.cat([past_states["buffer_context"], context], dim=-1)
                present_sample_ids = torch.cat([past_states["buffer_sample_ids"], sample_ids], dim=-1)
                present_num_segments = torch.cat([past_states["buffer_num_segments"], num_segments], dim=-1)
                present_segments = torch.cat([past_states["buffer_segments"], segment], dim=-1)
                present_buffer = past_states["buffer"]

            batch = input_ids.size(0)
            len_q = input_ids.size(1)
            len_buffer = present_position.size(1)

            segment_rel_2d = torch.masked_fill(
                segment[:, :, None] * num_segments[:, :, None]
                + present_segments[:, None, :]
                + segment_rel_offset[:, :, None],
                ~((sample_ids[:, :, None] == present_sample_ids[:, None, :])),  # not in the same sample
                0,  # avoid torch.gather overflow
            ).view(batch, len_q * len_buffer)

            segment_bucket = torch.gather(
                input=segment_rel,
                dim=1,
                index=segment_rel_2d.long(),
            ).view(batch, len_q, len_buffer)

            segment_bucket.masked_fill_(
                ~((sample_ids[:, :, None] == present_sample_ids[:, None, :])),  # not in the same span or sample
                1,  # bucket is used for in-context samples
            )

            # directional mask
            directional_mask_2d = present_position[:, None, :] <= position[:, :, None]
            # sample mask
            sample_mask_2d = (sample_ids[:, :, None] == 0) | (sample_ids[:, :, None] == present_sample_ids[:, None, :])
            # context mask
            attention_mask = present_context[:, None, :] | (
                context[:, :, None].logical_not() & directional_mask_2d.view(batch, len_q, len_buffer)
            )
            # span mask
            attention_mask = attention_mask & sample_mask_2d
            # length mask
            mask_1d = present_num_segments != 0
            attention_mask = mask_1d.view(batch, 1, len_buffer) & attention_mask

        hidden_states = self.input_embedding(input_ids, input_id_sub)
        position_bias = self.position_bias(position, present_position, segment_bucket)
        hidden_states, present_key_values, all_hidden_states, all_attentions = self.encoder(
            hidden_states,
            attention_mask,
            position_bias,
            output_attentions,
            output_hidden_states,
            present_buffer,
            use_cache,
        )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, present_key_values, all_hidden_states, all_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CpmBeeBeamHypotheses(BeamHypotheses):
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool, max_length: Optional[int] = None):
        """
        Override BeamHypotheses for CpmBee. The hyp to add is list but not tensor.
        """
        super().__init__(num_beams, length_penalty, early_stopping, max_length)

    def add(self, hyp: List, sum_logprobs: float, beam_indices: Optional[torch.LongTensor] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (len(hyp) ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, beam_indices))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)


class CPMBeeTransBlock(torch.nn.Module):
    def __init__(
        self,
        dim_model=4096,
        dim_ff=1024,
        dim_out=768,
        dtype=torch.float,
        eps=1e-6,
        dropout_p=0,
    ):
        super().__init__()
        if dropout_p is not None:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None
        self.w_out_res = torch.nn.Linear(dim_model, dim_out, bias=False)
        self.layernorm = torch.nn.LayerNorm(
            dim_out,
            dtype=dtype,
            eps=eps,
        )

    def forward(self, hidden_states: torch.Tensor):
        x_res = self.w_out_res(hidden_states)
        if self.dropout is not None:
            x_res = self.dropout(x_res)
        hidden_states = self.layernorm(x_res)
        return hidden_states


class CpmBeeWithTransform(CpmBeePreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: CpmBeeConfig):
        super().__init__(config)
        self.llm = CpmBeeModel(config)

        self.trans_block = CPMBeeTransBlock(config.hidden_size, config.hidden_size // 4, config.unet_cross_attention_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_id_sub: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        sample_ids: Optional[torch.Tensor] = None,
        num_segments: Optional[torch.Tensor] = None,
        segment: Optional[torch.Tensor] = None,
        segment_rel_offset: Optional[torch.Tensor] = None,
        segment_rel: Optional[torch.Tensor] = None,
        past_states: Optional[Dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[List] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        outputs = self.llm(input_ids, input_id_sub, position, context,
            sample_ids, num_segments, segment, segment_rel_offset,
            segment_rel, past_states, output_attentions, output_hidden_states,
            past_key_values, use_cache, return_dict, **kwargs,)
        if return_dict:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        #if self.trans_block is not None:
        #    hidden_states = self.trans_block(hidden_states)
        return outputs, hidden_states
