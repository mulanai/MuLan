from typing import Optional

import torch
from torch import nn

from .modeling_internvl import InternVLPreTrainedModel
from .modeling_qllama import LlamaForCausalLM, _expand_mask, _make_causal_mask


class InternVLTextModel(InternVLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.qllama = LlamaForCausalLM(config.qllama_config)  # frozen
        self.gradient_checkpointing = True

        text_hidden_size = config.qllama_config.hidden_size
        clip_embed_dim = config.clip_embed_dim
        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, clip_embed_dim))

    def get_input_embeddings(self):
        return self.qllama.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        return_pool=False,
    ):
        r"""
        Returns:
            text_outputs (`CausalLMOutputWithPast`, or `tuple(torch.FloatTensor)` if `return_dict=False`):
                The language model outputs. If `return_dict=True`, the output is a [`CausalLMOutputWithPast`] that
                contains the language model logits, the past key values and the hidden states if
                `output_hidden_states=True`.
        ```"""
        attention_mask = input_ids > 0

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.get_input_embeddings()(input_ids)
        attention_mask = _expand_mask(attention_mask, input_embeds.dtype).to(
            input_embeds.device)  # [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask += _make_causal_mask(
            (attention_mask.shape[0], attention_mask.shape[2]),
            input_embeds.dtype,
            device=input_embeds.device
        )
        if type(self.qllama.model) == LlamaForCausalLM:
            outputs = self.qllama.model.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=None,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state
        else:
            outputs = self.qllama.model.forward_train(
                inputs_embeds=input_embeds,
                vision_hidden_states=None,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ).last_hidden_state

        if return_pool:
            attention_mask = input_ids > 0
            text_embeds = outputs[torch.arange(outputs.shape[0]), attention_mask.sum(1) - 1]
            text_embeds = text_embeds @ self.text_projection
            return [outputs, text_embeds]

        return [outputs]
