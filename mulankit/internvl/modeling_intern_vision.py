from functools import partial
from typing import Optional
import torch
import torch.nn as nn

from .modeling_intern_vit import InternVisionModel
from .modeling_internvl import InternVLPreTrainedModel, AttentionPoolingBlock


class InternVLVisionModel(InternVLPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_model = InternVisionModel(config.vision_config)  # frozen
        self.gradient_checkpointing = True

        vision_hidden_size = config.vision_config.hidden_size
        clip_embed_dim = config.clip_embed_dim
        attn_pool_num_heads = config.attn_pool_num_heads
        self.clip_projector = AttentionPoolingBlock(  # frozen
            dim=vision_hidden_size, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)

    def encode_image(self, image, mode):
        vision_outputs = self.vision_model(
            pixel_values=image,
            output_hidden_states=False,
            return_dict=True)
        image_embeds = vision_outputs[0]
        image_embeds = self.clip_projector(image_embeds)
        return image_embeds

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
    ):
        return self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_embeds=pixel_embeds,
        )
