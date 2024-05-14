import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding

class TextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    """

    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class AttentionPool1d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # BNC -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class LanguageAdapter(nn.Module):
    def __init__(self, input_dim=4096, output_dim=768, num_queries=120, num_encoder_layers=1, num_decoder_layers=1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self.queries = torch.nn.Parameter(torch.randn((1, num_queries, output_dim)))
        self.transformer = nn.Transformer(
            batch_first=True,
            norm_first=True,
            d_model=output_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=output_dim * 4,
            dropout=0.0,
        )

    def forward(self, encoder_hidden_states):
        bsz = encoder_hidden_states.shape[0]
        encoder_hidden_states = self.proj(encoder_hidden_states)
        encoder_hidden_states = self.transformer(src=encoder_hidden_states, tgt=self.queries.repeat(bsz, 1, 1))
        return encoder_hidden_states


class LanguageAdapterXL(nn.Module):
    def __init__(
        self, input_dim, num_queries=120, num_encoder_layers=1, num_decoder_layers=1,
        require_text_embedding=True,
        attention_pooling=False,
    ):
        super().__init__()

        self.adapter1 = LanguageAdapter(input_dim=input_dim, output_dim=1280, num_queries=num_queries, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.adapter2 = LanguageAdapter(input_dim=input_dim, output_dim=768, num_queries=num_queries, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        
        self.attention_pooling = attention_pooling
        
        if attention_pooling:
            self.pool_caption_projection = AttentionPool1d(120, 2048, 8, 1280)
        else:
            self.pool_caption_projection = TextProjection(768, 1280)
        
        if require_text_embedding:
            self.add_embedding = TimestepEmbedding(2816, 1280)
        else:
            self.add_embedding = None
        
    def forward(self, encoder_hidden_states, text_embeds):
        encoder_hidden_states1 = self.adapter1(encoder_hidden_states)
        encoder_hidden_states2 = self.adapter2(encoder_hidden_states)
        
        encoder_hidden_states = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=-1)
        
        if self.attention_pooling:
            text_embeds = self.pool_caption_projection(encoder_hidden_states)
        else:       
            text_embeds = self.pool_caption_projection(text_embeds)
        return encoder_hidden_states, text_embeds
