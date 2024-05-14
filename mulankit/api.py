import os

import torch
from diffusers import PixArtAlphaPipeline, StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from .internvl.modeling_intern_text import InternVLTextModel
from .models.adapter import LanguageAdapter, LanguageAdapterXL, TextProjection
from .models.unet import UNetSDModel, UNetSDXLModel
from .patch import (
    MAX_TOKEN_LENGTH,
    encode_prompt_pixart,
    encode_prompt_sd15,
    encode_prompt_sdxl,
    setup,
)

SD15_PIPELINES = [
    StableDiffusionPipeline,
]
PIXART_PIPELINES = [
    PixArtAlphaPipeline,
]
SDXL_PIPELINES = []


def transform(
    pipe,
    adapter_path=None,
    adapter=None,
    text_encoder_path='OpenGVLab/InternVL-14B-224px',
    text_encoder=None,
    tokenizer=None,
    pipe_type=None,
    replace=False,
):
    if pipe_type is None:
        pipe_type = infer_pipe_type(pipe, adapter_path, adapter)

    text_encoder, tokenizer = load_internvl(text_encoder_path, text_encoder, tokenizer, torch_dtype=pipe.text_encoder.dtype)

    tokenizer.pad_token_id = 0  # set pad_token_id to 0
    tokenizer.model_max_length = MAX_TOKEN_LENGTH
    pipe.text_encoder = text_encoder.to(pipe.text_encoder.device)
    pipe.tokenizer = tokenizer

    if adapter is None:
        adapter = load_adapter(adapter_path, type=pipe_type)

    if pipe_type in ['sd15', 'sd21', 'sdxl']:
        adapter = adapter.to(pipe.text_encoder.device)
        if pipe_type in ['sd15', 'sd21']:
            def func(*args, **kwargs): return encode_prompt_sd15(pipe, *args, **kwargs)
            unet = UNetSDModel(pipe.unet, adapter, replace=replace)
            pipe.unet = unet.to(device=pipe.unet.device, dtype=pipe.unet.dtype)
        elif pipe_type == 'sdxl':
            def func(*args, **kwargs): return encode_prompt_sdxl(pipe, *args, **kwargs)
            unet = UNetSDXLModel(pipe.unet, adapter, replace=replace)
            if adapter.add_embedding is not None:
                print('load add embedding')
                unet.add_embedding = adapter.add_embedding
            pipe.unet = unet.to(device=pipe.unet.device, dtype=pipe.unet.dtype)
    elif pipe_type == 'pixart':
        def func(*args, **kwargs): return encode_prompt_pixart(pipe, *args, **kwargs)
        if adapter is not None:
            pipe.transformer.caption_projection = adapter.to(device=pipe.text_encoder.device, dtype=pipe.text_encoder.dtype)
    pipe.encode_prompt = func

    print('pipe_type', pipe_type)
    return pipe


def load_internvl(model_id, text_encoder, tokenizer, torch_dtype=None):
    if text_encoder is None:
        text_encoder = InternVLTextModel.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
        )
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, add_eos_token=True)

    return text_encoder, tokenizer


def load_adapter(path, device=None, type=None):
    if path is None:
        return None

    if type is None:
        type = infer_pipe_type(None, path)
    print(type, flush=True)
    
    if not os.path.exists(path):
        try:
            repo_id, filename = path.split('::')
            path = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            raise ValueError(f'{path} not found')
        
    adapter_state_dict = torch.load(path)

    if type in ['sd15', 'sd21']:
        adapter = LanguageAdapter(4096, 768 if type == 'sd15' else 1024, num_queries=77 if 'direct' in path else 120)
    elif type == 'pixart':
        adapter = TextProjection(4096, 1152)
    elif type == 'sdxl':
        adapter = LanguageAdapterXL(
            4096,
            require_text_embedding=any('add_embedding' in key for key in adapter_state_dict.keys()),
            attention_pooling=any('positional_embedding' in key for key in adapter_state_dict.keys()),
        )

    adapter.load_state_dict(adapter_state_dict)
    if device is not None:
        adapter = adapter.to(device)
    adapter.pipe_type = type
    return adapter


def infer_pipe_type(pipe, adapter_path, adapter=None):
    if adapter is not None:
        return adapter.pipe_type

    if adapter_path is None:
        adapter_path = ''

    if 'sd15' in adapter_path:
        return 'sd15'
    if 'sd21' in adapter_path:
        return 'sd21'
    if any([isinstance(pipe, cls) for cls in SD15_PIPELINES]):
        if 'sd15' in adapter_path:
            return 'sd15'
        if 'sd21' in adapter_path:
            return 'sd21'
    if any([isinstance(pipe, cls) for cls in SDXL_PIPELINES]) or 'sdxl' in adapter_path:
        return 'sdxl'
    if any([isinstance(pipe, cls) for cls in PIXART_PIPELINES]) or 'pixart' in adapter_path:
        return 'pixart'
    raise ValueError('Could not automaticly infer pipe type, please specify pipe_type')
