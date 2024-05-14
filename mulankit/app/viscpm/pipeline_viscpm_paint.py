# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np

import torch
from torch.utils.data.dataloader import default_collate
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.loaders import LoraLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, StableDiffusionPipeline
from .modeling_cpmbee import CpmBeeModel
from .tokenization_viscpmbee import VisCpmBeeTokenizer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]):, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor


class CPMBeeCollater:
    """
    针对 cpmbee 输入数据 collate, 对应 cpm-live 的 _MixedDatasetBatchPacker
    目前利用 torch 的原生 Dataloader 不太适合改造 in-context-learning
    并且原来实现为了最大化提高有效 token 比比例, 会有一个 best_fit 操作, 这个目前也不支持
    todo: 重写一下 Dataloader or BatchPacker
    """

    def __init__(self, tokenizer: VisCpmBeeTokenizer, max_len):
        self.tokenizer = tokenizer
        self._max_length = max_len
        self.pad_keys = ['input_ids', 'input_id_subs', 'context', 'segment_ids', 'segment_rel_offset',
                         'segment_rel', 'sample_ids', 'num_segments']

    def __call__(self, batch):
        batch_size = len(batch)

        tgt = np.full((batch_size, self._max_length), -100, dtype=np.int32)
        # 目前没有 best_fit, span 为全 0
        span = np.zeros((batch_size, self._max_length), dtype=np.int32)
        length = np.zeros((batch_size,), dtype=np.int32)

        batch_ext_table_map: Dict[Tuple[int, int], int] = {}
        batch_ext_table_ids: List[int] = []
        batch_ext_table_sub: List[int] = []
        raw_data_list: List[Any] = []

        for i in range(batch_size):
            instance_length = batch[i]['input_ids'][0].shape[0]
            length[i] = instance_length
            raw_data_list.extend(batch[i]['raw_data'])

            for j in range(instance_length):
                idx, idx_sub = batch[i]['input_ids'][0, j], batch[i]['input_id_subs'][0, j]
                tgt_idx = idx
                if idx_sub > 0:
                    # need to be in ext table
                    if (idx, idx_sub) not in batch_ext_table_map:
                        batch_ext_table_map[(idx, idx_sub)] = len(batch_ext_table_map)
                        batch_ext_table_ids.append(idx)
                        batch_ext_table_sub.append(idx_sub)
                    tgt_idx = batch_ext_table_map[(idx, idx_sub)] + self.tokenizer.vocab_size
                if j > 1 and batch[i]['context'][0, j - 1] == 0:
                    if idx != self.tokenizer.bos_id:
                        tgt[i, j - 1] = tgt_idx
                    else:
                        tgt[i, j - 1] = self.tokenizer.eos_id
            if batch[i]['context'][0, instance_length - 1] == 0:
                tgt[i, instance_length - 1] = self.tokenizer.eos_id

        if len(batch_ext_table_map) == 0:
            # placeholder
            batch_ext_table_ids.append(0)
            batch_ext_table_sub.append(1)

        # image
        if 'pixel_values' in batch[0]:
            data = {'pixel_values': default_collate([i['pixel_values'] for i in batch])}
        else:
            data = {}

        # image_bound
        if 'image_bound' in batch[0]:
            data['image_bound'] = default_collate([i['image_bound'] for i in batch])

        # bee inp
        for key in self.pad_keys:
            data[key] = pad(batch, key, max_length=self._max_length, padding_value=0, padding_side='right')

        data['context'] = data['context'] > 0
        data['length'] = torch.from_numpy(length)
        data['span'] = torch.from_numpy(span)
        data['target'] = torch.from_numpy(tgt)
        data['ext_table_ids'] = torch.from_numpy(np.array(batch_ext_table_ids))
        data['ext_table_sub'] = torch.from_numpy(np.array(batch_ext_table_sub))
        data['raw_data'] = raw_data_list

        return data


class VisCPMPaintBeePipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CpmBeeModel,
        tokenizer: VisCpmBeeTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker
        )

    def build_input(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        image_size: int = 512
    ):
        data_input = {'caption': prompt, 'objects': ''}
        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,
            image_bound
        ) = self.tokenizer.convert_data_to_id(data=data_input, shuffle_answer=False, max_depth=8)
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)
        data = {
            'pixel_values': torch.zeros(3, image_size, image_size).unsqueeze(0),
            'input_ids': torch.from_numpy(input_ids).unsqueeze(0),
            'input_id_subs': torch.from_numpy(input_id_subs).unsqueeze(0),
            'context': torch.from_numpy(context).unsqueeze(0),
            'segment_ids': torch.from_numpy(segment_ids).unsqueeze(0),
            'segment_rel_offset': torch.from_numpy(segment_rel_offset).unsqueeze(0),
            'segment_rel': torch.from_numpy(segment_rel).unsqueeze(0),
            'sample_ids': torch.from_numpy(sample_ids).unsqueeze(0),
            'num_segments': torch.from_numpy(num_segments).unsqueeze(0),
            'image_bound': image_bound,
            'raw_data': prompt,
        }

        uncond_data_input = {
            'caption': "" if negative_prompt is None else negative_prompt,
            'objects': ''
        }
        (
            input_ids,
            input_id_subs,
            context,
            segment_ids,
            segment_rel,
            n_segments,
            table_states,
            image_bound
        ) = self.tokenizer.convert_data_to_id(data=uncond_data_input, shuffle_answer=False, max_depth=8)
        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)
        uncond_data = {
            'pixel_values': torch.zeros(3, image_size, image_size).unsqueeze(0),
            'input_ids': torch.from_numpy(input_ids).unsqueeze(0),
            'input_id_subs': torch.from_numpy(input_id_subs).unsqueeze(0),
            'context': torch.from_numpy(context).unsqueeze(0),
            'segment_ids': torch.from_numpy(segment_ids).unsqueeze(0),
            'segment_rel_offset': torch.from_numpy(segment_rel_offset).unsqueeze(0),
            'segment_rel': torch.from_numpy(segment_rel).unsqueeze(0),
            'sample_ids': torch.from_numpy(sample_ids).unsqueeze(0),
            'num_segments': torch.from_numpy(num_segments).unsqueeze(0),
            'image_bound': image_bound,
            'raw_data': "" if negative_prompt is None else negative_prompt,
        }
        packer = CPMBeeCollater(
            tokenizer=self.tokenizer,
            max_len=max(data['input_ids'].size(-1), uncond_data['input_ids'].size(-1))
        )
        data = packer([data])
        uncond_data = packer([uncond_data])
        return data, uncond_data

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
            
        data, uncond_data = self.build_input(prompt, negative_prompt, image_size=512)
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device)
        for key, value in uncond_data.items():
            if isinstance(value, torch.Tensor):
                uncond_data[key] = value.to(self.device)

        batch, seq_length = data['input_ids'].size()
        dtype, device = data['input_ids'].dtype, data['input_ids'].device
        data['position'] = torch.arange(seq_length, dtype=dtype, device=device).repeat(batch, 1)

        batch, seq_length = uncond_data['input_ids'].size()
        dtype, device = uncond_data['input_ids'].dtype, uncond_data['input_ids'].device
        uncond_data['position'] = torch.arange(seq_length, dtype=dtype, device=device).repeat(batch, 1)

        with torch.no_grad():
            # llm_hidden_state = self.text_encoder.llm.input_embedding(data['input_ids'], data['input_id_subs'])
            _, hidden_states = self.text_encoder(
                input_ids=data['input_ids'],
                input_id_sub=data['input_id_subs'],
                position=data['position'],
                #length=data['length'],
                context=data['context'],
                sample_ids=data['sample_ids'],
                num_segments=data['num_segments'],
                segment=data['segment_ids'],
                segment_rel_offset=data['segment_rel_offset'],
                segment_rel=data['segment_rel'],
                #span=data['span'],
                #ext_table_ids=data['ext_table_ids'],
                #ext_table_sub=data['ext_table_sub'],
                #hidden_states=llm_hidden_state
            )

        with torch.no_grad():
            # uncond_llm_hidden_state = self.text_encoder.llm.input_embedding(uncond_data['input_ids'], uncond_data['input_id_subs'])
            _, uncond_hidden_states = self.text_encoder(
                input_ids=uncond_data['input_ids'],
                input_id_sub=uncond_data['input_id_subs'],
                position=uncond_data['position'],
                #length=uncond_data['length'],
                context=uncond_data['context'],
                sample_ids=uncond_data['sample_ids'],
                num_segments=uncond_data['num_segments'],
                segment=uncond_data['segment_ids'],
                segment_rel_offset=uncond_data['segment_rel_offset'],
                segment_rel=uncond_data['segment_rel'],
                #span=uncond_data['span'],
                #ext_table_ids=uncond_data['ext_table_ids'],
                #ext_table_sub=uncond_data['ext_table_sub'],
                #hidden_states=uncond_llm_hidden_state
            )

        text_hidden_states, uncond_text_hidden_states = hidden_states, uncond_hidden_states
        if self.text_encoder.trans_block is not None:
            text_hidden_states = self.text_encoder.trans_block(text_hidden_states)
            uncond_text_hidden_states = self.text_encoder.trans_block(uncond_text_hidden_states)
        bs_embed, seq_len, _ = text_hidden_states.shape
        text_hidden_states = text_hidden_states.repeat(1, num_images_per_prompt, 1)
        text_hidden_states = text_hidden_states.view(bs_embed * num_images_per_prompt, seq_len, -1)

        bs_embed, seq_len, _ = uncond_text_hidden_states.shape
        uncond_text_hidden_states = uncond_text_hidden_states.repeat(1, num_images_per_prompt, 1)
        uncond_text_hidden_states = uncond_text_hidden_states.view(bs_embed * num_images_per_prompt, seq_len, -1)

        prompt_embeds = torch.cat([uncond_text_hidden_states, text_hidden_states])
        return prompt_embeds

        # if prompt is not None and isinstance(prompt, str):
        #     batch_size = 1
        # elif prompt is not None and isinstance(prompt, list):
        #     batch_size = len(prompt)
        # else:
        #     batch_size = prompt_embeds.shape[0]

        # if prompt_embeds is None:
        #     # textual inversion: procecss multi-vector tokens if necessary
        #     if isinstance(self, TextualInversionLoaderMixin):
        #         prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        #     text_inputs = self.tokenizer(
        #         prompt,
        #         padding="max_length",
        #         max_length=self.tokenizer.model_max_length,
        #         truncation=True,
        #         return_tensors="pt",
        #     )
        #     text_input_ids = text_inputs.input_ids
        #     untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        #     if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        #         text_input_ids, untruncated_ids
        #     ):
        #         removed_text = self.tokenizer.batch_decode(
        #             untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
        #         )
        #         logger.warning(
        #             "The following part of your input was truncated because CLIP can only handle sequences up to"
        #             f" {self.tokenizer.model_max_length} tokens: {removed_text}"
        #         )

        #     if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        #         attention_mask = text_inputs.attention_mask.to(device)
        #     else:
        #         attention_mask = None

        #     prompt_embeds = self.text_encoder(
        #         text_input_ids.to(device),
        #         attention_mask=attention_mask,
        #     )
        #     prompt_embeds = prompt_embeds[0]

        # prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # bs_embed, seq_len, _ = prompt_embeds.shape
        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # # get unconditional embeddings for classifier free guidance
        # if do_classifier_free_guidance and negative_prompt_embeds is None:
        #     uncond_tokens: List[str]
        #     if negative_prompt is None:
        #         uncond_tokens = [""] * batch_size
        #     elif prompt is not None and type(prompt) is not type(negative_prompt):
        #         raise TypeError(
        #             f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
        #             f" {type(prompt)}."
        #         )
        #     elif isinstance(negative_prompt, str):
        #         uncond_tokens = [negative_prompt]
        #     elif batch_size != len(negative_prompt):
        #         raise ValueError(
        #             f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
        #             f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
        #             " the batch size of `prompt`."
        #         )
        #     else:
        #         uncond_tokens = negative_prompt

        #     # textual inversion: procecss multi-vector tokens if necessary
        #     if isinstance(self, TextualInversionLoaderMixin):
        #         uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

        #     max_length = prompt_embeds.shape[1]
        #     uncond_input = self.tokenizer(
        #         uncond_tokens,
        #         padding="max_length",
        #         max_length=max_length,
        #         truncation=True,
        #         return_tensors="pt",
        #     )

        #     if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        #         attention_mask = uncond_input.attention_mask.to(device)
        #     else:
        #         attention_mask = None

        #     negative_prompt_embeds = self.text_encoder(
        #         uncond_input.input_ids.to(device),
        #         attention_mask=attention_mask,
        #     )
        #     negative_prompt_embeds = negative_prompt_embeds[0]

        # if do_classifier_free_guidance:
        #     # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        #     seq_len = negative_prompt_embeds.shape[1]

        #     negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        #     negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        #     negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        #     # For classifier free guidance, we need to do two forward passes.
        #     # Here we concatenate the unconditional and text embeddings into a single batch
        #     # to avoid doing two forward passes
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # return prompt_embeds

    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
