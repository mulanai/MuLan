import math
import fire
import os
import random
import uuid
from datetime import datetime
from typing import Tuple
import traceback
import json

import gradio as gr
import numpy as np
import torch
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    AutoencoderKL,
    PixArtAlphaPipeline,
    DiffusionPipeline,
    Transformer2DModel,
    EulerDiscreteScheduler,
)
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

import mulankit
from mulankit.internvl.modeling_intern_text import InternVLTextModel
from mulankit.patch import (
    MAX_TOKEN_LENGTH,
    FORCE_SDXL_ZERO_POOL_PROMPT,
    FORCE_SDXL_ZERO_EMPTY_PROMPT,
    FORCE_SDXL_ZERO_NEGATIVE_POOL_PROMPT
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.environ.get('CSS_PATH', os.path.join(CURRENT_DIR, 'style.css'))
MODEL_CHOICES_PATH = os.environ.get('MODEL_CHOICES_PATH', os.path.join(CURRENT_DIR, 'meta.json'))
SAMPLE_CHOICES_PATH = os.environ.get('SAMPLE_CHOICES_PATH', os.path.join(CURRENT_DIR, 'samples.txt'))
INTERNVL_PATH = os.environ.get('INTERNVL_PATH', 'OpenGVLab/InternVL-14B-224px')
LOCAL_FILES = bool(os.environ.get('LOCAL_FILES', False))
SAFTY_CHECKER = bool(os.environ.get('SAFTY_CHECKER', True))

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "1") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_scheduler = None

DEFAULT_STYLE_NAME = "(No style)"
SCHEDULE_NAME = ["Default", "DPM-Solver", "LCM-Solver", "SDXL-Lightning", "SDXL-Lightning-1step"]
DEFAULT_SCHEDULE_NAME = "Default"
NUM_IMAGES_PER_PROMPT = 1


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    if not negative:
        negative = ""
    return p.replace("{prompt}", positive), n + negative


def load_file(path):
    if os.path.exists(path):
        return path
    repo, ckpt = path.split(':')
    path = hf_hub_download(repo, ckpt)
    return path


def load_model_auto_clean(models, model_category, pipe_path, adapter_path, vae_path, lora_path, delta_path):
    try:
        return load_model(models, model_category, pipe_path, adapter_path, vae_path, lora_path, delta_path)
    except Exception as e:
        error = traceback.format_exc()
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print('error loading model, move all models to cpu and reload')
            for k, v in models.items():
                v['None'].to('cpu')
            return load_model(models, model_category, pipe_path, adapter_path, vae_path, lora_path, delta_path)
        else:
            raise e


def load_from_single_file(model_category, pipe_path):
    model_category = model_category.lower()
    pipe_path = load_file(pipe_path)
    if model_category in ['sd15', 'sd21']:
        pipe = StableDiffusionPipeline.from_single_file(pipe_path, torch_dtype=torch.float16)
    elif model_category in ['sdxl']:
        pipe = StableDiffusionXLPipeline.from_single_file(pipe_path, torch_dtype=torch.float16)
    else:
        raise ValueError('invalid path')
    return pipe


def load_special_pipe(pipe_path):
    if pipe_path == 'openbmb/VisCPM-Paint':
        from .viscpm.tokenization_viscpmbee import VisCpmBeeTokenizer
        from .viscpm.pipeline_viscpm_paint import VisCPMPaintBeePipeline
        from .viscpm.modeling_cpmbee import CpmBeeWithTransform
        tokenizer_ = VisCpmBeeTokenizer.from_pretrained(pipe_path, trust_remote_code=True)
        text_encoder_ = CpmBeeWithTransform.from_pretrained(pipe_path, trust_remote_code=True)
        pipe = VisCPMPaintBeePipeline.from_pretrained(pipe_path, text_encoder=text_encoder_, tokenizer=tokenizer_)
    elif pipe_path == 'BAAI/AltDiffusion-m18':
        from diffusers import AltDiffusionPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler, AutoencoderKL
        from diffusers.pipelines.deprecated.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        from transformers import XLMRobertaTokenizer
        pipe = AltDiffusionPipeline(
            text_encoder=RobertaSeriesModelWithTransformation.from_pretrained(pipe_path, subfolder='text_encoder'),
            tokenizer=XLMRobertaTokenizer.from_pretrained(pipe_path, subfolder='tokenizer'),
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(pipe_path, subfolder='scheduler'),
            unet=UNet2DConditionModel.from_pretrained(pipe_path, subfolder='unet'),
            vae=AutoencoderKL.from_pretrained(pipe_path, subfolder='vae'),
            safety_checker=None,
            feature_extractor=None,
        )
        pipe = pipe.to(dtype=torch.float16)
    elif pipe_path == 'stabilityai/japanese-stable-diffusion-xl':
        pipe = DiffusionPipeline.from_pretrained("stabilityai/japanese-stable-diffusion-xl", trust_remote_code=True, torch_dtype=torch.float16)
    else:
        raise ValueError(f'could not load pipe {pipe_path}')
    return pipe


def load_delta(pipe, delta_path):
    print('load delta')
    if isinstance(pipe, PixArtAlphaPipeline):
        old_state_dict = pipe.transformer.state_dict()
        delta_state_dict = torch.load(delta_path, map_location='cpu')
        new_state_dict = {k: old_state_dict[k].to('cpu').float()+delta_state_dict[k].to('cpu').float()
                          for k in delta_state_dict.keys()}
        if 'adaln_single.emb.resolution_embedder.linear_1.weight' not in delta_state_dict.keys():
            dtype = pipe.transformer.dtype
            pipe.transformer = Transformer2DModel.from_config(pipe.transformer.config, sample_size=64)
            pipe.transformer.to(dtype)
        pipe.transformer.load_state_dict(new_state_dict)
    return pipe


def load_model(models, model_category, pipe_path, adapter_path, vae_path, lora_path, delta_path):
    gr.Info(f'Loading model {pipe_path} \n adapter {adapter_path}')
    print(models.keys(), flush=True)

    variant = None
    full_path = pipe_path
    if ':' in pipe_path:
        variant, pipe_path = pipe_path.split(':')

    # load pipe first
    if full_path not in models or adapter_path == 'None':
        if pipe_path.endswith('.safetensors'):
            pipe = load_from_single_file(model_category, pipe_path)
        else:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    pipe_path,
                    variant=variant,
                    torch_dtype=torch.float16,
                    local_files_only=LOCAL_FILES
                )
            except Exception as e:
                pipe = load_special_pipe(pipe_path)
        if SAFTY_CHECKER:
            pipe.safety_checker = None
        models[full_path] = {'None': pipe.to(device)}

    pipe = models[full_path]['None']
    if vae_path != 'None':
        if 'vae' not in models:
            models['vae'] = {}
        if vae_path not in models['vae']:
            try:
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
            except Exception as e:
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, subfolder='vae')
            models['vae'][vae_path] = vae
        vae = models['vae'][vae_path]
        pipe.vae = vae.to(pipe.vae.device)

    if delta_path != 'None':
        pipe = load_delta(pipe, delta_path)

    if adapter_path != 'None' or delta_path != 'None':
        pipe = mulankit.transform(
            pipe,
            adapter_path if adapter_path != 'None' else None,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            replace=full_path in models and len(models[full_path].keys()) > 1,
        )
        models[full_path][adapter_path] = pipe.to(device)

    if lora_path != 'None':
        pipe.load_lora_weights(load_file(lora_path))
        pipe.fuse_lora()

    global default_scheduler
    default_scheduler = pipe.scheduler

    gr.Info('Model loaded')
    print('loaded', flush=True)
    return models, gr.update(interactive=True)


def save_image(img):
    unique_name = str(uuid.uuid4()) + '.png'
    save_path = os.path.join(f'output/{datetime.now().date()}')
    os.makedirs(save_path, exist_ok=True)
    unique_name = os.path.join(save_path, unique_name)
    img.save(unique_name)
    return unique_name


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def generate(
    prompt: str,
    negative_prompt: str = "",
    style: str = DEFAULT_STYLE_NAME,
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    schedule: str = 'default',
    guidance_scale: float = 4.5,
    num_inference_steps: int = 20,
    randomize_seed: bool = False,
    num_of_image: int = 1,
    pipes=None,
    selected_model=None,
    selected_adapter=None,
    selected_vae=None,
    selected_lora=None,
    selected_delta=None,
    max_token_length=MAX_TOKEN_LENGTH,
    force_sdxl_zero_pool_prompt=FORCE_SDXL_ZERO_POOL_PROMPT,
    force_sdxl_zero_empty_prompt=FORCE_SDXL_ZERO_EMPTY_PROMPT,
    force_sdxl_zero_negative_pool_prompt=FORCE_SDXL_ZERO_NEGATIVE_POOL_PROMPT,
    use_resolution_binning: bool = True,
    upload_image=None,
    progress=gr.Progress(track_tqdm=True),
):
    pipe = pipes[selected_model][selected_adapter]

    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.manual_seed(seed)

    if schedule == 'DPM-Solver':
        if not isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif schedule == "LCM-Solver":
        if not isinstance(pipe.scheduler, LCMScheduler):
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif schedule == 'SDXL-Lightning':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif schedule == 'SDXL-Lightning-1step':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample")
    else:
        pipe.scheduler = default_scheduler

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    prompt, negative_prompt = apply_style(style, prompt, negative_prompt)

    if not use_negative_prompt:
        negative_prompt = ""

    mulankit.setup(
        max_token_length=max_token_length,
        force_sdxl_zero_pool_prompt=force_sdxl_zero_pool_prompt,
        force_sdxl_zero_empty_prompt=force_sdxl_zero_empty_prompt,
        force_sdxl_zero_negative_pool_prompt=force_sdxl_zero_negative_pool_prompt,
    )

    kwargs = {}
    if isinstance(pipe, PixArtAlphaPipeline):
        kwargs["use_resolution_binning"] = use_resolution_binning

    fail = 0
    while fail < 3:
        try:
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                num_images_per_prompt=num_of_image,
                output_type="pil",
                **kwargs,
            ).images
            break
        except Exception as e:
            error = traceback.format_exc()
            print('error loading model, move all models to cpu and reload')
            for k, v in pipes.items():
                v['None'].to('cpu')
            pipe.to('cuda')
            fail += 1
            continue

    image_paths = [save_image(img) for img in images]
    return image_paths, seed


def load_model_list(category):
    category = category.lower()
    meta = json.load(open(MODEL_CHOICES_PATH, 'r'))[category]
    return meta['model'], meta['adapter'], meta['vae'], meta['lora'], meta.get('delta', ['None'])


def reload_model_list(category):
    model_list, adapter_list, vae_list, lora_list, delta_list = load_model_list(category)
    return (
        gr.update(choices=model_list, value=model_list[0]),
        gr.update(choices=adapter_list, value=adapter_list[0]),
        gr.update(choices=vae_list, value=vae_list[0]),
        gr.update(choices=lora_list, value=lora_list[0]),
        gr.update(choices=delta_list, value=delta_list[0])
    )


def put_images_to_history(outputs, state):
    if 'history' not in state:
        state['history'] = []
    state['history'] += outputs
    return state['history']


def main(
    ip='0.0.0.0',
    port=10024,
    share=False,
):
    print('launching demo', flush=True)

    DESCRIPTION = """
    # 🌻 MuLan: Adapting Multilingual Diffusion Models for 110 + Languages
    [[Technical Report (coming soon)]](#) [[Code]](https://github.com/mulanai/MuLan) [[Model]](https://huggingface.co/mulanai)
    """
    if not torch.cuda.is_available():
        DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

    COPY_RIGHT = """
    <div align="center">
    Feel free to steal this gradio's source code. Templated by [MuLan]().
    </div>
    """

    examples = [line.strip() for line in open(SAMPLE_CHOICES_PATH, 'r').readlines()]

    MODEL_CHOICES, ADAPTER_CHOICES, VAE_CHOICES, LORA_CHOICES, DELTA_CHOICES = load_model_list(DEFAULT_CATEGORY)

    state_, _ = load_model({'history': []}, DEFAULT_CATEGORY, MODEL_CHOICES[0], ADAPTER_CHOICES[0], VAE_CHOICES[0], LORA_CHOICES[0], DELTA_CHOICES[0])

    with gr.Blocks(css=CSS_PATH) as demo:
        state = gr.State(state_)
        del state_

        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Models"):
                        model_category = gr.Radio(
                            show_label=True,
                            label="Model Category",
                            container=True,
                            interactive=True,
                            choices=CATEGORIES,
                            value=DEFAULT_CATEGORY,
                        )
                        with gr.Group():
                            selected_model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], allow_custom_value=True, label='Model')
                            selected_adapter = gr.Dropdown(choices=ADAPTER_CHOICES, value=ADAPTER_CHOICES[0], allow_custom_value=True, label='Language Adapter')

                            with gr.Accordion('More', open=False):
                                selected_vae = gr.Dropdown(choices=VAE_CHOICES, value=VAE_CHOICES[0], allow_custom_value=True, label='VAE')
                                selected_lora = gr.Dropdown(choices=LORA_CHOICES, value=LORA_CHOICES[0], allow_custom_value=True, label='LoRA')
                                selected_delta = gr.Dropdown(choices=DELTA_CHOICES, value=DELTA_CHOICES[0], allow_custom_value=True, label='Delta')

                            with gr.Row():
                                reload_button = gr.Button('Reload List')
                                load_button = gr.Button('Load')
                    with gr.Tab("Image"):
                        upload_image = gr.Image()

                    with gr.Tab("History"):
                        history_image = gr.Gallery(show_label=False, columns=1)

            with gr.Column(scale=2):
                with gr.Group():
                    with gr.Row():
                        prompt = gr.Text(
                            label="Prompt",
                            show_label=False,
                            max_lines=1,
                            placeholder="Enter your prompt",
                            container=False,
                        )
                        run_button = gr.Button("Run", scale=0)
                    result = gr.Gallery(columns=NUM_IMAGES_PER_PROMPT, show_label=False, height=520, interactive=False)

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Basic"):
                        inference_steps = gr.Slider(
                            label="inference steps",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=25,
                        )

                        with gr.Group():
                            resolution_bining = gr.Checkbox(label="Resolution Bining", value=True)
                            with gr.Row():
                                width = gr.Slider(
                                    label="Width",
                                    minimum=256,
                                    maximum=MAX_IMAGE_SIZE,
                                    step=32,
                                    value=512,
                                )
                                height = gr.Slider(
                                    label="Height",
                                    minimum=256,
                                    maximum=MAX_IMAGE_SIZE,
                                    step=32,
                                    value=512,
                                )
                            num_of_image = gr.Slider(
                                label="Number of Images",
                                minimum=1,
                                maximum=25,
                                step=1,
                                value=1,
                            )
                            seed = gr.Slider(
                                label="Seed",
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                value=0,
                            )
                            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                        with gr.Group():
                            use_negative_prompt = gr.Checkbox(label="Use Negative Prompt", value=False, visible=True)
                            negative_prompt = gr.Text(
                                max_lines=1,
                                placeholder="Enter a negative prompt",
                                visible=False,
                                show_label=False,
                            )
                    with gr.Tab("Style"):
                        style_selection = gr.Radio(
                            show_label=True,
                            container=True,
                            interactive=True,
                            choices=STYLE_NAMES,
                            value=DEFAULT_STYLE_NAME,
                            label="Image Style",
                        )
                    with gr.Tab("Scheduler"):
                        schedule = gr.Radio(
                            show_label=True,
                            container=True,
                            interactive=True,
                            choices=SCHEDULE_NAME,
                            value=DEFAULT_SCHEDULE_NAME,
                            label="Sampler Schedule",
                            visible=True,
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=0,
                            maximum=10,
                            step=0.1,
                            value=4.5,
                        )
                    with gr.Tab("Others"):
                        max_token_length = gr.Slider(
                            label="Max Token Length",
                            minimum=80,
                            maximum=500,
                            step=20,
                            value=MAX_TOKEN_LENGTH,
                        )
                        force_sdxl_zero_pool_prompt = gr.Checkbox(label="force_sdxl_zero_pool_prompt", value=FORCE_SDXL_ZERO_POOL_PROMPT)
                        force_sdxl_zero_empty_prompt = gr.Checkbox(label="force_sdxl_zero_empty_prompt", value=FORCE_SDXL_ZERO_EMPTY_PROMPT)
                        force_sdxl_zero_negative_pool_prompt = gr.Checkbox(label="force_sdxl_zero_negative_pool_prompt", value=FORCE_SDXL_ZERO_NEGATIVE_POOL_PROMPT)

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=[result, seed],
            fn=generate,
            examples_per_page=5,
        )

        # gr.Markdown(COPY_RIGHT)

        model_category.change(reload_model_list, inputs=[model_category], outputs=[selected_model, selected_adapter, selected_vae, selected_lora, selected_delta])

        use_negative_prompt.change(
            fn=lambda x: gr.update(visible=x),
            inputs=use_negative_prompt,
            outputs=negative_prompt,
        )

        num_of_image.change(
            fn=lambda x: gr.update(columns=int(math.sqrt(x))),
            inputs=num_of_image,
            outputs=result,
        )

        load_button.click(
            fn=lambda x: gr.update(interactive=False),
            inputs=run_button,
            outputs=run_button,
        ).then(
            fn=load_model_auto_clean,
            inputs=[state, model_category, selected_model, selected_adapter, selected_vae, selected_lora, selected_delta],
            outputs=[state, run_button],
        )

        reload_button.click(reload_model_list, inputs=[model_category], outputs=[selected_model, selected_adapter, selected_vae, selected_lora, selected_delta])

        gr.on(
            triggers=[
                prompt.submit,
                negative_prompt.submit,
                run_button.click,
            ],
            fn=generate,
            inputs=[
                prompt,
                negative_prompt,
                style_selection,
                use_negative_prompt,
                seed,
                width,
                height,
                schedule,
                guidance_scale,
                inference_steps,
                randomize_seed,
                num_of_image,
                state,
                selected_model,
                selected_adapter,
                selected_vae,
                selected_lora,
                selected_delta,
                max_token_length,
                force_sdxl_zero_pool_prompt,
                force_sdxl_zero_empty_prompt,
                force_sdxl_zero_negative_pool_prompt,
                resolution_bining,
                upload_image,
            ],
            outputs=[result, seed],
        ).then(
            put_images_to_history,
            inputs=[result, state], outputs=[history_image]
        )

    print('launched', flush=True)
    demo.launch(server_name=ip, server_port=port, share=share)


if __name__ == "__main__":
    text_encoder = InternVLTextModel.from_pretrained(INTERNVL_PATH, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, use_fast=False, add_eos_token=True)
    tokenizer.pad_token_id = 0  # set pad_token_id to 0
    tokenizer.model_max_length = 80

    style_list = [
        {
            "name": "(No style)",
            "prompt": "{prompt}",
            "negative_prompt": "",
        },
        {
            "name": "Cinematic",
            "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
            "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
        },
        {
            "name": "Photographic",
            "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
            "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
        },
        {
            "name": "Anime",
            "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
            "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
        },
        {
            "name": "Manga",
            "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
            "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
        },
        {
            "name": "Digital Art",
            "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
            "negative_prompt": "photo, photorealistic, realism, ugly",
        },
        {
            "name": "Pixel art",
            "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
            "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
        },
        {
            "name": "Fantasy art",
            "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
            "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
        },
        {
            "name": "Neonpunk",
            "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
            "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
        },
        {
            "name": "3D Model",
            "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
            "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
        },
    ]
    styles = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
    STYLE_NAMES = list(styles.keys())

    CATEGORIES = ['SD15', 'SD21', 'SDXL', 'Pixart', "Other"]
    DEFAULT_CATEGORY = CATEGORIES[0]

    fire.Fire(main)
