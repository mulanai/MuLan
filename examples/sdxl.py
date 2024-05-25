import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import make_image_grid

import mulankit

# pretrained_model_name_or_path = 'playgroundai/playground-v2-1024px-aesthetic'
# pretrained_model_name_or_path = 'Lykon/dreamshaper-xl-1-0'
pretrained_model_name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
pipe = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path, 
    safety_checker=None, local_files_only=True,
    force_zeros_for_empty_prompt=False,
)
mulankit.setup(force_sdxl_zero_empty_prompt=False, force_sdxl_zero_pool_prompt=False)
pipe = mulankit.transform(
    pipe,
    adapter_path='mulanai/mulan-lang-adapter::sdxl_aesthetic.pth',
)
pipe = pipe.to('cuda', dtype=torch.float16)

images = pipe(
    [
        'a cat',
        '一辆红色汽车',
        '한 미녀가 모자를 쓰고 있다',
        'Pirate ship sailing into a bioluminescence sea with a galaxy in the sky, epic, 4k, ultra',
        'كلب على شاطئ البحر',
        'Космонавты едут верхом.',
    ],
    guidance_scale=5,
    num_inference_steps=20,
    generator=torch.manual_seed(12345),
).images
make_image_grid(images, 2, 3).save('{}.jpg'.format(pretrained_model_name_or_path.replace('/','_')))
