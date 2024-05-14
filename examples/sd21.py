import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid

import mulankit

pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', safety_checker=None, local_files_only=True)
pipe = mulankit.transform(
    pipe,
    adapter_path='mulanai/mulan-lang-adapter::sd21_aesthetic.pth',
)
pipe = pipe.to('cuda', dtype=torch.float16)

images = pipe(
    prompt=[
        '一辆红色汽车',
        '한 미녀가 모자를 쓰고 있다',
        'Pirate ship sailing into a bioluminescence sea with a galaxy in the sky, epic, 4k, ultra',
        'كلب على شاطئ البحر',
        'Космонавты едут верхом.',
        '色とりどりの積み木で作られたアルパカ、サイボパンク',
    ],
    generator=torch.manual_seed(0),
).images
make_image_grid(images, 2, 3).save('sd21.jpg')
