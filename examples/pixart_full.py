import torch
from diffusers import PixArtAlphaPipeline, Transformer2DModel
from diffusers.utils import make_image_grid

import mulankit

pipe = PixArtAlphaPipeline.from_pretrained('PixArt-alpha/PixArt-XL-2-1024-MS')
# pipe = PixArtAlphaPipeline.from_pretrained('PixArt-alpha/PixArt-XL-2-512x512')

# TODO: we only need to load internvl text encoder
pipe = mulankit.transform(
    pipe,
    adapter_path='mulanai/mulan-lang-adapter::pixart.pth',
)

pipe.transformer = Transformer2DModel.from_pretrained('mulanai/mulan-pixart', subfolder='pixart-alpha-1024-ms')
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

make_image_grid(images, 2, 3).save(
    # 'pixart_512.jpg'
    'pixart_1024.jpg'
)