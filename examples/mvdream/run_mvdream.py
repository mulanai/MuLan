import torch
import kiui
import argparse
from pipeline_mvdream import MVDreamPipeline
import mulankit

pipe = MVDreamPipeline.from_pretrained(
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

pipe = mulankit.transform(
    pipe,
    adapter_path='mulanai/mulan-lang-adapter::sd21_aesthetic.pth',
)

pipe = pipe.to("cuda")

parser = argparse.ArgumentParser(description="MVDream")
parser.add_argument("--prompt", type=str, default="一只可爱的猫头鹰 3D模型")
args = parser.parse_args()

image = pipe(args.prompt, guidance_scale=5, num_inference_steps=30, elevation=0)
# grid = np.concatenate(
#     [
#         np.concatenate([image[0], image[2]], axis=0),
#         np.concatenate([image[1], image[3]], axis=0),
#     ],
#     axis=1,
# )
for i in range(4):
    kiui.write_image(f'test_mvdream_{i}.jpg', image[i])
