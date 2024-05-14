# Usage of MuLan

Install mulankit from pypi:

```bash
pip install mulankit
```

Try it now with our gradio demo:

```bash
python -m mulankit.app.webui
```

Now you could use our api to apply MuLan adapter to diffusers pipeline.

```python
from diffusers import StableDiffusionPipeline
import torch
import mulankit

pipe = StableDiffusionPipeline.from_pretrained('Lykon/dreamshaper-8')
pipe = mulankit.transform(pipe, 'mulanai/mulan-lang-adapter::sd15_aesthetic.pth')
pipe = pipe.to('cuda', dtype=torch.float16)

image = pipe('a cat').images[0]
image.save('out.jpg')
```

Please refer to [examples](examples/) for more details.
