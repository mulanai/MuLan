from setuptools import setup, find_packages

deps = [
    'fire',
    'diffusers',
    'huggingface_hub',
    'transformers',
    'numpy',
    'gradio',
    'timm',
    'sentencepiece',
    'peft',
    'einops',
]

setup(
    name='mulankit',
    description='MuLan: Adapting Multilingual Diffusion Models for 110+ Languages.',
    url='https://github.com/mulanai/MuLan',
    author='Zeqiang Lai',
    author_email='laizeqiang@outlook.com',
    packages=find_packages(),
    version='1.0.2',
    include_package_data=True,
    install_requires=deps,
    package_data={
        'mulankit': [
            'app/style.css',
            'app/meta.json',
            'app/samples.txt',
        ],
    },
)
