# Examples of Diffusion Models

This repository contains some example applications in which diffusion-based image generation models are used.

Each example/subproject is independent; specific sources, necessary packages, setup & Co. are explained inside each subproject folder.

- [`ddpm/`](./ddpm/README.md)
- [`diffusers/`](./diffusers/README.md)
- [`inpainting_app/`](./inpainting_app/README.md)

## Setup

In the following, I provide a recipe to set up a [conda](https://docs.conda.io/en/latest/) environment with the necessary packages. Note that a GPU (with at least 12 GB or memory) is required to train and/or using the models.

```bash
# Create the necessary Python environment
# NOTE: specific folders might require their own environment
# and have their own requirements.txt
conda env create -f conda.yaml
conda activate genai

# If you have CUDA, install CUDA support with the propper CUDA version, e.g. v12.1
pip install torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/cu121
# OTHERWISE, install CPU version -- BUT many examples won't work!
pip install torch torchvision torchaudio torchtext
# Compile rest of dependencies and install them
pip-compile requirements.in
pip install -r requirements.txt

# If we need a new dependency,
# add it to requirements.in 
# (WATCH OUT: try to follow alphabetical order)
# And then:
pip-compile requirements.in
pip install -r requirements.txt
```

For these examples, I used 

- a Lenovo ThinkPad P14s Gen 2i running Ubuntu 25.04
- and a GeForce NVIDIA RTX 3060 (12 GB) with CUDA driver version 580 and toolkit version 13.0.

## Related Resources

Other related repositories of mine:

- My personal notes on the O'Reilly book [Generative Deep Learning, 2nd Edition, by David Foster](https://github.com/mxagar/generative_ai_book)
- My personal notes on the Udacity Nanodegree [Generative AI](https://github.com/mxagar/generative_ai_udacity)
- [HuggingFace Guide: `mxagar/tool_guides/hugging_face`](https://github.com/mxagar/tool_guides/tree/master/hugging_face)
- [Deep Learning Methods for CV and NLP: `mxagar/computer_vision_udacity/CVND_Advanced_CV_and_DL.md`](https://github.com/mxagar/computer_vision_udacity/blob/main/03_Advanced_CV_and_DL/CVND_Advanced_CV_and_DL.md)
- [Notes on Generative Adversarial Networks (GANs): `mxagar/deep_learning_udacity/05_GAN/DLND_GANs.md`](https://github.com/mxagar/deep_learning_udacity/blob/main/05_GAN/DLND_GANs.md)


Mikel Sagardia, 2025  
No guaranties.
