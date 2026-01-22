# Practical Image Generation Examples with HuggingFace Diffusers

This mini-project shows image generation examples using the [HuggingFace `diffusers` library](https://huggingface.co/docs/diffusers/en/index).
The focus is on running, composing, and experimenting with modern diffusion models rather than training them from scratch.

For a technical introduction in image generation and diffusers, check these associated blog posts:

- [Image Generation (1/2): A Conceptual Guide for Developers and ML Practitioners](https://mikelsagardia.io/blog/diffusion-for-developers.html)
- [Image Generation (2/2): Hands-On Examples with HuggingFace](https://mikelsagardia.io/blog/diffusion-hands-on.html)

For environment setup instructions (Python, CUDA, dependencies), check the [`../README.md`](../README.md) in the parent folder.

This folder contains a single notebook:

:point_right: [`diffusers_and_co.ipynb`](./diffusers_and_co.ipynb)

In this notebook, the following topics and examples are covered:

- Basic usage of HuggingFace Diffusers
    - Loading pretrained diffusion pipelines and running inference with minimal configuration
    - Understanding common parameters such as `num_inference_steps`, `guidance_scale`, and seeds
- Text-to-image generation
    - Stable Diffusion XL Turbo (SDXL Turbo)
    - Comparison with higher-quality, slower models (e.g. Playground V2)
- Model distillation concepts
    - Using SDXL Turbo as an example of adversarial diffusion distillation
    - Understanding why some models can generate images in 1–4 steps
- Chained diffusion pipelines
    - Using the output of one model as the input condition for another
    - Example: SDXL Turbo -> Kandinsky image-to-image
- In-painting workflows
    - Mask-based image editing
    - Semantic consistency in diffusion-based in-painting

Some of the parts in the notebook were taken and modified from course material in the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608).

Check [`../inpainting_app/`](../inpainting_app/) for a simple in-painting application that builds on the concepts introduced here.

Mikel Sagardia, 2026.  
No guarantees.
