# Segementation-Aware In-Painting Application

This mini-project demonstrates a segmentation-aware image in-painting application built by composing pretrained vision models.
It combines zero-shot image segmentation with diffusion-based in-painting, enabling users to selectively re-generate foreground or background regions of an image while preserving visual and semantic consistency.

For a technical introduction in image generation and diffusers, check these associated blog posts:

- [Image Generation: A Conceptual Guide for Developers and ML Practitioners](https://mikelsagardia.io/blog/diffusion-for-developers.html)
- [Image Generation: Hands-On Examples with HuggingFace](https://mikelsagardia.io/blog/diffusion-hands-on.html)

For an environment setup, check the [`../README.md`](../README.md) in the folder above.

The folder contains two files:

- [`app.py`](./app.py): A lightweight Python module that wraps the notebook logic into a Gradio-based web application:
    - Defines the GUI layout (image canvases, point selection, text prompts, buttons)
    - Connects UI events to segmentation and in-painting functions
    - Launches a local web app for interactive experimentation

- [`inpainting.ipynb`](./inpainting.ipynb): A development and experimentation notebook where the core logic is implemented and tested:
    - Loading and running the Segment Anything Model (SAM) for zero-shot segmentation
    - Generating binary masks from user-provided points
    - Running Stable Diffusion XL In-Painting on selected regions
    - Iterating on prompts, masks, and model parameters interactively

The notebook serves as a playground and reference implementation, while `app.py` turns the same logic into a reusable, interactive application. Some of the parts in this Python module were taken and modified from course material in the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608).

### How to Use It

- User input
    - Upload an image
    - Select points indicating the region of interest (foreground or background)
- Segmentation is run: The [Segment Anything Model (SAM)](https://huggingface.co/docs/transformers/v4.41.1/model_doc/sam) generates a binary mask based on the selected points
- In-painting
    - A [Stable Diffusion XL in-painting pipeline](https://huggingface.co/docs/diffusers/v0.22.1/api/pipelines/stable_diffusion/inpaint) re-generates the selected region
    - The unmasked region is preserved to maintain spatial and semantic consistency
- Interactive output: Results are displayed instantly in the browser via Gradio


Mikel Sagardia, 2026.  
No guarantees.
