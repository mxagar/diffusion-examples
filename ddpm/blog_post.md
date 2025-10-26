# Image Generation with Diffusion

<!--
# Log in/out to Docker Hub
docker logout
docker login

# Pull the official image (first time)
docker pull excalidraw/excalidraw

# Start app
docker run --rm -dit --name excalidraw -p 5000:80 excalidraw/excalidraw:latest
# Open browser at http://localhost:5000

# Stop
docker stop excalidraw
docker rm excalidraw
docker ps

-->

Blog Post 1  
Title: An Intorduction to Image Generation with Diffusers (1/2)  
Subtitle: A Conceptual Guide for Developers

Blog Post 2  
Title: An Intorduction to Image Generation with Diffusers (2/2)  
Subtitle: Hands-On Examples with Hugging Face


Concepts to explain:

- Discriminative models learn decision boundaries
- Generative models learn distributions where they sample from
- Realistic images are really points in a large vast of noise
- Images are represented in a latent space as high-dimensional vectors
- Conditional (text2image, image2image) and unconditional
- Multimodal (text and image)
- Main image generation meythods: VAEs, GANs, Diffusion
- The impossible triangle: coverage, quality, speed


In machine learning, any sample or data point/instance can be represented as a vector of features $$x$$.
These features could be the RGB values of the pixels of an image, the words (tokens) of a text,
or other more compressed representations that still contain a differentiable meaning.

In that context, up until recently, mainly **discriminative models** were used in the industry.
These models are trained with annotated data, for instance, the classes the samples belong to: $$x$$ is of class $$y = $$ *cat* or *dog*.
Then, the model learns decision boundaries, so that it is able to predict the class of a new unseen sample.
Mathematically, we can represent that as $$y = f(x)$$, or better as $$p(y|x)$$,
i.e., the probability $$p$$ of each class $$y$$ given the instance/sample $$x$$.

On the other hand, in the past years, **generative models** have gained popularity.
These do not capture decision boundaries explicitly, instead, they learn the probability distribution of the data.
Therefore, they can cample in those distributions and generate new unseen samples.
Following the same mathematical notation, we can represent that as $$p(x)$$ or $$p(x, y)$$, if the classes are considered.

![Discriminative vs. Generative Models](../assets/discriminative_vs_generative.png)

A dataset of 2D samples $$x$$ (i.e., 2 features) used to fit a discriminative and a generative model.
Discriminative models learn decision boundaries and are able to predict the class $$y$$ of new, unseen instance.
Generative models learn the data distribution $$p(x)$$ and are able to sample new unseen instances.
Image by the author.

In terms of *ease of control*, generative models can be of two main types:

- *Unconditional*, $$p(x)$$: These models learn the data distribution $$p(x)$$ and blindly create samples from it, without much control. You can check [these artificial faces](https://thispersondoesnotexist.com/), as an example.
- *Conditional*, $$p(x|\textrm{condition})$$: They generate new samples compliant to an input we provide, e.g., a class, a text prompt or an image.
In the realm of the text modality, probably the most popular generative model is OpenAI's [(Chat)GPT](#), which is able to produce words (tokens), and subsequently conversations, conditioned by a prompt or user instruction.
When it comes to the modality of images, it's difficult to point to a single winner, but common models are
[Dall-E](#), [Midjourney](#), or [Stable Diffusion]([#](https://huggingface.co/spaces/google/sdxl)) &mdash; all of them are `text-to-image` conditional models.

In terms of the *modalities* they can work with, generative models can be:

- *Uni-modal*: These models can handle/produce samples of a single modality, e.g., text or images.
- *Multi-modal*: They are able to work with instances of different modalities simulatneously.
They can achieve that by creating a common *latent space* for all modalities, or mappings between them.
Latent spaces are compressed vector spaces that capture the semantics of the vectors that form them.
As a result, given a text-image multi-modal model, we can ask it about something on an image.
Notable examples are [GPT4-Vision](https://openai.com/research/gpt-4v-system-card) and [LLaVA](https://huggingface.co/spaces/badayvedat/LLaVA)

## Types of Generation Models

![VAEs and GANs](../assets/vae_and_gan.png)

![Denoising Diffusion](../assets/diffusion_idea.png)

![Impossible Triangle](../assets/impossible_triangle.png)


## Denoising Diffusion




![Denoising UNet](../assets/denoising_unet.png)

