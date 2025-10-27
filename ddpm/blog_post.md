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

<div style="height: 20px;"></div>
<div align="center" style="border: 1px solid #e4f312ff; background-color: #fcd361b9; padding: 1em; border-radius: 6px;">
<strong>
This is the first post of a series of two.
You can find the <a href="https://mikelsagardia.io/posts/">second part here</a>.
Also, you can find the accompanying code <a href="https://github.com/mxagar/diffusion-examples">this GitHub repository</a>.
</strong>
</div>
<div style="height: 30px;"></div>


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

In machine learning, any sample or data point/instance can be represented as a vector of features $x$.
These features could be the RGB values of the pixels of an image or the words (tokens) of a text represented as an index of a vocabulary.
In deep learning, these vectors are often transformed into *embeddings* or *latent* vectors, which are compressed representations that still contain a differentiable meaning.

![Image and Text Embeddings](../assets/embeddings.png)

Any sample or data point of any modality can represented as an n-dimensional vector $x$ in machine learning; in the figure, images and words (tokens) are represented as 2D vector embeddings. These embeddings contain conceptual information in a compressed form. When semantics and/or similarities between samples are captured, algebraic operations can be used with the vectors, resulting in coherent, logical outputs. Image by the author.

Up until recently, mainly **discriminative models** have been used, which predict properties of those embeddings.
These models are trained with annotated data, for instance, the classes the samples belong to: $x$ is of class $y = $ *cat* or *dog*.
Then, the model learns decision boundaries, so that it is able to predict the class of a new unseen sample.
Mathematically, we can represent that as $y = f(x)$, or better as $p(y|x)$,
i.e., the probability $p$ of each class $y$ given the instance/sample $x$.

On the other hand, in the past years, **generative models** have gained popularity.
These do not capture decision boundaries explicitly, instead, they learn the probability distribution of the data.
Therefore, they can sample in those distributions and generate new unseen examples.
Following the same mathematical notation, we can say that the models learn $p(x)$ or $p(x, y)$, if the classes are considered.

![Discriminative vs. Generative Models](../assets/discriminative_vs_generative.png)

A dataset of 2D samples $x$ (i.e., 2 features) used to fit a discriminative and a generative model.
Discriminative models learn decision boundaries and are able to predict the class $y$ of new, unseen instance.
Generative models learn the data distribution $p(x)$ and are able to sample new unseen instances.
Image by the author.

In terms of *ease of control*, generative models can be of two main types:

- *Unconditional*, $p(x)$: These models learn the data distribution $p(x)$ and blindly create samples from it, without much control. You can check [these artificial faces](https://thispersondoesnotexist.com/), as an example.
- *Conditional*, $p(x|\textrm{condition})$: They generate new samples compliant to an input we provide, e.g., a class, a text prompt or an image.
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

> Discriminative models learn to predict concrete values of a data sample (e.g., a class or a value), whereas generative models learn the data distribution and are able to sample it.
> Additionally, this sampling can often be conditioned by a prompt.

## Types of Image Generation Models

There are three main families of generative models for image generation:

- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Denoising Diffusion Probabilistic Models (Diffusers)



![VAEs and GANs](../assets/vae_and_gan.png)



![Denoising Diffusion](../assets/diffusion_idea.png)



<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

So which of these models should we use? To answer that question, we need to consider that generative models are usually evaluated in terms of [three competing properties, which lead to a so-called generative learning trilemma](https://arxiv.org/pdf/2112.07804):

- **Quality**: if the distributions of the generated images and real images are close, the quality is considered good. In practice, pretrained CNNs can be used to create image embeddings, leading to vector distributions. Then, the difference between the distributions is measured with the [Wasserstein distance metric](https://en.wikipedia.org/wiki/Wasserstein_metric). GANs and Diffusers have a particularly good quality, whereas VAEs have often a lesser one.
- **Coverage**: this measures how diverse the captured distributions are, i.e., the number of modi or peaks we have in the vector spaces; for instance, in a dataset of dog images, we would expect as many dog breeds as possible, which would be represented as many dense regions differentiable from each other. VAEs and Diffusers have good coverage, whereas GANs tend to deliver less diverse results.
- **Speed**: this refers to the sampling speed, i.e., how fast we can create new images. GANs and VAEs are the fastest approaches, while Diffusers require longer computation times.

![Impossible Triangle](../assets/impossible_triangle.png)

Generative learning trilemma: sample diversity coverage, generation quality and generation speed are competing properties of generative methods &mdash; or is [Stable Diffusion](#) the solution to that trilemma?
Image reproduced by the author, but based on the work by [Xiao et al., 2022](https://arxiv.org/pdf/2112.07804).

However, [Stable Diffusion](#)

In the next section, I will go deeper into the topic of **denoising diffusion models** and will introduce how **Stable Diffusion** works.

## Denoising Diffusion




![Denoising UNet](../assets/denoising_unet.png)

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>


Stable Diffusion


## Code

:construction: TBD.

## Conclusions

:construction: TBD.

