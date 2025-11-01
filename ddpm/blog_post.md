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

## Types of Image Generation Approaches

There are three main families of generative approaches for image generation:

- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Denoising Diffusion Probabilistic Models (Diffusers)

[**Autoencoders**](#) are architectures which compress the input $x$ to a lower dimensional latent vector $z$ and then they expand it again to try to recreate $x$. The compression side is called *encoder*, the middle layer which produces the latent vector is the *bottleneck*, and the expansion side is named the *decoder*. As mentioned, the final output $x'$ tries to approxiumate $x$ as best as possible; the differential of the error is used to update the weights of all the layers. Many types of layers and configurations can be used for the encoder & decoder parts; e.g., with images often [convolutional layers](#), [pooling](#), [dropout](#), and [batch normalization](#) are used to compress the image, whereas the expansion usually is implemented with [transpose convolutions](#).

[**Variational Autoencoders (VAEs)**](#) are autoencoders in which the elements of the latent $z$ vector are Gaussian distributions, i.e., for each latent element, they produce a mean and a variance, and then a value is sampled from that element distribution to produce the latent values. The practical effect is that VAEs produce latent spaces in which it is possible to interpolate with much smaller discontinuities than with non-variational autoencoders.

VAEs have been typically implemented for compression, denoising and anomaly detection; even though they can generate new samples using only their decoder, they usually have a lesser realism. However, they are fundamental to understand generative models, since they intuitively introduce most of the concepts revisited by posterior approaches. If you want to check some examples, have a look at [A](#), [B](#), [C](#).

[**Generative Adversarial Networks (GANs)**](#) were presented by Goodfellow et al. in 2014 and they meant a significant advancement in realistic image generation. They also have a decoder, named *generator*, and an encoder, called *discriminator*, but they are arranged differently, as shown in the figure below. 

The *generator* $G$ tries to generate realistic images as if the belonged to a real dataset distribution, starting with latent vector $z$ expanded from a noise seed. On the other hand, the *discriminator* $D$ tries to determine whether an image $x$ is real or fake (i.e., generated: $x' = G(z)$). Usually, $D$ and $G$ have mirrored architectures and their layers are equivalent to the ones used in VAEs.

The training phase looks as follows:

- First, $D$ is trained: we create batches of real images $x$ and batches of fake images $x' = G(z)$, pass them to the discriminator $D$ (i.e., we get $D(x), D(G(z))$), and compute the error with respect to the correct labels. That error is backpropagated to update only the weights of $D$.
- Second, $G$ is trained: we create new fake images with the generator $G$ and pass them to the discriminator $D$. The prediction error is backpropagated to update only the weights of $G$.
- Both steps are alternated for several iterations, until the error metrics don't improve.

Once the model is trained, the inference is done with the generator $G$ alone.

GANs are notoriously difficult to train, [because of several factors](#) out of the scope of this blog post. Fortunately, [guidelines](#) which aid the training process have been proposed. Also, method improvements have been presented, such as the [Wasserstein GAN with Gradient Penalty](#), which alleviates the major training difficulties, and [conditional GANs](#), which provide control to the user during generation (e.g., create male or female faces).

![VAEs and GANs](../assets/vae_and_gan.png)

Variational Autoencoders (VAEs, left) and Generative Adversarial Networks (GANs, right) were the most popular generative models up until the advent of Diffusers in the past years. VAEs learn to compress and decompress inputs with an *encoder-decoder* architecture which produces a *latent* space with the compressed samples. GANs learn to produce realistic samples adversarialy: they generate fake samples (with the *generator*) and try to fool a binary classifier (the *discriminator*) which needs to differentiate between real and fake samples. 

Finally, we arrive at the [**Denoising Diffusion Probabilistic Models (Diffusers)**](https://arxiv.org/abs/2006.11239), presented by Ho et al. in 2020.
In few years they have outperformed GANs for image generation and have become the standard method for the task. The core idea is that we train a model which takes

- a noisy image $x_t$ (in the beggining it will be a pure random noise map)
- and an associated noise variance $\beta_t$ (in the beginning it will be a high variance value)

and it predicts the noise map $\epsilon_t$ overlaid on the image, so that we can substract it to from the noisy image get the noise-free image $x_0$.
The process is performed in small, gradual steps, and following a noise rate schedule which decreases the value of $\beta_t$.

As we can see in the figure below, two iterative phases are distinguished, which consist each of them in $T$ steps:

1. **Forward diffusion, used during training** &mdash; Starting with a real clean image $x_0$, we add a noise map $\epsilon$ to it, generated from a variance value $\beta$. Then, we pass the noisy image through a *UNet* model, which should predict the added noise map $\epsilon$. The error is backpropagated to update the weights. The image at step $t$ does not only contain the noise added in the previous step, but also the noise accumulated from prior steps. The forward process is done gradually in around $T = 1000$ steps, in which the noise is added following a cosine schedule.
2. **Reverse diffusion, used during inference** &mdash; We perform the inference starting with a pure, random noise map. In each step, we pass the noisy image through the *UNet* to predict the step noise map $\epsilon_t$, substract it to the image $x_t$ and obtain the next, less noisy image $x_{t-1}$. The process is repeated for around $T in [20,100]$ steps, until we geat a clear new image $x_0$.

![Denoising Diffusion](../assets/diffusion_idea.png)

In denoising diffusion models a UNet encoder-decoder model is trained to predict the noise in an image. To that end, during training (forward diffusion), noise is gradually added to an image and we query the model to predict the noise map. During inference (reverse diffusion), we start with a pure noise map and query the model to remove the noise step by step &mdash; until we get a clean new image!

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

So which of these approaches should we use?

To answer that question, we need to consider that generative models are usually evaluated in terms of [three competing properties, which lead to a so-called generative learning trilemma](https://arxiv.org/pdf/2112.07804):

- **Quality**: if the distributions of the generated images and real images are close, the quality is considered good. In practice, pretrained CNNs can be used to create image embeddings, leading to vector distributions. Then, the difference between the distributions is measured with the [Wasserstein distance metric](https://en.wikipedia.org/wiki/Wasserstein_metric). GANs and Diffusers have a particularly good quality, whereas VAEs have often a lesser one.
- **Coverage**: this measures how diverse the captured distributions are, i.e., the number of modi or peaks we have in the vector spaces; for instance, in a dataset of dog images, we would expect as many dog breeds as possible, which would be represented as many dense regions differentiable from each other. VAEs and Diffusers have good coverage, whereas GANs tend to deliver less diverse results.
- **Speed**: this refers to the sampling speed, i.e., how fast we can create new images. GANs and VAEs are the fastest approaches, while Diffusers require longer computation times.

![Impossible Triangle](../assets/impossible_triangle.png)

Generative learning trilemma: sample diversity coverage, generation quality and generation speed are competing properties of generative methods &mdash; or is [Stable Diffusion](#) the solution to that trilemma?
Image reproduced by the author, but based on the work by [Xiao et al., 2022](https://arxiv.org/pdf/2112.07804).

However, [Stable Diffusion](https://arxiv.org/abs/2112.10752) [Stable Diffusion XL](https://arxiv.org/abs/2307.01952)

In the next section, I will go deeper into the topic of **denoising diffusion models** and will introduce how **Stable Diffusion** works.

## Deep Dive into Denoising Diffusion

Now, let's go deeper into the 

<!--
Some corollary notes on training and inference:

    In the forward diffusion process (traning of the U-Net), the noise map (e_t, epsilon) is computed from the variance scalar (b_t, beta) and added to the image to obtain a noisy image x_t. Then, the noisy image is passed to the U-Net. The U-Net tries to guess the noise map, and the prediction error is used to update the weights via backpropagation.
    In the reverse diffusion process (inference), the noise map is not computed using a formula which depends on the variance, but it is predicted by the U-Net model. Then, this noise map is substracted to the image to remove a noise step from it.

More details

The forward diffusion function q adds the required noise between two consecutive noisy images (x_(t-1) -> x_(t)); it is defined as:

x_t = q(x_t | x_(t-1)) = sqrt(1-b_t) * x_(t-1) + sqrt(b_t) * e_(t-1)

where

x: image
t: step in noise adding schedule
b, beta: variance
e, epsilon: Gaussian map with mean 0, standard deviation 1

However, a reparametrization trick allows to formulate the function such as any stage of the noisy image (t) can be computed from the original, noise-free image (x_0):

x_t = q(x_t | x_0) = sqrt(m(a_t)) * x_0 + sqrt(1 - m(a_t)) * e_t

where

a_t, alpla_t = 1 - b_t
m(a_t) = prod(i=0:t; a_i)

Note that

    the value m(a_t) is the signal ratio
    whereas the 1 - m(a_t) is the noise ratio.

Additionally, e_t is exactly what the U-Net model is trying to output given b_t and the noisy image!

Diffusion schedules vary the signal and noise ratios in such a way that during training

    the signal ratio decreases (i.e., increase the value of t in m(a_t)) from 1-offset to 0 following a cosine function
    and the noise ratio increases from offset to 1 folllowing the complementary cosine function.

During inference or image generation the schedule is reversed.

The reverse diffusion function p removes noise between two consecutive noisy images (x_(t) -> x_(t-1)); it has this form:

x_(t-1) = p(x_(t-1) | x_t) = f(m(a_t); x_t)

This reverse diffusion function is derived from the reparametrized forward diffusion and other concepts; it has a simple linear form but fractional coefficients dependent on the signal ratio 1 - m(a_t) and the noise ratio m(a_t). More importantly, it uses the noise map e which is predicted by the trained U-Net, i.e., the U-Net is trained using the forward diffusion to be able to create the necessary noise map value to be substracted in the reverse diffusion.

It's worth mentioning that both the forward and reverse diffusion processes are Gaussian, meaning that the noise added in the forward process and removed in the reverse process is Gaussian. This Gaussian structure allows the formulation of the reverse process based on Bayes' theorem.

The U-Net noise model has the following properties:

    Input: noisy image x_t at step t, as well as variance b_t.
        The variance scalar is expanded to be a vector using sinusoidal embedding. Sinusoidal embedding is basically a R -> R^n map which for each unique scalar generates a unique and different vector. It is related to the sinusoidal embedding from the Transformers paper, but there it was used to add positional embeddings. Later, in the NeRF paper, sinusoidal embeddings were modified to map scalars to vectors, as done in the diffusion U-Net model.
        The image and the variance vector are concatenated in the beginning of the network.
    Output: noise map e_t corresponding to the input; if we substract e_t to the noisy image x_t we should obtain the noise-free image x_0. However, obviously, that works better if done progressively in the reverse diffusion function.
    As in every U-Net, the initial tensor is progressively reduced in spatial size while its channels are increased; then, the reduced vector is expanded to have a bigger spatial size but less channels. The final tensor has the same shape as the input image. The architecture consists of these blocks:
        ResidualBlock: basic block used everywhere which performs batch normalization and 2 convolutions, while adding a skip connection between input and output, as presented in the ResNet architecture. Residual blocks learn the identity map and allow for deeper network, since the vanishing gradient issue is alleviated.
        DownBlock: two ResidualBlocks are used and an average pooling so that the image size is decreased and the channels are increased.
        UpBlock: upsampling is applied to the image to increase its spatial size and two ResidualBlocks are applied so that the channels are decreased.
        Skip connections: the ouput of each ResidualBlocks in a DownBlock is passed to the associated UpBlock with same tensor size, where the tensors are concatenated.
    Two networks are maintained: the usual one with the weights computed during gradient descend and the Exponential Moving Average (EMA) network, which contains the EMA of the weights. The EMA network is not that susceptible to spikes and fluctuations.

-->

![Denoising UNet](../assets/denoising_unet.png)

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>


[Stable Diffusion](https://arxiv.org/abs/2112.10752) Rombach et al. 2021

[Stable Diffusion XL](https://arxiv.org/abs/2307.01952) Podell et al. 2023

## Conclusions

:construction: TBD.

