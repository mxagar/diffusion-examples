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
This is the second post of a series of two.
You can find the <a href="https://mikelsagardia.io/posts/">first part here</a>.
Also, you can find the accompanying code <a href="https://github.com/mxagar/diffusion-examples/diffusers">this GitHub repository</a>.
</strong>
</div>
<div style="height: 30px;"></div>


Blog Post 1  
Title: An Intorduction to Image Generation with Diffusers (1/2)  
Subtitle: A Conceptual Guide for Developers

Blog Post 2  
Title: An Intorduction to Image Generation with Diffusers (2/2)  
Subtitle: Hands-On Examples with Hugging Face

<p align="center">
<img src="../assets/ai_drawing_ai_dallev3.png" alt="An AI drawing an AI drawing a AI. Image generated using Dalle-E v3" width="1000"/>
<small style="color:grey">An AI drawing an AI drawing a AI... Image generated using 
<a href="https://openai.com/index/dall-e-3/">Dall-E v3</a>. Prompt: <i>A friendly humanoid robot sits at a wooden table in a bright, sunlit room, happily drawing on a sketchbook. Soft light colors, landscape, peaceful, productive, and joyful atmosphere. The robot is drawing an image of itself drawing, creating a recursive effect. Large window in the background with greenery outside, warm natural lighting.</i>
</small>
</p>

In very few years, image generation has become an almost ubiquitous tool that we take for granted. [Variational Autoencoders (VAEs - )](#) were followed by [Generative Adversarial Networks (GANs - )](#), and finally [Denoising Diffusion Probabilistic Models (DDPMs - )](#) conquered the landscape, with remarkable models like [Stable Diffusion XL](#). In the [first post of this series](#) I explain how each of these models work and I provide an example implementation of the DDPM model, which is trained to run car image generation.

In this second part, I would like to focus on the application side of the diffusers, in particular, using the invaluable tools provided by [HuggingFace](https://huggingface.co/).



