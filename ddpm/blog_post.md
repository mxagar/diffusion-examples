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

Concepts to explain:

- Discriminative models learn decision boundaries
- Generative models learn distributions where they sample from
- Realistic images are really points in a large vast of noise
- Images are represented in a latent space as high-dimensional vectors
- Conditional (text2image, image2image) and unconditional
- Multimodal (text and image)
- Main image generation meythods: VAEs, GANs, Diffusion
- The impossible triangle: coverage, quality, speed

