from typing import Callable, Optional, Sequence, Tuple, List, Any
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw


# Constants
IMG_SIZE: int = 512

# Type aliases
Point = Tuple[int, int]
Points = List[Point]
AnnotatedMasks = List[Tuple[np.ndarray, str]]
SamOutput = Tuple[Image.Image, AnnotatedMasks]
GetProcessedInputsFn = Callable[[Image.Image, Sequence[Sequence[Point]]], np.ndarray]
InpaintFn = Callable[[Image.Image, np.ndarray, str, Optional[str], int, float], Image.Image]


def _preprocess_to_square(input_img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Pad to square (white) and resize."""
    w, h = input_img.size
    if w != h:
        new_size = max(w, h)
        new_image = Image.new("RGB", (new_size, new_size), "white")
        left = (new_size - w) // 2
        top = (new_size - h) // 2
        new_image.paste(input_img, (left, top))
        input_img = new_image
    return input_img.resize((size, size))


def _draw_crosshair(img: Image.Image, points: Sequence[Point], size: int = 10) -> Image.Image:
    """Draw crosshairs on a copy of the image (non-destructive)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    for x, y in points:
        draw.line((x - size, y, x + size, y), fill="green", width=5)
        draw.line((x, y - size, x, y + size), fill="green", width=5)
    return out


def generate_app(get_processed_inputs: GetProcessedInputsFn, inpaint: InpaintFn) -> gr.Blocks:
    """
    Build the Gradio UI.

    Args:
        get_processed_inputs: Callable that runs SAM and returns a boolean/binary mask (H,W).
            Expected signature: (image_pil, [[(x,y), ...]]) -> np.ndarray mask
        inpaint: Callable that runs inpainting and returns a PIL image.
            Expected signature: (image_pil, mask_np, prompt, negative_prompt, seed, cfg) -> PIL.Image

    Returns:
        A gr.Blocks app (already launched).
    """

    with gr.Blocks() as demo:
        # Session state (per user)
        points_state = gr.State([])     # type: ignore[var-annotated]
        image_state = gr.State(None)    # type: ignore[var-annotated]

        gr.Markdown(
            """
            # Image inpainting
            1. Upload an image by clicking on the first canvas.
            2. Click on the subject you would like to keep. SAM will run immediately.
               Add more points to refine the mask.
            3. Write a prompt (and optionally a negative prompt). Adjust CFG and seed.
               Toggle the checkbox to infill subject instead of background.
            4. Click "Run inpaint". Adjust prompt/settings and run again if needed.

            # EXAMPLES
            Scroll down to see a few examples. Click one to fill image + prompts.
            You still need step 2 and step 4.
            """
        )

        with gr.Row():
            display_img = gr.Image(
                label="Input",
                interactive=True,
                type="pil",
                height=IMG_SIZE,
                width=IMG_SIZE,
            )

            sam_mask = gr.AnnotatedImage(
                label="SAM result",
                height=IMG_SIZE,
                width=IMG_SIZE,
                color_map={"background": "#a89a00"},
            )

            result = gr.Image(
                label="Output",
                type="pil",
                height=IMG_SIZE,
                width=IMG_SIZE,
            )

        with gr.Row():
            cfg = gr.Slider(
                label="Classifier-Free Guidance Scale",
                minimum=0.0,
                maximum=20.0,
                value=7.0,
                step=0.05,
            )
            random_seed = gr.Number(label="Random seed", value=74294536, precision=0)
            invert_checkbox = gr.Checkbox(label="Infill subject instead \nof background")

        with gr.Row():
            prompt = gr.Textbox(label="Prompt for infill")
            neg_prompt = gr.Textbox(label="Negative prompt")

            reset_btn = gr.ClearButton(
                value="Reset",
                components=[display_img, sam_mask, result, prompt, neg_prompt, invert_checkbox],
            )

            run_btn = gr.Button(value="Run inpaint")

        with gr.Row():
            gr.Examples(
                [
                    ["car.png", "a car driving on planet Mars. Studio lights, 1970s", "artifacts, low quality, distortion", 74294536],
                    ["dragon.jpeg", "a dragon in a medieval village", "artifacts, low quality, distortion", 97],
                    ["monalisa.png", "a fantasy landscape with flying dragons", "artifacts, low quality, distortion", 97],
                ],
                inputs=[display_img, prompt, neg_prompt, random_seed],
            )

        # Callbacks

        def on_image_change(img: Optional[Image.Image]) -> Tuple[Optional[Image.Image], Points, Optional[Image.Image]]:
            """Preprocess image; reset points and stored original image when user uploads/changes."""
            if img is None:
                return None, [], None
            processed = _preprocess_to_square(img, IMG_SIZE)
            return processed, [], processed.copy()

        def run_sam(image_pil: Optional[Image.Image], points: Points) -> SamOutput:
            if image_pil is None or not points:
                raise gr.Error("Upload an image and click at least one point to segment with SAM.")

            # Your get_processed_inputs expects [[points]]
            mask = get_processed_inputs(image_pil, [points])

            # Ensure mask is HxW and uint8/bool; then resize to IMG_SIZE with nearest neighbor
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8))
            mask_img = mask_img.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
            res_mask = (np.array(mask_img) > 127)

            base = image_pil.resize((IMG_SIZE, IMG_SIZE))
            return base, [(res_mask, "background"), (~res_mask, "subject")]

        def on_select(
            img: Image.Image,
            points: Points,
            original_img: Optional[Image.Image],
            evt: gr.SelectData,
        ) -> Tuple[SamOutput, Image.Image, Points, Image.Image]:
            """Handle click: store original once, append point, run SAM, overlay crosshair."""
            if original_img is None:
                original_img = img.copy()

            x, y = int(evt.index[0]), int(evt.index[1])
            points = [*points, (x, y)]

            sam_out = run_sam(original_img, points)
            overlayed = _draw_crosshair(img, points)

            return sam_out, overlayed, points, original_img

        def on_run_inpaint(
            prompt_text: str,
            negative_text: str,
            cfg_val: float,
            seed_val: float,
            invert: bool,
            original_img: Optional[Image.Image],
            points: Points,
        ) -> Image.Image:
            if original_img is None or not points:
                raise gr.Error("Upload an image and click at least one point before inpainting.")

            base_img, masks = run_sam(original_img, points)
            # masks[0] is (background_mask, "background")
            amask = masks[0][0]

            what = "subject" if invert else "background"
            if invert:
                amask = ~amask

            gr.Info(f"Inpainting {what}...")

            # Gradio Number gives float; ensure int for seed
            seed_int = int(seed_val)

            out = inpaint(
                original_img,
                amask.astype(np.uint8) * 255,  # many pipelines like 0/255 masks
                prompt_text,
                negative_text if negative_text else None,
                seed_int,
                float(cfg_val),
            )
            return out.resize((IMG_SIZE, IMG_SIZE))

        # Wire events
        display_img.change(
            fn=on_image_change,
            inputs=[display_img],
            outputs=[display_img, points_state, image_state],
        )

        display_img.select(
            fn=on_select,
            inputs=[display_img, points_state, image_state],
            outputs=[sam_mask, display_img, points_state, image_state],
        )

        # When the image is cleared/reset, clear state too
        reset_btn.click(
            fn=lambda: ([], None),
            inputs=[],
            outputs=[points_state, image_state],
        )

        run_btn.click(
            fn=on_run_inpaint,
            inputs=[prompt, neg_prompt, cfg, random_seed, invert_checkbox, image_state, points_state],
            outputs=[result],
        )

    demo.queue(max_size=1).launch(share=True, debug=True)
    return demo
