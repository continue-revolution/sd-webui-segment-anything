import gc
import os
import numpy as np
from PIL import Image
import torch
import gradio as gr
from collections import OrderedDict
from modules import scripts, shared
from launch import run_pip
from modules.safe import unsafe_torch_load, load
from modules.processing import StableDiffusionProcessingImg2Img
from modules.devices import device, torch_gc, cpu

try:
    from segment_anything import SamPredictor, build_sam
except ImportError:
    run_pip("install segment_anything", "segment_anything")
    from segment_anything import SamPredictor, build_sam


model_cache = OrderedDict()
sam_model_dir = os.path.join(scripts.basedir(), "models", "sam")
model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']


refresh_symbol = '\U0001f504'       # 🔄


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


def show_mask(image, mask, random_color=False, alpha=0.5):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    image[mask] = image[mask] * (1 - alpha) + 255 * \
        color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)


def load_sam_model(sam_checkpoint):
    sam_checkpoint = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = build_sam(checkpoint=sam_checkpoint)
    sam.to(device=device)
    torch.load = load
    return sam


def clear_sam_cache():
    model_cache.clear()
    gc.collect()
    torch_gc()


def refresh_sam_models(*inputs):
    global model_list
    model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
        os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']
    dd = inputs[0]
    if dd in model_list:
        selected = dd
    elif len(model_list) > 0:
        selected = model_list[0]
    else:
        selected = None
    return gr.Dropdown.update(choices=model_list, value=selected)


def sam_predict(model_name, input_image, positive_points, negative_points):
    print("Initializing SAM")
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]

    if model_name in model_cache:
        sam = model_cache[model_name]
        if shared.cmd_opts.lowvram:
            sam.to(device=device)
    elif model_name in model_list:
        clear_sam_cache()
        model_cache[model_name] = load_sam_model(model_name)
        sam = model_cache[model_name]
    else:
        Exception(
            f"{model_name} not found, please download model to models/sam.")

    predictor = SamPredictor(sam)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor.set_image(image_np_rgb)
    point_coords = np.array(positive_points + negative_points)
    point_labels = np.array(
        [1] * len(positive_points) + [0] * len(negative_points))
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()
    print("Creating output image")
    masks_gallery = []
    mask_images = []

    for mask in masks:
        blended_image = show_mask(image_np, mask)
        masks_gallery.append(Image.fromarray(mask))
        mask_images.append(Image.fromarray(blended_image))
    return mask_images + masks_gallery


class Script(scripts.Script):

    def title(self):
        return 'Segment Anything'

    def show(self, is_img2img):
        # TODO: Here I bypassed a bug inside module.img2img line 154, should be scripts_img2img instead.
        # return scripts.AlwaysVisible if is_img2img else False
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # if is_img2img:
        with gr.Accordion('Segment Anything', open=False, elem_id=id('accordion')):
            with gr.Column():
                gr.HTML(value="<p>Left click the image to add one positive point (black dot). Right click the image to add one negative point (red dot). Left click the point to remove it.</p>", label="Positive points")
                with gr.Row():
                    model_name = gr.Dropdown(label="Model", elem_id="sam_model", choices=model_list,
                                             value=model_list[0] if len(model_list) > 0 else None)
                    refresh_models = ToolButton(value=refresh_symbol)
                    refresh_models.click(
                        refresh_sam_models, model_name, model_name)
                input_image = gr.Image(label="Image for Segment Anything", elem_id="sam_input_image",
                                       show_label=False, source="upload", type="pil", image_mode="RGBA")
                dummy_component = gr.Label(visible=False)
                mask_image = gr.Gallery(
                    label='Segment Anything Output', show_label=False, elem_id='sam_gallery').style(grid=3)
                run_button = gr.Button(
                    value="Preview Segmentation", visible=False, elem_id="sam_run_button")
                gr.Button(
                    value="You cannot preview segmentation because you have not added dot prompt.", elem_id="sam_no_button")
                with gr.Row():
                    enabled = gr.Checkbox(
                        value=False, label="Copy to Inpaint Upload", elem_id="sam_impaint_checkbox")
                    chosen_mask = gr.Radio(label="Choose your favorite mask: ", value="0", choices=[
                                           "0", "1", "2"], type="index")

            run_button.click(
                fn=sam_predict,
                _js='submit_sam',
                inputs=[model_name, input_image,
                        dummy_component, dummy_component],
                outputs=[mask_image],
                show_progress=False)
        return [enabled, input_image, mask_image, chosen_mask]

    def process(self, p: StableDiffusionProcessingImg2Img, enabled=False, input_image=None, mask=None, chosen_mask=0):
        if not enabled or input_image is None or mask is None or not isinstance(p, StableDiffusionProcessingImg2Img):
            return
        p.init_images = [input_image]
        p.image_mask = Image.open(mask[chosen_mask + 3]['name'])
