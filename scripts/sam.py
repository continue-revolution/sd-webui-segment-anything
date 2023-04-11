import gc
import os
import numpy as np
from PIL import Image
import torch
import gradio as gr
from collections import OrderedDict
from modules import scripts, shared
from modules.ui import gr_show
from modules.safe import unsafe_torch_load, load
from modules.processing import StableDiffusionProcessingImg2Img
from modules.devices import device, torch_gc, cpu
from segment_anything import SamPredictor, sam_model_registry


sam_model_cache = OrderedDict()
sam_model_dir = os.path.join(scripts.basedir(), "models/sam")
sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
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
    model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
    sam_checkpoint = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    torch.load = load
    return sam


def clear_sam_cache():
    sam_model_cache.clear()
    gc.collect()
    torch_gc()


def refresh_sam_models(*inputs):
    global sam_model_list
    sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
        os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']
    dd = inputs[0]
    if dd in sam_model_list:
        selected = dd
    elif len(sam_model_list) > 0:
        selected = sam_model_list[0]
    else:
        selected = None
    return gr.Dropdown.update(choices=sam_model_list, value=selected)


def sam_predict(sam_model_name, input_image, positive_points, negative_points,
                dino_checkbox, dino_model_name, text_prompt, text_threshold, box_threshold):
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    
    if dino_checkbox:
        print("Initializing GroundingDINO ")

    print("Initializing SAM")
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        if shared.cmd_opts.lowvram:
            sam.to(device=device)
    elif sam_model_name in sam_model_list:
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam_model(sam_model_name)
        sam = sam_model_cache[sam_model_name]
    else:
        Exception(
            f"{sam_model_name} not found, please download model to models/sam.")

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
        with gr.Accordion('Segment Anything', open=False, elem_id=id('accordion'), visible=is_img2img):
            with gr.Column():
                gr.HTML(value="<p>Left click the image to add one positive point (black dot). Right click the image to add one negative point (red dot). Left click the point to remove it.</p>", label="Positive points")
                with gr.Row():
                    sam_model_name = gr.Dropdown(label="Model", elem_id="sam_model", choices=sam_model_list,
                                            value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                    refresh_models = ToolButton(value=refresh_symbol)
                    refresh_models.click(
                        refresh_sam_models, sam_model_name, sam_model_name)
                input_image = gr.Image(label="Image for Segment Anything", elem_id="sam_input_image",
                                    show_label=False, source="upload", type="pil", image_mode="RGBA")
                dummy_component = gr.Label(visible=False)
                
                dino_checkbox = gr.Checkbox(value=False, label="Enable GroundingDINO for text prompt->object detection->segmentation")
                with gr.Column(visible=False) as dino_column:
                    dino_models = ["GroundingDINO_SwinT_OGC (694MB)", "GroundingDINO_SwinB (938MB)"]
                    dino_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)", 
                                                    elem_id="dino_model", choices=dino_models, value=dino_models[0])
                    text_prompt = gr.Textbox(label="GroundingDINO Detection Prompt")
                    with gr.Row():
                        box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)
                        text_threshold = gr.Slider(label="GroundingDINO Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001)

                mask_image = gr.Gallery(
                    label='Segment Anything Output', show_label=False, elem_id='sam_gallery').style(grid=3)

                with gr.Row(elem_id="sam_generate_box", elem_classes="generate-box"):
                    gr.Button(value="You cannot preview segmentation because you have not added dot prompt.", elem_id="sam_no_button")
                    run_button = gr.Button(value="Preview Segmentation", elem_id="sam_run_button")
                with gr.Row():
                    enabled = gr.Checkbox(
                        value=False, label="Copy to Inpaint Upload", elem_id="sam_impaint_checkbox")
                    chosen_mask = gr.Radio(label="Choose your favorite mask: ", value="0", choices=[
                                        "0", "1", "2"], type="index")

            run_button.click(
                fn=sam_predict,
                _js='submit_sam',
                inputs=[sam_model_name, input_image,        # SAM
                        dummy_component, dummy_component,   # Point prompts
                        dino_checkbox, dino_model_name, text_prompt, text_threshold, box_threshold],  # DINO Prompts
                outputs=[mask_image],
                show_progress=False)
            
            dino_checkbox.change(
                fn=gr_show,
                inputs=[dino_checkbox],
                outputs=[dino_column],
                show_progress=False)
        return [enabled, input_image, mask_image, chosen_mask]

    def process(self, p: StableDiffusionProcessingImg2Img, enabled=False, input_image=None, mask=None, chosen_mask=0):
        if not enabled or input_image is None or mask is None or not isinstance(p, StableDiffusionProcessingImg2Img):
            return
        p.init_images = [input_image]
        p.image_mask = Image.open(mask[chosen_mask + 3]['name'])
