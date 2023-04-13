import gc
import os
import copy
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
from scripts.dino import dino_model_list, dino_predict

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


def show_box(image, box, color=np.array([255, 0, 0, 255]), thickness=2):
    x, y, w, h = box
    image[y:y+thickness, x:x+w] = color
    image[y+h:y+h+thickness, x:x+w] = color
    image[y:y+h, x:x+thickness] = color
    image[y:y+h, x+w:x+w+thickness] = color
    return image
        

def show_mask(image_np, mask, random_color=False, alpha=0.5, box=None):
    image = copy.deepcopy(image_np)
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    image[mask] = image[mask] * (1 - alpha) + 255 * \
        color.reshape(1, 1, -1) * alpha
    if box is not None:
        image = show_box(image, box)
    return image.astype(np.uint8)


def load_sam_model(sam_checkpoint):
    model_type = '_'.join(sam_checkpoint.split('_')[1:-1])
    sam_checkpoint = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
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
                dino_checkbox, dino_model_name, text_prompt, box_threshold):
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    dino_enabled = dino_checkbox and text_prompt is not None

    boxes_filt = dino_predict(input_image, dino_model_name, text_prompt, box_threshold) if dino_enabled else None

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

    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictor(sam)
    predictor.set_image(image_np_rgb)
    point_coords = np.array(positive_points + negative_points)
    point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))

    box = boxes_filt[0].numpy() if boxes_filt is not None else None
    masks, _, _ = predictor.predict(
        point_coords=point_coords if len(point_coords) > 0 else None,
        point_labels=point_labels if len(point_coords) > 0 else None,
        box=box,
        multimask_output=True,
    )
    box = predictor.transform.apply_boxes(box, image_np.shape[:2]).flatten().astype(int) if box is not None else None

    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()

    print("Creating output image")
    masks_gallery = []
    mask_images = []
    for mask in masks:
        blended_image = show_mask(image_np, mask, box=box)
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
        with gr.Accordion('Segment Anything', open=False, elem_id=id('accordion'), visible=is_img2img):
            with gr.Column():
                gr.HTML(value="<p>Left click the image to add one positive point (black dot). Right click the image to add one negative point (red dot). Left click the point to remove it.</p>")
                with gr.Row():
                    sam_model_name = gr.Dropdown(label="Model", elem_id="sam_model", choices=sam_model_list,
                                            value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                    refresh_models = ToolButton(value=refresh_symbol)
                    refresh_models.click(
                        refresh_sam_models, sam_model_name, sam_model_name)
                input_image = gr.Image(label="Image for Segment Anything", elem_id="sam_input_image",
                                    show_label=False, source="upload", type="pil", image_mode="RGBA")
                dummy_component = gr.Label(visible=False)

                gr.HTML(value="<p>GroundingDINO + Segment Anything can achieve [text prompt]->[object detection]->[segmentation]</p>")
                dino_checkbox = gr.Checkbox(value=False, label="Enable GroundingDINO", elem_id="dino_enable_checkbox")
                with gr.Column(visible=False) as dino_column:
                    gr.HTML(value="<p>Due to the limitation of Segment Anything, when there are point prompts, at most 1 box prompt will be allowed; when there are multiple box prompts, no point prompts are allowed.</p>")
                    gr.HTML(value="<p>In the current version: only one box will be used for segmentation. A preview of future features has been shown in the tab below.</p>")
                    dino_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)", 
                                                  elem_id="dino_model", choices=dino_model_list, value=dino_model_list[0])

                    text_prompt = gr.Textbox(label="GroundingDINO Detection Prompt", elem_id="dino_text_prompt")
                    text_prompt.change(fn=lambda x: None, inputs=[dummy_component], outputs=None, _js="registerDinoTextObserver")

                    box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)

                    with gr.Tabs():
                        with gr.TabItem(label="Automatic", id=0) as dino_auto_tab:
                            gr.HTML(value="<p>[TODO] In this mode: When no point prompt, all boxes generated by GroundingDINO applied to generate masks; when point prompt, only the first box generated by GroundingDINO applied to generate masks.</p>")
                            # TODO: Implement the case when no point prompt.
                        with gr.TabItem(label="Preview", id=1) as dino_preview_tab:
                            gr.HTML(value="<p>[TODO] In this mode: You may preview boxes and select all boxes you want. You may annotate on the image, but when you select multiple boxes, your annotation will not affect mask generation. You may annotate the image and choose a specific box to preview segmentation and select the mask you want the most.</p>")
                            # TODO: Implement appropriate UI components with disable-able button and backend function.
                        with gr.TabItem(label="Batch", id=2) as dino_batch_tab:
                            gr.HTML(value="<p>[TODO] In this mode: You may configurate the following items and generate masked image for all images under a directory. This mode is designed for generating LoRA/LyCORIS training set.</p>")
                            dino_batch_source_dir = gr.Textbox(label="Source directory")
                            dino_batch_dest_dir = gr.Textbox(label="Destination directory")
                            with gr.Row():
                                dino_batch_output_per_image = gr.Radio(choices=["1", "3"], value="1", type="index", label="Output per image: ")
                                dino_batch_save_mask = gr.Checkbox(value=True, label="Save mask")
                                dino_batch_save_image_with_mask = gr.Checkbox(value=True, label="Save original image with mask and detection box")
                            dino_batch_run_button = gr.Button(value="Start batch process")
                            dino_batch_progress = gr.Text(value="", show_label=False)
                            # TODO: Implement batch process function and disable this button when necessary.
                            dino_batch_run_button.click(
                                fn=None,
                                inputs=[dino_model_name, text_prompt, box_threshold, 
                                        dino_batch_source_dir, dino_batch_dest_dir, 
                                        dino_batch_output_per_image, dino_batch_save_mask, 
                                        dino_batch_save_image_with_mask],
                                outputs=[dino_batch_progress]
                            )

                with gr.Column() as output_elements:
                    mask_image = gr.Gallery(label='Segment Anything Output', show_label=False, elem_id='sam_gallery').style(grid=3)

                    with gr.Row(elem_id="sam_generate_box", elem_classes="generate-box"):
                        gr.Button(value="You cannot preview segmentation because you have not added dot prompt or enabled GroundingDINO with text prompts.", elem_id="sam_no_button")
                        run_button = gr.Button(value="Preview Segmentation", elem_id="sam_run_button")
                    with gr.Row():
                        enabled = gr.Checkbox(value=False, label="Copy to Inpaint Upload", elem_id="sam_impaint_checkbox")
                        switch = gr.Button(value="Switch to Inpaint Upload")
                        chosen_mask = gr.Radio(label="Choose your favorite mask: ", value="0", choices=["0", "1", "2"], type="index")

            run_button.click(
                fn=sam_predict,
                _js='submit_sam',
                inputs=[sam_model_name, input_image,        # SAM
                        dummy_component, dummy_component,   # Point prompts
                        dino_checkbox, dino_model_name, text_prompt, box_threshold], # DINO prompts
                outputs=[mask_image],
                show_progress=False)
            
            dino_checkbox.change(
                fn=gr_show,
                inputs=[dino_checkbox],
                outputs=[dino_column],
                show_progress=False)
            
            switch.click(
                fn=lambda x: None,
                _js="switchToInpaintUpload",
                inputs=[dummy_component],
                outputs=None
            )
            
            def select_dino_tab(tab):
                return gr.update(visible=tab in [0, 1])

            for i, elem in enumerate([dino_auto_tab, dino_preview_tab, dino_batch_tab]):
                elem.select(
                    fn=lambda tab=i: select_dino_tab(tab),
                    inputs=[],
                    outputs=[output_elements],
                )
        return [enabled, input_image, mask_image, chosen_mask]

    def process(self, p: StableDiffusionProcessingImg2Img, enabled=False, input_image=None, mask=None, chosen_mask=0):
        if not enabled or input_image is None or mask is None or not isinstance(p, StableDiffusionProcessingImg2Img):
            return
        p.init_images = [input_image]
        p.image_mask = Image.open(mask[chosen_mask + 3]['name'])
