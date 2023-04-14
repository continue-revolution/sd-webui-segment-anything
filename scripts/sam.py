import gc
import os
import copy
import glob
import numpy as np
from PIL import Image
import torch
import gradio as gr
from collections import OrderedDict
from scipy.ndimage import binary_dilation
from modules import scripts, shared
from modules.ui import gr_show
from modules.safe import unsafe_torch_load, load
from modules.processing import StableDiffusionProcessingImg2Img
from modules.devices import device, torch_gc, cpu
from modules.paths import models_path
from segment_anything import SamPredictor, sam_model_registry
from scripts.dino import dino_model_list, dino_predict_internal, show_boxes, clear_dino_cache

sam_model_cache = OrderedDict()
scripts_sam_model_dir = os.path.join(scripts.basedir(), "models/sam") 
sd_sam_model_dir = os.path.join(models_path, "sam")
sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir 
sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']


refresh_symbol = '\U0001f504'       # 🔄


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"
        

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
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

def update_mask(mask_gallery, chosen_mask, dilation_amt, input_image):
    print("Dilation Amount: ", dilation_amt)
    mask_image = Image.open(mask_gallery[chosen_mask + 3]['name'])
    binary_img = np.array(mask_image.convert('1'))
    if dilation_amt:
        # Convert the image to a binary numpy array
        mask_image, binary_img = dilate_mask(binary_img, dilation_amt)
    
    blended_image = Image.fromarray(show_masks(np.array(input_image), binary_img.astype(np.bool_)[None, ...]))
    matted_image = np.array(input_image)
    matted_image[~binary_img] = np.array([0, 0, 0, 0])
    return [blended_image, mask_image, Image.fromarray(matted_image)]

def clear_cache():
    clear_sam_cache()
    clear_dino_cache()


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

def dilate_mask(mask, dilation_amt):
    # Create a dilation kernel
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)

    # Dilate the image
    dilated_binary_img = binary_dilation(mask, dilation_kernel)

    # Convert the dilated binary numpy array back to a PIL image
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)

    return dilated_mask, dilated_binary_img


def init_sam_model(sam_model_name):
    print("Initializing SAM")
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        if shared.cmd_opts.lowvram:
            sam.to(device=device)
        return sam
    elif sam_model_name in sam_model_list:
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam_model(sam_model_name)
        return sam_model_cache[sam_model_name]
    else:
        Exception(
            f"{sam_model_name} not found, please download model to models/sam.")


def sam_predict(sam_model_name, input_image, positive_points, negative_points,
                dino_checkbox, dino_model_name, text_prompt, box_threshold,
                dino_preview_checkbox, dino_preview_boxes_selection):
    print("Start SAM Processing")
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    dino_enabled = dino_checkbox and text_prompt is not None

    boxes_filt = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold) if dino_enabled else None
    if dino_enabled and dino_preview_checkbox is not None and dino_preview_checkbox and dino_preview_boxes_selection is not None:
        valid_indices = [int(i) for i in dino_preview_boxes_selection if int(i) < boxes_filt.shape[0]]
        boxes_filt = boxes_filt[valid_indices]

    sam = init_sam_model(sam_model_name)

    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictor(sam)
    predictor.set_image(image_np_rgb)

    if dino_enabled and boxes_filt.shape[0] > 1:
        print(f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts disgarded.")
        boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes.to(device),
            multimask_output=True,
        )
        
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        boxes = boxes.cpu().numpy().astype(int)    
    else:
        print(f"SAM inference with {0 if boxes_filt is None else boxes_filt.shape[0]} box, {len(positive_points)} positive prompts, {len(negative_points)} negative prompts.")
        point_coords = np.array(positive_points + negative_points)
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))

        box = boxes_filt[0].numpy() if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
        masks, _, _ = predictor.predict(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            box=box,
            multimask_output=True,
        )

        masks = masks[:, None, ...]
        boxes = predictor.transform.apply_boxes(box, image_np.shape[:2]).astype(int) if box is not None else None

    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()

    print("Creating output image")
    mask_images = []
    masks_gallery = []
    matted_images = []

    for mask in masks:
        blended_image = show_masks(show_boxes(image_np, boxes), mask)
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
        mask_images.append(Image.fromarray(blended_image))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        matted_images.append(Image.fromarray(image_np_copy))

    return mask_images + masks_gallery + matted_images


def dino_predict(sam_model_name, input_image, dino_model_name, text_prompt, box_threshold):
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    boxes_filt = dino_predict_internal(
        input_image, dino_model_name, text_prompt, box_threshold).numpy()

    sam = init_sam_model(sam_model_name)

    predictor = SamPredictor(sam)
    predictor.set_image(image_np_rgb)
    boxes_filt = predictor.transform.apply_boxes(
        boxes_filt, image_np.shape[:2]).astype(int)
    
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()
    
    boxes_choice = [str(i) for i in range(boxes_filt.shape[0])]
    return Image.fromarray(show_boxes(image_np, boxes_filt, show_index=True)), gr.update(choices=boxes_choice, value=boxes_choice)

def dino_batch_process(
    batch_sam_model_name, batch_dino_model_name, batch_text_prompt, batch_box_threshold, batch_dilation_amt,
    dino_batch_source_dir, dino_batch_dest_dir,
    dino_batch_output_per_image, dino_batch_save_mask, dino_batch_save_image_with_mask):
    if batch_text_prompt is None or batch_text_prompt == "":
        return "Please add text prompts to generate masks"
    print("Start batch processing")
    sam = init_sam_model(batch_sam_model_name)
    predictor = SamPredictor(sam)
    
    all_files = glob.glob(os.path.join(dino_batch_source_dir, "*"))
    for image_index, input_image_file in enumerate(all_files):
        print(f"processing {image_index}/{len(all_files)} {input_image_file}")
        input_image = Image.open(input_image_file).convert("RGBA")
        image_np = np.array(input_image)
        image_np_rgb = image_np[..., :3]

        boxes_filt = dino_predict_internal(input_image, batch_dino_model_name, batch_text_prompt, batch_box_threshold)

        predictor.set_image(image_np_rgb)
        boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes.to(device),
            multimask_output=(dino_batch_output_per_image == 1),
        )
        
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        boxes = boxes.cpu().numpy().astype(int)
        
        filename, ext = os.path.splitext(os.path.basename(input_image_file))

        for idx, mask in enumerate(masks):
            blended_image = show_masks(show_boxes(image_np, boxes), mask)
            _, merged_mask = dilate_mask(np.any(mask, axis=0), batch_dilation_amt)
            image_np_copy = copy.deepcopy(image_np)
            image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
            output_image = Image.fromarray(image_np_copy)
            output_image.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_output{ext}"))
            if dino_batch_save_mask:
                output_mask = Image.fromarray(merged_mask)
                output_mask.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_mask{ext}"))
            if dino_batch_save_image_with_mask:
                output_blend = Image.fromarray(blended_image)
                output_blend.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_blend{ext}"))
    
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()
    
    return "Done"
    

class Script(scripts.Script):

    def title(self):
        return 'Segment Anything'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Segment Anything', open=False, elem_id=id('accordion'), visible=is_img2img):
            with gr.Tabs():
                with gr.TabItem(label="Single Image"):
                    with gr.Column():
                        gr.HTML(value="<p>Left click the image to add one positive point (black dot). Right click the image to add one negative point (red dot). Left click the point to remove it.</p>")
                        with gr.Row():
                            sam_model_name = gr.Dropdown(label="SAM Model", elem_id="sam_model", choices=sam_model_list,
                                                    value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                            refresh_models = ToolButton(value=refresh_symbol)
                            refresh_models.click(refresh_sam_models, sam_model_name, sam_model_name)
                        input_image = gr.Image(label="Image for Segment Anything", elem_id="sam_input_image",
                                            show_label=False, source="upload", type="pil", image_mode="RGBA")
                        remove_dots = gr.Button(value="Remove all point prompts")
                        dummy_component = gr.Label(visible=False)

                        gr.HTML(value="<p>GroundingDINO + Segment Anything can achieve [text prompt]->[object detection]->[segmentation]</p>")
                        dino_checkbox = gr.Checkbox(value=False, label="Enable GroundingDINO", elem_id="dino_enable_checkbox")
                        with gr.Column(visible=False) as dino_column:
                            gr.HTML(value="<p>Due to the limitation of Segment Anything, when there are point prompts, at most 1 box prompt will be allowed; when there are multiple box prompts, no point prompts are allowed.</p>")
                            dino_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)", 
                                                        elem_id="dino_model", choices=dino_model_list, value=dino_model_list[0])

                            text_prompt = gr.Textbox(label="GroundingDINO Detection Prompt", elem_id="dino_text_prompt")
                            text_prompt.change(fn=lambda _: None, inputs=[dummy_component], outputs=None, _js="registerDinoTextObserver")

                            box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)

                            dino_preview_checkbox = gr.Checkbox(value=False, label="I want to preview GroundingDINO detection result and select the boxes I want.", elem_id="dino_preview_checkbox")
                            with gr.Column(visible=False) as dino_preview:
                                dino_preview_boxes = gr.Image(label="Image for GroundingDINO", elem_id="dino_box_output",
                                                            show_label=False, type="pil", image_mode="RGBA")
                                with gr.Row(elem_id="dino_generate_box", elem_classes="generate-box"):
                                    gr.Button(value="Add text prompt to generate bounding box", elem_id="dino_no_button")
                                    dino_preview_boxes_button = gr.Button(value="Generate bounding box", elem_id="dino_run_button")
                                dino_preview_boxes_selection = gr.CheckboxGroup(label="Select your favorite boxes: ", elem_id="dino_preview_boxes_selection")
                                dino_preview_boxes_selection.change(fn=lambda _: None, inputs=[dino_preview_boxes_selection], outputs=None, _js="onChangeDinoPreviewBoxesSelection")

                                dino_preview_boxes_button.click(
                                    fn=dino_predict,
                                    _js="submit_dino",
                                    inputs=[sam_model_name, input_image, dino_model_name, text_prompt, box_threshold],
                                    outputs=[dino_preview_boxes, dino_preview_boxes_selection]
                                )

                        mask_image = gr.Gallery(label='Segment Anything Output', show_label=False, elem_id='sam_gallery').style(grid=3)

                        with gr.Row(elem_id="sam_generate_box", elem_classes="generate-box"):
                            gr.Button(value="Add dot prompt or enable GroundingDINO with text prompts to preview segmentation", elem_id="sam_no_button")
                            run_button = gr.Button(value="Preview Segmentation", elem_id="sam_run_button")
                            
                        gr.Checkbox(value=False, label="Preview automatically when add/remove points", elem_id="sam_realtime_preview_checkbox")
                            
                        with gr.Row():
                            enabled = gr.Checkbox(value=False, label="Copy to Inpaint Upload", elem_id="sam_impaint_checkbox")
                            chosen_mask = gr.Radio(label="Choose your favorite mask: ", value="0", choices=["0", "1", "2"], type="index")
                            
                        dilation_checkbox = gr.Checkbox(value=False, label="Expand Mask", elem_id="dilation_enable_checkbox")
                        with gr.Column(visible=False) as dilation_column:
                            dilation_amt = gr.Slider(minimum=0, maximum=100, default=0, value=0, label="Specify the amount that you wish to expand the mask by (recommend 30)", elem_id="dilation_amt")
                            expanded_mask_image = gr.Gallery(label="Expanded Mask", elem_id="sam_expanded_mask").style(grid=3)
                            update_mask_button = gr.Button(value="Update Mask", elem_id="update_mask_button")
                        
                        switch = gr.Button(value="Switch to Inpaint Upload")
                        
                with gr.TabItem(label="Batch Process"):
                    gr.HTML(value="<p>You may configurate the following items and generate masked image for all images under a directory. This mode is designed for generating LoRA/LyCORIS training set.</p>")
                    gr.HTML(value="<p>The current workflow is [text prompt]->[object detection]->[segmentation]. Semantic segmentation support is coming soon!</p>")
                    
                    with gr.Row():
                        batch_sam_model_name = gr.Dropdown(label="SAM Model", elem_id="batch_sam_model", choices=sam_model_list,
                                                           value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                        batch_refresh_models = ToolButton(value=refresh_symbol)
                        batch_refresh_models.click(refresh_sam_models, sam_model_name, sam_model_name)
                    
                    batch_dino_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)",
                                                        elem_id="batch_dino_model", choices=dino_model_list, value=dino_model_list[0])
                    
                    batch_text_prompt = gr.Textbox(label="GroundingDINO Detection Prompt", elem_id="batch_dino_text_prompt")

                    batch_box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)
                    
                    batch_dilation_amt = gr.Slider(minimum=0, maximum=100, default=0, value=0, label="Specify the amount that you wish to expand the mask by (recommend 0-10)", elem_id="batch_dino_dilation_amt")
                    
                    dino_batch_source_dir = gr.Textbox(label="Source directory")
                    dino_batch_dest_dir = gr.Textbox(label="Destination directory")
                    with gr.Row():
                        dino_batch_output_per_image = gr.Radio(
                            choices=["1", "3"], value="3", type="index", label="Output per image: ")
                        dino_batch_save_mask = gr.Checkbox(value=True, label="Save mask")
                        dino_batch_save_image_with_mask = gr.Checkbox(
                            value=True, label="Save original image with mask and bounding box")
                    dino_batch_run_button = gr.Button(value="Start batch process")
                    dino_batch_progress = gr.Text(value="", show_label=False)
                    dino_batch_run_button.click(
                        fn=dino_batch_process,
                        inputs=[batch_sam_model_name, batch_dino_model_name, batch_text_prompt, batch_box_threshold, batch_dilation_amt,
                                dino_batch_source_dir, dino_batch_dest_dir,
                                dino_batch_output_per_image, dino_batch_save_mask,
                                dino_batch_save_image_with_mask],
                        outputs=[dino_batch_progress]
                    )

                unload = gr.Button(value="Unload all models from memory")

            run_button.click(
                fn=sam_predict,
                _js='submit_sam',
                inputs=[sam_model_name, input_image,        # SAM
                        dummy_component, dummy_component,   # Point prompts
                        dino_checkbox, dino_model_name, text_prompt, box_threshold, # DINO prompts
                        dino_preview_checkbox, dino_preview_boxes_selection],  # DINO preview prompts
                outputs=[mask_image],)
            
            dino_checkbox.change(
                fn=gr_show,
                inputs=[dino_checkbox],
                outputs=[dino_column],
                show_progress=False)

            dino_preview_checkbox.change(
                fn=gr_show,
                inputs=[dino_preview_checkbox],
                outputs=[dino_preview],
                show_progress=False)
            
            switch.click(
                fn=lambda _: None,
                _js="switchToInpaintUpload",
                inputs=[dummy_component],
                outputs=None
            )

            remove_dots.click(
                fn=lambda _: None,
                _js="removeDots",
                inputs=[dummy_component],
                outputs=None
            )
            
            unload.click(
                fn=clear_cache,
                inputs=[],
                outputs=[]
            )

            dilation_checkbox.change(
                fn=gr_show,
                inputs=[dilation_checkbox],
                outputs=[dilation_column],
                show_progress=False)

            update_mask_button.click(
                fn=update_mask,
                inputs=[mask_image, chosen_mask, dilation_amt, input_image],
                outputs=[expanded_mask_image])
        
        return [enabled, input_image, mask_image, chosen_mask, dilation_checkbox, expanded_mask_image]

    def process(self, p: StableDiffusionProcessingImg2Img, enabled=False, input_image=None, mask=None, chosen_mask=0, dilation_enabled=False, expanded_mask=None):
        if not enabled or input_image is None or mask is None or not isinstance(p, StableDiffusionProcessingImg2Img):
            return
        p.init_images = [input_image]
        p.image_mask = Image.open(expanded_mask[1]['name'] if dilation_enabled and expanded_mask is not None else mask[chosen_mask + 3]['name'])
