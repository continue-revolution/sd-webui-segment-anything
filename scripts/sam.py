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
from modules.ui_components import FormRow
from modules.safe import unsafe_torch_load, load
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessing
from modules.devices import device, torch_gc, cpu
from modules.paths import models_path
from segment_anything import SamPredictor, sam_model_registry
from scripts.dino import dino_model_list, dino_predict_internal, show_boxes, clear_dino_cache, dino_install_issue_text
from scripts.auto import clear_sem_sam_cache, register_auto_sam, semantic_segmentation, sem_sam_garbage_collect, image_layer_internal, categorical_mask_image


refresh_symbol = '\U0001f504'       # 🔄
sam_model_cache = OrderedDict()
scripts_sam_model_dir = os.path.join(scripts.basedir(), "models/sam") 
sd_sam_model_dir = os.path.join(models_path, "sam")
sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir 
sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']


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


def update_mask(mask_gallery, chosen_mask, dilation_amt, input_image):
    print("Dilation Amount: ", dilation_amt)
    mask_image = Image.open(mask_gallery[chosen_mask + 3]['name'])
    binary_img = np.array(mask_image.convert('1'))
    if dilation_amt:
        mask_image, binary_img = dilate_mask(binary_img, dilation_amt)
    blended_image = Image.fromarray(show_masks(np.array(input_image), binary_img.astype(np.bool_)[None, ...]))
    matted_image = np.array(input_image)
    matted_image[~binary_img] = np.array([0, 0, 0, 0])
    return [blended_image, mask_image, Image.fromarray(matted_image)]


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


def clear_cache():
    clear_sam_cache()
    clear_dino_cache()
    clear_sem_sam_cache()


def garbage_collect(sam):
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
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


def dilate_mask(mask, dilation_amt):
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img


def create_mask_output(image_np, masks, boxes_filt, gui):
    print("Creating output image")
    mask_images = []
    masks_gallery = []
    matted_images = []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
        if gui:
            blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
            mask_images.append(Image.fromarray(blended_image))
            image_np_copy = copy.deepcopy(image_np)
            image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
            matted_images.append(Image.fromarray(image_np_copy))
    return mask_images + masks_gallery + matted_images


def create_mask_batch_output(
    input_image_file, dino_batch_dest_dir, 
    image_np, masks, boxes_filt, batch_dilation_amt, 
    dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask):
    print("Creating batch output image")
    filename, ext = os.path.splitext(os.path.basename(input_image_file))
    ext = ".png" # JPEG not compatible with RGBA
    for idx, mask in enumerate(masks):
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        merged_mask = np.any(mask, axis=0)
        if dino_batch_save_background:
            merged_mask = ~merged_mask
        if batch_dilation_amt:
            _, merged_mask = dilate_mask(merged_mask, batch_dilation_amt)
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
        if dino_batch_save_image:
            output_image = Image.fromarray(image_np_copy)
            output_image.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_output{ext}"))
        if dino_batch_save_mask:
            output_mask = Image.fromarray(merged_mask)
            output_mask.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_mask{ext}"))
        if dino_batch_save_image_with_mask:
            output_blend = Image.fromarray(blended_image)
            output_blend.save(os.path.join(dino_batch_dest_dir, f"{filename}_{idx}_blend{ext}"))


def sam_predict(sam_model_name, input_image, positive_points, negative_points,
                dino_checkbox, dino_model_name, text_prompt, box_threshold,
                dino_preview_checkbox, dino_preview_boxes_selection, gui=True):
    print("Start SAM Processing")
    if sam_model_name is None:
        return [], "SAM model not found. Please download SAM model from extension README."
    if input_image is None:
        return [], "SAM requires an input image. Please upload an image first."
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    dino_enabled = dino_checkbox and text_prompt is not None
    boxes_filt = None
    sam_predict_result = " done."
    if dino_enabled:
        boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)
        if install_success and dino_preview_checkbox is not None and dino_preview_checkbox and dino_preview_boxes_selection is not None:
            valid_indices = [int(i) for i in dino_preview_boxes_selection if int(i) < boxes_filt.shape[0]]
            boxes_filt = boxes_filt[valid_indices]
        if not install_success:
            if len(positive_points) == 0 and len(negative_points) == 0:
                return [], f"GroundingDINO installment has failed. Check your terminal for more detail and {dino_install_issue_text}. "
            else:
                sam_predict_result += f" However, GroundingDINO installment has failed. Your process automatically fall back to point prompt only. Check your terminal for more detail and {dino_install_issue_text}. "
    sam = init_sam_model(sam_model_name)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictor(sam)
    predictor.set_image(image_np_rgb)
    if dino_enabled and boxes_filt.shape[0] > 1:
        sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts disgarded"
        print(sam_predict_status)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=True)
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    else:
        num_box = 0 if boxes_filt is None else boxes_filt.shape[0]
        num_points = len(positive_points) + len(negative_points)
        if num_box == 0 and num_points == 0:
            garbage_collect(sam)
            if dino_enabled and dino_preview_checkbox and num_box == 0:
                return [], "It seems that you are using a high box threshold with no point prompts. Please lower your box threshold and re-try."
            return [], "You neither added point prompts nor enabled GroundingDINO. Segmentation cannot be generated."
        sam_predict_status = f"SAM inference with {num_box} box, {len(positive_points)} positive prompts, {len(negative_points)} negative prompts"
        print(sam_predict_status)
        point_coords = np.array(positive_points + negative_points)
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        box = copy.deepcopy(boxes_filt[0].numpy()) if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
        masks, _, _ = predictor.predict(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            box=box,
            multimask_output=True)
        masks = masks[:, None, ...]
    garbage_collect(sam)
    return create_mask_output(image_np, masks, boxes_filt, gui), sam_predict_status + sam_predict_result


def dino_predict(input_image, dino_model_name, text_prompt, box_threshold):
    if input_image is None:
        return None, gr.update(), gr.update(visible=True, value=f"GroundingDINO requires input image.")
    if text_prompt is None or text_prompt == "":
        return None, gr.update(), gr.update(visible=True, value=f"GroundingDINO requires text prompt.")
    image_np = np.array(input_image)
    boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)
    if not install_success:
        return None, gr.update(), gr.update(visible=True, value=f"GroundingDINO installment failed. Preview failed. See your terminal for more detail and {dino_install_issue_text}")
    boxes_filt = boxes_filt.numpy()
    boxes_choice = [str(i) for i in range(boxes_filt.shape[0])]
    return Image.fromarray(show_boxes(image_np, boxes_filt.astype(int), show_index=True)), gr.update(choices=boxes_choice, value=boxes_choice), gr.update(visible=False)


def dino_batch_process(
    batch_sam_model_name, batch_dino_model_name, batch_text_prompt, batch_box_threshold, batch_dilation_amt,
    dino_batch_source_dir, dino_batch_dest_dir,
    dino_batch_output_per_image, dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask):
    if batch_text_prompt is None or batch_text_prompt == "":
        return "Please add text prompts to generate masks"
    print("Start batch processing")
    sam = init_sam_model(batch_sam_model_name)
    predictor = SamPredictor(sam)
    
    process_info = ""
    all_files = glob.glob(os.path.join(dino_batch_source_dir, "*"))
    for image_index, input_image_file in enumerate(all_files):
        print(f"Processing {image_index}/{len(all_files)} {input_image_file}")
        try:
            input_image = Image.open(input_image_file).convert("RGBA")
        except:
            print(f"File {input_image_file} not image, skipped.")
            continue
        image_np = np.array(input_image)
        image_np_rgb = image_np[..., :3]

        boxes_filt, install_success = dino_predict_internal(input_image, batch_dino_model_name, batch_text_prompt, batch_box_threshold)
        if not install_success:
            return f"GroundingDINO installment failed. Batch processing failed. See your terminal for more detail and {dino_install_issue_text}"

        if boxes_filt is None or boxes_filt.shape[0] == 0:
            msg = f"GroundingDINO generated 0 box for image {input_image_file}, please lower the box threshold if you want any segmentation for this image. "
            print(msg)
            process_info += (msg + "\n")
            continue
        
        predictor.set_image(image_np_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=(dino_batch_output_per_image == 1))
        
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
        boxes_filt = boxes_filt.cpu().numpy().astype(int)
        
        create_mask_batch_output(
            input_image_file, dino_batch_dest_dir, 
            image_np, masks, boxes_filt, batch_dilation_amt, 
            dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask)
    
    garbage_collect(sam)
    return process_info + "Done"


def cnet_seg(
    sam_model_name, cnet_seg_input_image, cnet_seg_processor, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print(f"Start semantic segmentation with processor {cnet_seg_processor}")
    auto_sam_output_mode = "coco_rle" if "seg" in cnet_seg_processor else "binary_mask"
    sam = load_sam_model(sam_model_name)
    register_auto_sam(sam, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, auto_sam_output_mode)
    outputs = semantic_segmentation(cnet_seg_input_image, cnet_seg_processor)
    sem_sam_garbage_collect()
    garbage_collect(sam)
    return outputs


def image_layout(
    sam_model_name, layout_input_image_or_path, layout_output_path, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print("Start processing image layout")
    sam = load_sam_model(sam_model_name)
    register_auto_sam(sam, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, "binary_mask")
    outputs = image_layer_internal(layout_input_image_or_path, layout_output_path)
    sem_sam_garbage_collect()
    garbage_collect(sam)
    return outputs


def categorical_mask(
    sam_model_name, crop_processor, crop_category_input, crop_input_image, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print("Start processing categorical mask")
    sam = load_sam_model(sam_model_name)
    register_auto_sam(sam, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, "coco_rle")
    outputs = categorical_mask_image(crop_processor, crop_category_input, crop_input_image)
    sem_sam_garbage_collect()
    garbage_collect(sam)
    if isinstance(outputs, str):
        return [], outputs
    output_gallery = create_mask_output(np.array(crop_input_image), outputs[None, ...], None, False)
    return output_gallery, "Done"


def categorical_mask_batch(
    sam_model_name, crop_processor, crop_category_input, crop_batch_dilation_amt, crop_batch_source_dir, crop_batch_dest_dir, 
    crop_batch_save_image, crop_batch_save_mask, crop_batch_save_image_with_mask, crop_batch_save_background, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area):
    print("Start processing categorical mask in batch")
    sam = load_sam_model(sam_model_name)
    register_auto_sam(sam, auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, "coco_rle")
    all_files = glob.glob(os.path.join(crop_batch_source_dir, "*"))
    process_info = ""
    for image_index, input_image_file in enumerate(all_files):
        print(f"Processing {image_index}/{len(all_files)} {input_image_file}")
        try:
            crop_input_image = Image.open(input_image_file).convert("RGBB")
            outputs = categorical_mask_image(crop_processor, crop_category_input, crop_input_image)
            if isinstance(outputs, str):
                outputs = f"Image {image_index}: {outputs}"
                print(outputs)
                process_info += outputs + "\n"
                continue
            create_mask_batch_output(
                input_image_file, crop_batch_dest_dir, 
                np.array(crop_input_image), outputs[None, ...], None, crop_batch_dilation_amt, 
                crop_batch_save_image, crop_batch_save_mask, crop_batch_save_background, crop_batch_save_image_with_mask)
        except:
            print(f"File {input_image_file} not image, skipped.")
    sem_sam_garbage_collect()
    garbage_collect(sam)
    return outputs


def priorize_sam_scripts(is_img2img):
    cnet_idx = None
    sam_idx = None
    if is_img2img:
        for idx, s in enumerate(scripts.scripts_img2img.alwayson_scripts):
            if s.title() == "Segment Anything":
                sam_idx = idx
            elif s.title() == "ControlNet":
                cnet_idx = idx
        if cnet_idx is not None and sam_idx is not None and cnet_idx < sam_idx:
            scripts.scripts_img2img.alwayson_scripts[cnet_idx], scripts.scripts_img2img.alwayson_scripts[
                sam_idx] = scripts.scripts_img2img.alwayson_scripts[sam_idx], scripts.scripts_img2img.alwayson_scripts[cnet_idx]
    else:
        for idx, s in enumerate(scripts.scripts_txt2img.alwayson_scripts):
            if s.title() == "Segment Anything":
                sam_idx = idx
            elif s.title() == "ControlNet":
                cnet_idx = idx
        if cnet_idx is not None and sam_idx is not None and cnet_idx < sam_idx:
            scripts.scripts_txt2img.alwayson_scripts[cnet_idx], scripts.scripts_txt2img.alwayson_scripts[
                sam_idx] = scripts.scripts_txt2img.alwayson_scripts[sam_idx], scripts.scripts_txt2img.alwayson_scripts[cnet_idx]


def ui_sketch_inner():
    sam_inpaint_color_sketch = gr.Image(label="Color sketch inpainting", source="upload", interactive=True, type="pil", tool="color-sketch", image_mode="RGBA")
    sam_inpaint_color_sketch_orig = gr.State(None)
    with FormRow():
        sam_inpaint_mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4)
        sam_inpaint_mask_alpha = gr.Slider(label="Mask transparency")
    def update_orig(image, state):
        if image is not None:
            same_size = state is not None and state.size == image.size
            has_exact_match = np.any(np.all(np.array(image) == np.array(state), axis=-1))
            edited = same_size and has_exact_match
            return image if not edited or state is None else state
    sam_inpaint_color_sketch.change(update_orig, [sam_inpaint_color_sketch, sam_inpaint_color_sketch_orig], sam_inpaint_color_sketch_orig)
    return sam_inpaint_color_sketch, sam_inpaint_color_sketch_orig, sam_inpaint_mask_blur, sam_inpaint_mask_alpha    


def ui_sketch(sam_input_image):
    sam_sketch_checkbox = gr.Checkbox(value=False, label="Enable Sketch")
    with gr.Column(visible=False) as sketch_column:
        sam_inpaint_copy_button = gr.Button(value="Copy from input image")
        sam_inpaint_color_sketch, sam_inpaint_color_sketch_orig, sam_inpaint_mask_blur, sam_inpaint_mask_alpha = ui_sketch_inner()
    sam_inpaint_copy_button.click(
            fn=lambda x: x,
            inputs=[sam_input_image],
            outputs=[sam_inpaint_color_sketch])
    sam_sketch_checkbox.change(
        fn=gr_show,
        inputs=[sam_sketch_checkbox],
        outputs=[sketch_column],
        show_progress=False)
    return sam_sketch_checkbox, sam_inpaint_color_sketch, sam_inpaint_color_sketch_orig, sam_inpaint_mask_blur, sam_inpaint_mask_alpha

def ui_dilation(sam_output_mask_gallery, sam_output_chosen_mask, sam_input_image):
    sam_dilation_checkbox = gr.Checkbox(value=False, label="Expand Mask")
    with gr.Column(visible=False) as dilation_column:
        sam_dilation_amt = gr.Slider(minimum=0, maximum=100, default=0, value=0, label="Specify the amount that you wish to expand the mask by (recommend 30)")
        sam_dilation_output_gallery = gr.Gallery(label="Expanded Mask").style(grid=3)
        sam_dilation_submit = gr.Button(value="Update Mask")
        sam_dilation_submit.click(
            fn=update_mask,
            inputs=[sam_output_mask_gallery, sam_output_chosen_mask, sam_dilation_amt, sam_input_image],
            outputs=[sam_dilation_output_gallery])
    sam_dilation_checkbox.change(
        fn=gr_show,
        inputs=[sam_dilation_checkbox],
        outputs=[dilation_column],
        show_progress=False)
    return sam_dilation_checkbox, sam_dilation_output_gallery


def ui_inpaint(is_img2img, max_cn):
    with FormRow():
        img2img_inpaint_upload_enable_copy_label = "Copy to Inpaint Upload" if is_img2img else "Please go to img2img to copy to inpaint upload."
        img2img_inpaint_upload_enable_copy = gr.Checkbox(value=False, label=img2img_inpaint_upload_enable_copy_label, interactive=is_img2img)
        cnet_inpaint_copy_enable = gr.Checkbox(value=False, label='Copy to ControlNet Inpaint', visible=(max_cn > 0))
        cnet_inpaint_invert = gr.Checkbox(value=False, label='ControlNet inpaint not masked', visible=(max_cn > 0))
        cnet_inpaint_idx = gr.Radio(value="0" if max_cn > 0 else None, choices=[str(i) for i in range(max_cn)], label='ControlNet Inpaint Index', type="index", visible=(max_cn > 0))
    return img2img_inpaint_upload_enable_copy, cnet_inpaint_copy_enable, cnet_inpaint_invert, cnet_inpaint_idx


def ui_batch(is_dino):
    dino_batch_dilation_amt = gr.Slider(minimum=0, maximum=100, default=0, value=0, label="Specify the amount that you wish to expand the mask by (recommend 0-10)")
    dino_batch_source_dir = gr.Textbox(label="Source directory")
    dino_batch_dest_dir = gr.Textbox(label="Destination directory")
    with gr.Row():
        dino_batch_output_per_image = gr.Radio(choices=["1", "3"], value="3", type="index", label="Output per image: ", visible=is_dino)
        dino_batch_save_image = gr.Checkbox(value=True, label="Save masked image")
        dino_batch_save_mask = gr.Checkbox(value=True, label="Save mask")
        dino_batch_save_image_with_mask = gr.Checkbox(value=True, label="Save original image with mask and bounding box")
        dino_batch_save_background = gr.Checkbox(value=False, label="Save background instead of foreground")
    dino_batch_run_button = gr.Button(value="Start batch process")
    dino_batch_progress = gr.Text(value="", label="GroundingDINO batch progress status")
    return dino_batch_dilation_amt, dino_batch_source_dir, dino_batch_dest_dir, dino_batch_output_per_image, dino_batch_save_image, dino_batch_save_mask, dino_batch_save_image_with_mask, dino_batch_save_background, dino_batch_run_button, dino_batch_progress


class Script(scripts.Script):

    def title(self):
        return 'Segment Anything'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        if self.max_cn_num() > 0:
            priorize_sam_scripts(is_img2img)
        tab_prefix = ("img2img" if is_img2img else "txt2img") + "_sam_"
        ui_process = ()
        with gr.Accordion('Segment Anything', open=False):
            with gr.Row():
                sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list, value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                sam_refresh_models = ToolButton(value=refresh_symbol)
                sam_refresh_models.click(refresh_sam_models, sam_model_name, sam_model_name)
            with gr.Tabs():
                with gr.TabItem(label="Single Image"):
                    gr.HTML(value="<p>Left click the image to add one positive point (black dot). Right click the image to add one negative point (red dot). Left click the point to remove it.</p>")
                    sam_input_image = gr.Image(label="Image for Segment Anything", elem_id=f"{tab_prefix}input_image", source="upload", type="pil", image_mode="RGBA")
                    sam_remove_dots = gr.Button(value="Remove all point prompts")
                    sam_dummy_component = gr.Label(visible=False)
                    sam_remove_dots.click(
                        fn=lambda _: None,
                        _js="samRemoveDots",
                        inputs=[sam_dummy_component],
                        outputs=None)
                    gr.HTML(value="<p>GroundingDINO + Segment Anything can achieve [text prompt]->[object detection]->[segmentation]</p>")
                    dino_checkbox = gr.Checkbox(value=False, label="Enable GroundingDINO", elem_id=f"{tab_prefix}dino_enable_checkbox")
                    with gr.Column(visible=False) as dino_column:
                        gr.HTML(value="<p>Due to the limitation of Segment Anything, when there are point prompts, at most 1 box prompt will be allowed; when there are multiple box prompts, no point prompts are allowed.</p>")
                        dino_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)", choices=dino_model_list, value=dino_model_list[0])
                        dino_text_prompt = gr.Textbox(placeholder="You must enter text prompts to enable groundingdino. Otherwise this extension will fall back to point prompts only.", label="GroundingDINO Detection Prompt", elem_id=f"{tab_prefix}dino_text_prompt")
                        dino_box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)
                        dino_preview_checkbox = gr.Checkbox(value=False, label="I want to preview GroundingDINO detection result and select the boxes I want.", elem_id=f"{tab_prefix}dino_preview_checkbox")
                        with gr.Column(visible=False) as dino_preview:
                            dino_preview_boxes = gr.Image(label="Image for GroundingDINO", type="pil", image_mode="RGBA")
                            dino_preview_boxes_button = gr.Button(value="Generate bounding box", elem_id=f"{tab_prefix}dino_run_button")
                            dino_preview_boxes_selection = gr.CheckboxGroup(label="Select your favorite boxes: ", elem_id=f"{tab_prefix}dino_preview_boxes_selection")
                            dino_preview_result = gr.Text(value="", label="GroundingDINO preview status", visible=False)
                            dino_preview_boxes_button.click(
                                fn=dino_predict,
                                _js="submit_dino",
                                inputs=[sam_input_image, dino_model_name, dino_text_prompt, dino_box_threshold],
                                outputs=[dino_preview_boxes, dino_preview_boxes_selection, dino_preview_result])
                        dino_preview_checkbox.change(
                            fn=gr_show,
                            inputs=[dino_preview_checkbox],
                            outputs=[dino_preview],
                            show_progress=False)
                    dino_checkbox.change(
                        fn=gr_show,
                        inputs=[dino_checkbox],
                        outputs=[dino_column],
                        show_progress=False)
                    sam_output_mask_gallery = gr.Gallery(label='Segment Anything Output').style(grid=3)
                    sam_submit = gr.Button(value="Preview Segmentation", elem_id=f"{tab_prefix}run_button")
                    sam_result = gr.Text(value="", label="Segment Anything status")
                    sam_submit.click(
                        fn=sam_predict,
                        _js='submit_sam',
                        inputs=[sam_model_name, sam_input_image,        # SAM
                                sam_dummy_component, sam_dummy_component,   # Point prompts
                                dino_checkbox, dino_model_name, dino_text_prompt, dino_box_threshold,  # DINO prompts
                                dino_preview_checkbox, dino_preview_boxes_selection],  # DINO preview prompts
                        outputs=[sam_output_mask_gallery, sam_result])
                    with FormRow():
                        sam_output_chosen_mask = gr.Radio(label="Choose your favorite mask: ", value="0", choices=["0", "1", "2"], type="index")
                        gr.Checkbox(value=False, label="Preview automatically when add/remove points", elem_id=f"{tab_prefix}realtime_preview_checkbox")
                    img2img_inpaint_upload_enable_copy, cnet_inpaint_copy_enable, cnet_inpaint_invert, cnet_inpaint_idx = ui_inpaint(is_img2img, self.max_cn_num())
                    sam_dilation_checkbox, sam_dilation_output_gallery = ui_dilation(sam_output_mask_gallery, sam_output_chosen_mask, sam_input_image)
                    sam_single_image_process = (
                        img2img_inpaint_upload_enable_copy, sam_input_image, sam_output_mask_gallery, sam_output_chosen_mask, 
                        sam_dilation_checkbox, sam_dilation_output_gallery, 
                        cnet_inpaint_copy_enable, cnet_inpaint_invert, cnet_inpaint_idx)
                    if is_img2img:
                        sam_single_image_process += ui_sketch(sam_input_image)
                    ui_process += sam_single_image_process

                with gr.TabItem(label="Batch Process"):
                    gr.Markdown(value="You may configurate the following items and generate masked image for all images under a directory. This mode is designed for generating LoRA/LyCORIS training set.")
                    gr.Markdown(value="The current workflow is [text prompt]->[object detection]->[segmentation]. Semantic segmentation support is in Auto SAM panel.")
                    dino_batch_model_name = gr.Dropdown(label="GroundingDINO Model (Auto download from huggingface)", choices=dino_model_list, value=dino_model_list[0])
                    dino_batch_text_prompt = gr.Textbox(label="GroundingDINO Detection Prompt")
                    dino_batch_box_threshold = gr.Slider(label="GroundingDINO Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001)
                    dino_batch_dilation_amt, dino_batch_source_dir, dino_batch_dest_dir, dino_batch_output_per_image, dino_batch_save_image, dino_batch_save_mask, dino_batch_save_image_with_mask, dino_batch_save_background, dino_batch_run_button, dino_batch_progress = ui_batch(True)
                    dino_batch_run_button.click(
                        fn=dino_batch_process,
                        inputs=[sam_model_name, dino_batch_model_name, dino_batch_text_prompt, dino_batch_box_threshold, dino_batch_dilation_amt,
                                dino_batch_source_dir, dino_batch_dest_dir, dino_batch_output_per_image, 
                                dino_batch_save_image, dino_batch_save_mask, dino_batch_save_background, dino_batch_save_image_with_mask],
                        outputs=[dino_batch_progress])
                    
                with gr.TabItem(label="Auto SAM"):
                    gr.Markdown("Auto SAM is mainly for semantic segmentation and image layout generation, which is supported based on ControlNet. You must have ControlNet extension installed, and you should not change its directory name (sd-webui-controlnet).")
                    gr.Markdown("The annotator directory inside the SAM extension directory is only a symbolic link. This is to save your space and make the extension repository clean.")

                    with gr.Accordion(label="Auto SAM Config", open=False):
                        gr.Markdown("You may configurate automatic sam generation. See [here](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L35-L96) for explanation of each parameter. If you still cannot understand, use default.")
                        with gr.Row():
                            auto_sam_points_per_side = gr.Number(label="points_per_side", value=32, precision=0)
                            auto_sam_points_per_batch = gr.Number(label="points_per_batch", value=64, precision=0)
                            auto_sam_pred_iou_thresh = gr.Slider(minimum=0, maximum=1, value=0.88, step=0.01, label="pred_iou_thresh")
                            auto_sam_stability_score_thresh = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, label="stability_score_thresh")
                            auto_sam_stability_score_offset = gr.Number(label="stability_score_offset", value=1)
                        with gr.Row():
                            auto_sam_box_nms_thresh = gr.Slider(label="box_nms_thresh", value=0.7, minimum=0, maximum=1, step=0.01)
                            auto_sam_crop_n_layers = gr.Number(label="crop_n_layers", value=0, precision=0)
                            auto_sam_crop_nms_thresh = gr.Slider(label="crop_nms_thresh", value=0.7, minimum=0, maximum=1, step=0.01)
                            auto_sam_crop_overlap_ratio = gr.Slider(label="crop_overlap_ratio", value=512/1500, minimum=0, maximum=1, step=0.01)
                            auto_sam_crop_n_points_downscale_factor = gr.Number(label="crop_n_points_downscale_factor", value=1, precision=0)
                        auto_sam_min_mask_region_area = gr.Number(label="min_mask_region_area", value=0, precision=0)
                        auto_sam_config = (auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
                                           auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
                                           auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
                                           auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area)

                    with gr.Tabs():
                        with gr.TabItem(label="ControlNet"):
                            gr.Markdown("You can enhance semantic segmentation for control_v11p_sd15_seg from lllyasviel. Non-semantic segmentation for [Edit-Anything](https://github.com/sail-sg/EditAnything) will be supported [when they convert their models to lllyasviel format](https://github.com/sail-sg/EditAnything/issues/14).")
                            cnet_seg_processor = gr.Radio(choices=["seg_ufade20k", "seg_ofade20k", "seg_ofcoco", "random"], value="seg_ufade20k", label="Choose preprocessor for semantic segmentation: ")
                            cnet_seg_input_image = gr.Image(label="Image for Auto Segmentation", source="upload", type="pil", image_mode="RGBA")
                            cnet_seg_output_gallery = gr.Gallery(label="Auto segmentation output").style(grid=2)
                            cnet_seg_submit = gr.Button(value="Generate segmentation image")
                            cnet_seg_status = gr.Text(value="", label="Segmentation status")
                            cnet_seg_submit.click(
                                fn=cnet_seg,
                                inputs=[sam_model_name, cnet_seg_input_image, cnet_seg_processor, *auto_sam_config],
                                outputs=[cnet_seg_output_gallery, cnet_seg_status])
                            with gr.Row(visible=(self.max_cn_num() > 0)):
                                cnet_seg_enable_copy = gr.Checkbox(value=False, label='Copy to ControlNet Segmentation')
                                cnet_seg_idx = gr.Radio(value="0" if self.max_cn_num() > 0 else None, choices=[str(i) for i in range(self.max_cn_num())], label='ControlNet Segmentation Index', type="index")
                            auto_sam_process = (cnet_seg_output_gallery, cnet_seg_enable_copy, cnet_seg_idx)
                            ui_process += auto_sam_process

                        with gr.TabItem(label="Image Layout"):
                            gr.Markdown("You can generate image layout either in single image or in batch. Since there might be A LOT of outputs, there is no gallery for review. You need to go to the output folder for either single image or batch process.")
                            layout_mode = gr.Radio(choices=["single image", "batch process"], value="single image", type="index", label="Choose mode: ")
                            layout_input_image = gr.Image(label="Image for Image Layout", source="upload", type="pil", image_mode="RGBA")
                            layout_input_path = gr.Textbox(label="Input path", placeholder="Enter input path", visible=False)
                            layout_output_path = gr.Textbox(label="Output path", placeholder="Enter output path")
                            layout_submit_single = gr.Button(value="Generate layout for single image")
                            layout_submit_batch = gr.Button(value="Generate layout for batch process", visible=False)
                            layout_status = gr.Text(value="", label="Image layout status")
                            def layout_show(mode):
                                is_single = mode == 0
                                return gr.update(visible=is_single), gr.update(visible=is_single), gr.update(visible=not is_single), gr.update(visible=not is_single)
                            layout_mode.change(
                                fn=layout_show,
                                inputs=[layout_mode],
                                outputs=[layout_input_image, layout_submit_single, layout_input_path, layout_submit_batch])
                            layout_submit_single.click(
                                fn=image_layout,
                                inputs=[sam_model_name, layout_input_image, layout_output_path, *auto_sam_config],
                                outputs=[layout_status])
                            layout_submit_batch.click(
                                fn=image_layout,
                                inputs=[sam_model_name, layout_input_path, layout_output_path, *auto_sam_config],
                                outputs=[layout_status])

                        with gr.TabItem(label="Mask by Category"):
                            gr.Markdown("You can mask images by their categories via semantic segmentation. Please enter category ids (integers), separated by `+`. Visit [here](https://github.com/Mikubill/sd-webui-controlnet/blob/main/annotator/oneformer/oneformer/data/datasets/register_ade20k_panoptic.py#L12-L207) for ade20k and [here](https://github.com/Mikubill/sd-webui-controlnet/blob/main/annotator/oneformer/detectron2/data/datasets/builtin_meta.py#L20-L153) for coco to get category->id map.")
                            crop_processor = gr.Radio(choices=["seg_ufade20k", "seg_ofade20k", "seg_ofcoco"], value="seg_ufade20k", label="Choose preprocessor for semantic segmentation: ")
                            crop_category_input = gr.Textbox(placeholder="Enter categody ids, separated by +. For example, if you want bed+person, your input should be 7+12 for ade20k and 65+1 for coco.", label="Enter category IDs")
                            with gr.Tabs():
                                with gr.TabItem(label="Single Image"):
                                    crop_input_image = gr.Image(label="Image to be masked", source="upload", type="pil", image_mode="RGBA")
                                    crop_output_gallery = gr.Gallery(label="Output").style(grid=3)
                                    crop_padding = gr.Number(value=-2, visible=False, interactive=False)
                                    crop_submit = gr.Button(value="Generate mask")
                                    crop_result = gr.Text(value="", label="Categorical mask status")
                                    crop_submit.click(
                                        fn=categorical_mask,
                                        inputs=[sam_model_name, crop_processor, crop_category_input, crop_input_image, *auto_sam_config],
                                        outputs=[crop_output_gallery, crop_result])
                                    crop_img2img_inpaint_enable, crop_cnet_inpaint_enable, crop_cnet_inpaint_invert, crop_cnet_inpaint_idx = ui_inpaint(is_img2img, self.max_cn_num())
                                    crop_dilation_checkbox, crop_dilation_output_gallery = ui_dilation(crop_output_gallery, crop_padding, crop_input_image)
                                    crop_single_image_process = (
                                        crop_img2img_inpaint_enable, crop_input_image, crop_output_gallery, 
                                        crop_dilation_checkbox, crop_dilation_output_gallery, 
                                        crop_cnet_inpaint_enable, crop_cnet_inpaint_invert, crop_cnet_inpaint_idx)
                                    if is_img2img:
                                        crop_single_image_process += ui_sketch(sam_input_image)
                                    ui_process += crop_single_image_process

                                with gr.TabItem(label="Batch Process"):
                                    crop_batch_dilation_amt, crop_batch_source_dir, crop_batch_dest_dir, _, crop_batch_save_image, crop_batch_save_mask, crop_batch_save_image_with_mask, crop_batch_save_background, crop_batch_run_button, crop_batch_progress = ui_batch(False)
                                    crop_batch_run_button.click(
                                        fn=categorical_mask_batch,
                                        inputs=[sam_model_name, crop_processor, crop_category_input, crop_batch_dilation_amt, crop_batch_source_dir, crop_batch_dest_dir, crop_batch_save_image, crop_batch_save_mask, crop_batch_save_image_with_mask, crop_batch_save_background, *auto_sam_config],
                                        outputs=[crop_batch_progress])

                with gr.TabItem(label="Upload Mask to ControlNet Inpainting"):
                    gr.Markdown("This panel is for those who want to upload mask to ControlNet inpainting, or draw mask once for both img2img and ControlNet inpainting. It is not part of SAM's functionality. By checking the box below, you agree that you will disable all functionalities of SAM.")
                    gr.Markdown("This panel will exist until ControlNet Inpainting support uploading human-painted masks. It is not something related to SAM, but it is beneficial to Stable Diffusion community.")
                    with FormRow():
                        cnet_upload_enable = gr.Checkbox(value=False, label="Disable SAM functionality and upload manually created mask to ControlNet inpaint.")
                        cnet_upload_invert = gr.Checkbox(value=False, label="ControlNet inpaint not masked")
                        cnet_upload_num = gr.Radio(value="0" if self.max_cn_num() > 0 else None, choices=[str(i) for i in range(self.max_cn_num())], label='ControlNet Inpaint Number', type="index")
                        cnet_upload_to_img2img_enable = gr.Checkbox(value=False, visible=is_img2img, label="Also upload to img2img inpainting upload.")
                    with gr.Column(visible=False) as cnet_upload_panel:
                        if is_img2img:
                            cnet_upload_choice = gr.Radio(choices=["Upload", "Draw"], value="Upload", type="index")
                        with gr.Row() as cnet_upload_row:
                            cnet_upload_img_inpaint = gr.Image(label="Image for ControlNet Inpaint", source="upload", interactive=True, type="pil")
                            cnet_upload_mask_inpaint = gr.Image(label="Mask for ControlNet Inpaint", source="upload", interactive=True, type="pil")
                        if is_img2img:
                            with gr.Column(visible=False) as cnet_draw_panel:
                                cnet_upload_inpaint_color_sketch, cnet_upload_inpaint_color_sketch_orig, cnet_upload_inpaint_mask_blur, cnet_upload_inpaint_mask_alpha = ui_sketch_inner()
                            cnet_upload_choice.change(
                                fn=lambda x: (gr_show(x == 0), gr_show(x == 1)),
                                inputs=[cnet_upload_choice],
                                outputs=[cnet_upload_row, cnet_draw_panel])
                    cnet_upload_enable.change(
                        fn=gr_show,
                        inputs=[cnet_upload_enable],
                        outputs=[cnet_upload_panel],
                        show_progress=False)
                    cnet_upload_process = (cnet_upload_enable, cnet_upload_invert, cnet_upload_num, cnet_upload_to_img2img_enable, cnet_upload_img_inpaint, cnet_upload_mask_inpaint)
                    if is_img2img:
                        cnet_upload_process += (cnet_upload_choice, cnet_upload_inpaint_color_sketch, cnet_upload_inpaint_color_sketch_orig, cnet_upload_inpaint_mask_blur, cnet_upload_inpaint_mask_alpha)
                    ui_process += cnet_upload_process

                with gr.Row():
                    switch = gr.Button(value="Switch to Inpaint Upload")
                    unload = gr.Button(value="Unload all models from memory")
                    switch.click(
                        fn=lambda _: None,
                        _js="switchToInpaintUpload",
                        inputs=[sam_dummy_component],
                        outputs=None)
                    unload.click(
                        fn=clear_cache,
                        inputs=[],
                        outputs=[])
        
        return ui_process

    def process(self, p: StableDiffusionProcessingImg2Img, *args):
                # enable_copy_inpaint=False, input_image=None, mask=None, chosen_mask=0, dilation_enabled=False, expanded_mask=None, enable_copy_cn_inpaint=False, cn_num=0):
                # cnet_upload_enable=False, cnet_upload_to_img2img_enable=False, cnet_upload_num=0, cnet_upload_img_inpaint=None, cnet_upload_mask_inpaint=None):
        pass # TODO
        # if cnet_upload_enable:
        #     if cnet_upload_to_img2img_enable:
        #         p.init_images = [cnet_upload_img_inpaint]
        #         p.image_mask = cnet_upload_mask_inpaint
        #     self.set_p_value(p, 'control_net_input_image', cnet_upload_num, {"image": cnet_upload_img_inpaint, "mask": cnet_upload_mask_inpaint.convert("L")})
        
        # if input_image is not None and mask is not None:
        #     image_mask = Image.open(expanded_mask[1]['name'] if dilation_enabled and expanded_mask is not None else mask[chosen_mask + 3]['name'])
        #     if enable_copy_inpaint and isinstance(p, StableDiffusionProcessingImg2Img):
        #         p.init_images = [input_image]
        #         p.image_mask = image_mask
        #     if enable_copy_cn_inpaint and cn_num < self.max_cn_num():
        #         self.set_p_value(p, 'control_net_input_image', cn_num, {"image": input_image, "mask": image_mask.convert("L")})
        
    def set_p_value(self, p: StableDiffusionProcessing, attr: str, idx: int, v):
        value = getattr(p, attr, None)
        if isinstance(value, list):
            value[idx] = v
        else:
            # if value is None, ControlNet uses default value
            value = [value] * self.max_cn_num()
            value[idx] = v
        setattr(p, attr, value)

    def max_cn_num(self):
        if shared.opts.data is None:
            return 0
        return int(shared.opts.data.get('control_net_max_models_num', 0))
