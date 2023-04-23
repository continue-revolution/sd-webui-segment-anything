import os
import gc
import glob
import copy
from PIL import Image
from collections import OrderedDict
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator
from modules import scripts, shared
from modules.paths import extensions_dir
from modules.devices import torch_gc


global_sam: SamAutomaticMaskGenerator = None
sem_seg_cache = OrderedDict()
sam_annotator_dir = os.path.join(scripts.basedir(), "annotator")
original_uniformer_inference_segmentor = None
original_oneformer_draw_sem_seg = None


def blend_image_and_seg(image, seg, alpha=0.5):
    image_blend = image * (1 - alpha) + np.array(seg) * alpha
    return Image.fromarray(image_blend.astype(np.uint8))


def create_symbolic_link():
    cnet_annotator_dir = os.path.join(extensions_dir, "sd-webui-controlnet/annotator")
    if os.path.isdir(cnet_annotator_dir):
        if not os.path.isdir(sam_annotator_dir):
            os.symlink(cnet_annotator_dir, sam_annotator_dir, target_is_directory=True)
        return True
    return False


def clear_sem_sam_cache():
    sem_seg_cache.clear()
    gc.collect()
    torch_gc()
    

def sem_sam_garbage_collect():
    if shared.cmd_opts.lowvram:
        for model_key, model in sem_seg_cache:
            if model_key == "uniformer":
                from annotator.uniformer import unload_uniformer_model
                unload_uniformer_model()
            else:
                model.unload_model()
    gc.collect()
    torch_gc()


def strengthen_sem_seg(class_ids, img):
    print("Auto SAM strengthening semantic segmentation")
    import pycocotools.mask as maskUtils
    semantc_mask = copy.deepcopy(class_ids)
    annotations = global_sam.generate(img)
    annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
    print(f"Auto SAM generated {len(annotations)} masks")
    for ann in annotations:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        propose_classes_ids = torch.tensor(class_ids[valid_mask])
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0].numpy()
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        semantc_mask[valid_mask] = top_1_propose_class_ids.numpy()
    print("Auto SAM strengthen process end")
    return semantc_mask


def random_segmentation(img):
    print("Auto SAM generating random segmentation for Edit-Anything")
    img_np = np.array(img.convert("RGB"))
    annotations = global_sam.generate(img_np)
    annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
    print(f"Auto SAM generated {len(annotations)} masks")
    H, W, C = img_np.shape
    cnet_input = np.zeros((H, W), dtype=np.uint16)
    for idx, annotation in enumerate(annotations):
        current_seg = annotation['segmentation']
        cnet_input[current_seg] = idx + 1 # TODO: Add random mask, not the ugly detected map
    detected_map = np.zeros((cnet_input.shape[0], cnet_input.shape[1], 3))
    detected_map[:, :, 0] = cnet_input % 256
    detected_map[:, :, 1] = cnet_input // 256
    from annotator.util import HWC3
    detected_map = HWC3(detected_map.astype(np.uint8))
    print("Auto SAM generation process end")
    return [blend_image_and_seg(img_np, detected_map), Image.fromarray(detected_map)], "Random segmentation done. Left is blended image, right is ControlNet input."


def image_layer_image(layout_input_image, layout_output_path):
    img_np = np.array(layout_input_image.convert("RGB"))
    annotations = global_sam.generate(img_np)
    print(f"AutoSAM generated {len(annotations)} annotations")
    annotations = sorted(annotations, key=lambda x: x['area'])
    for idx, annotation in enumerate(annotations):
        img_tmp = np.zeros((img_np.shape[0], img_np.shape[1], 3))
        img_tmp[annotation['segmentation']] = img_np[annotation['segmentation']]
        img_np[annotation['segmentation']] = np.array([0, 0, 0])
        img_tmp = Image.fromarray(img_tmp.astype(np.uint8))
        img_tmp.save(os.path.join(layout_output_path, f"{idx}.png"))
    img_np = Image.fromarray(img_np.astype(np.uint8))
    img_np.save(os.path.join(layout_output_path, "leftover.png"))


def image_layer_internal(layout_input_image_or_path, layout_output_path):
    if isinstance(layout_input_image_or_path, str):
        print("Image layer division batch processing")
        all_files = glob.glob(os.path.join(layout_input_image_or_path, "*"))
        for image_index, input_image_file in enumerate(all_files):
            print(f"Processing {image_index}/{len(all_files)} {input_image_file}")
            try:
                input_image = Image.open(input_image_file)
                output_directory = os.path.join(layout_output_path, os.path.splitext(os.path.basename(input_image_file))[0])
                image_layer_image(input_image, output_directory)
            except:
                print(f"File {input_image_file} not image, skipped.")
    else:
        image_layer_image(layout_input_image_or_path, layout_output_path)
    return "Done"


def inject_inference_segmentor(model, img):
    original_result = original_uniformer_inference_segmentor(model, img)
    original_result[0] = strengthen_sem_seg(original_result[0], img)
    return original_result


def inject_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8, is_text=True, edge_color=(1.0, 1.0, 1.0)):
    if isinstance(sem_seg, torch.Tensor):
        sem_seg = sem_seg.numpy() # TODO: inject another function for oneformer
    return original_oneformer_draw_sem_seg(self, strengthen_sem_seg(sem_seg), area_threshold, alpha, is_text, edge_color)


def inject_oodss(self, sem_seg, area_threshold=None, alpha=0.8, is_text=True, edge_color=(1.0, 1.0, 1.0)):
    return sem_seg


def inject_show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10), opacity=0.5, title='', block=True):
    return result[0]


def _uniformer(img):
    if "uniformer" not in sem_seg_cache:
        from annotator.uniformer import apply_uniformer
        sem_seg_cache["uniformer"] = apply_uniformer
    result = sem_seg_cache["uniformer"](img)
    return result


def _oneformer(img, dataset="coco"):
    oneformer_key = f"oneformer_{dataset}"
    if oneformer_key not in sem_seg_cache:
        from annotator.oneformer import OneformerDetector
        sem_seg_cache[oneformer_key] = OneformerDetector(OneformerDetector.configs[dataset])
    result = sem_seg_cache[oneformer_key](img)
    return result


def semantic_segmentation(input_image, annotator_name, processor_res):
    if input_image is None:
        return [], "No input image."
    if "seg" in annotator_name:
        if not os.path.isdir(os.path.join(scripts.basedir(), "annotator")) and not create_symbolic_link():
            return [], "ControlNet extension not found."
        global original_uniformer_inference_segmentor
        global original_oneformer_draw_sem_seg
        from annotator.util import resize_image, HWC3
        input_image = resize_image(HWC3(np.array(input_image)), processor_res)
        print("Generating semantic segmentation without SAM")
        if annotator_name == "seg_ufade20k":
            original_semseg = _uniformer(input_image)
            print("Generating semantic segmentation with SAM")
            import annotator.uniformer as uniformer
            original_uniformer_inference_segmentor = uniformer.inference_segmentor
            uniformer.inference_segmentor = inject_inference_segmentor
            sam_semseg = _uniformer(input_image)
            uniformer.inference_segmentor = original_uniformer_inference_segmentor
            output_gallery = [original_semseg, sam_semseg, blend_image_and_seg(input_image, original_semseg), blend_image_and_seg(input_image, sam_semseg)]
            return output_gallery, "Uniformer semantic segmentation of ade20k done. Left is segmentation before SAM, right is segmentation after SAM."
        else:
            dataset = annotator_name.split('_')[-1][2:]
            original_semseg = _oneformer(input_image, dataset=dataset)
            print("Generating semantic segmentation with SAM")
            from annotator.oneformer.oneformer.demo.visualizer import Visualizer
            original_oneformer_draw_sem_seg = Visualizer.draw_sem_seg
            Visualizer.draw_sem_seg = inject_sem_seg
            sam_semseg = _oneformer(input_image, dataset=dataset)
            Visualizer.draw_sem_seg = original_oneformer_draw_sem_seg
            output_gallery = [original_semseg, sam_semseg, blend_image_and_seg(input_image, original_semseg), blend_image_and_seg(input_image, sam_semseg)]
            return output_gallery, f"Oneformer semantic segmentation of {dataset} done. Left is segmentation before SAM, right is segmentation after SAM."
    else:
        return random_segmentation(input_image)


def categorical_mask_image(crop_processor, crop_processor_res, crop_category_input, crop_input_image):
    if crop_input_image is None:
        return "No input image."
    if not os.path.isdir(os.path.join(scripts.basedir(), "annotator")) and not create_symbolic_link():
        return "ControlNet extension not found."
    filter_classes = crop_category_input.split('+')
    if len(filter_classes) == 0:
        return "No class selected."
    try:
        filter_classes = [int(i) for i in filter_classes]
    except:
        return "Illegal class id. You may have input some string."
    from annotator.util import resize_image, HWC3
    crop_input_image = resize_image(HWC3(np.array(crop_input_image)), crop_processor_res)
    crop_input_image_copy = copy.deepcopy(crop_input_image)
    global original_uniformer_inference_segmentor
    global original_oneformer_draw_sem_seg
    print(f"Generating categories with processor {crop_processor}")
    if crop_processor == "seg_ufade20k":
        import annotator.uniformer as uniformer
        original_uniformer_inference_segmentor = uniformer.inference_segmentor
        uniformer.inference_segmentor = inject_inference_segmentor
        tmp_ouis = uniformer.show_result_pyplot
        uniformer.show_result_pyplot = inject_show_result_pyplot
        sam_semseg = _uniformer(crop_input_image)
        uniformer.inference_segmentor = original_uniformer_inference_segmentor
        uniformer.show_result_pyplot = tmp_ouis
    else:
        dataset = crop_processor.split('_')[-1][2:]
        from annotator.oneformer.oneformer.demo.visualizer import Visualizer
        tmp_oodss = Visualizer.draw_sem_seg
        Visualizer.draw_sem_seg = inject_sem_seg
        original_oneformer_draw_sem_seg = inject_oodss
        sam_semseg = _oneformer(crop_input_image, dataset=dataset)
        Visualizer.draw_sem_seg = tmp_oodss
    mask = np.zeros(sam_semseg.shape, dtype=np.bool_)
    for i in filter_classes:
        mask[sam_semseg == i] = True
    return mask, crop_input_image_copy


def register_auto_sam(sam, 
    auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
    auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
    auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
    auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, auto_sam_output_mode):
    global global_sam
    global_sam = SamAutomaticMaskGenerator(
        sam, auto_sam_points_per_side, auto_sam_points_per_batch, 
        auto_sam_pred_iou_thresh, auto_sam_stability_score_thresh, 
        auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
        auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
        auto_sam_crop_n_points_downscale_factor, None, 
        auto_sam_min_mask_region_area, auto_sam_output_mode)
