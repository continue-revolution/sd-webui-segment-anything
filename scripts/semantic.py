import os
import gc
from collections import OrderedDict
import torch
from segment_anything import SamAutomaticMaskGenerator
from modules import scripts, shared
from modules.paths import extensions_dir
from modules.devices import torch_gc


global_sam = None
sem_seg_cache = OrderedDict()
original_uniformer_inference_segmentor = None
original_oneformer_draw_sem_seg = None


def strengthen_sem_seg(class_ids, img):
    import pycocotools.mask as maskUtils
    semantc_mask = class_ids.clone()
    annotations = global_sam(img)
    annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
    for ann in annotations:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        semantc_mask[valid_mask] = top_1_propose_class_ids
    return semantc_mask


def inject_inference_segmentor(model, img):
    original_result = original_uniformer_inference_segmentor(model, img)
    original_result[0] = strengthen_sem_seg(original_result[0], img)
    return original_result


def inject_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8, is_text=True, edge_color=(1.0, 1.0, 1.0)):
    if isinstance(sem_seg, torch.Tensor):
        sem_seg = sem_seg.numpy()
    return original_oneformer_draw_sem_seg(self, strengthen_sem_seg(sem_seg), area_threshold, alpha, is_text, edge_color)


def _uniformer(img, res=512):
    from annotator.util import resize_image, HWC3
    img = resize_image(HWC3(img), res)
    if "uniformer" not in sem_seg_cache:
        from annotator.uniformer import apply_uniformer
        sem_seg_cache["uniformer"] = apply_uniformer
    result = sem_seg_cache["uniformer"](img)
    return result, True


def _oneformer(img, res=512, dataset="coco"):
    from annotator.util import resize_image, HWC3
    img = resize_image(HWC3(img), res)
    model_key = f"oneformer_{dataset}"
    if model_key not in sem_seg_cache:
        from annotator.oneformer import OneformerDetector
        sem_seg_cache[model_key] = OneformerDetector(OneformerDetector.configs[dataset])
    result = sem_seg_cache[model_key](img)
    return result, True


def create_symbolic_link():
    os.symlink(os.path.join(extensions_dir, "sd-webui-controlnet/annotator"), scripts.basedir())
    

def clear_sem_sam_cache():
    sem_seg_cache.clear()
    gc.collect()
    torch_gc()
    

def sem_sam_garbage_collect():
    if shared.cmd_opts.lowvram:
        for _, model in sem_seg_cache:
            model.unload_model()
    gc.collect()
    torch_gc()


def semantic_segmentation(
    annotator_name, input_image, processor_res, sam,
    points_per_side=64,
    points_per_batch=64,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    stability_score_offset=1.0,
    box_nms_thresh=0.7,
    crop_n_layers=1,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500,
    crop_n_points_downscale_factor=1,
    point_grids=None,
    min_mask_region_area=100):
    global global_sam
    global_sam = SamAutomaticMaskGenerator(
        sam, points_per_side, points_per_batch, 
        pred_iou_thresh, stability_score_thresh, 
        stability_score_offset, box_nms_thresh, 
        crop_n_layers, crop_nms_thresh, crop_overlap_ratio, 
        crop_n_points_downscale_factor, point_grids, 
        min_mask_region_area, "coco_rle")
    try:
        import annotator.uniformer as uniformer
    except:
        create_symbolic_link()
        import annotator.uniformer as uniformer
    if annotator_name == "seg_ufade20k":
        uniformer.inference_segmentor = inject_inference_segmentor
        return _uniformer(input_image, processor_res)
    else:
        from annotator.oneformer.oneformer.demo.visualizer import Visualizer
        Visualizer.draw_sem_seg = inject_sem_seg
        return _oneformer(input_image, processor_res, annotator_name.split('_')[-1][2:])
    