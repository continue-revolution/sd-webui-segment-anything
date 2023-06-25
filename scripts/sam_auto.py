from typing import List, Tuple
import os
import glob
import copy
from PIL import Image
import numpy as np
import torch
from sam_hq.automatic import SamAutomaticMaskGeneratorHQ
from scripts.sam_log import logger


class AutoSAM:

    def __init__(self) -> None:
        self.auto_sam: SamAutomaticMaskGeneratorHQ = None


    def blend_image_and_seg(self, image: np.ndarray, seg: np.ndarray, alpha=0.5) -> Image.Image:
        image_blend = image * (1 - alpha) + np.array(seg) * alpha
        return Image.fromarray(image_blend.astype(np.uint8))


    def strengthen_semmantic_seg(self, class_ids: np.ndarray, img: np.ndarray) -> np.ndarray:
        logger.info("AutoSAM strengthening semantic segmentation")
        import pycocotools.mask as maskUtils
        semantc_mask = copy.deepcopy(class_ids)
        annotations = self.auto_sam.generate(img)
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        logger.info(f"AutoSAM generated {len(annotations)} masks")
        for ann in annotations:
            valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
            propose_classes_ids = torch.tensor(class_ids[valid_mask])
            num_class_proposals = len(torch.unique(propose_classes_ids))
            if num_class_proposals == 1:
                semantc_mask[valid_mask] = propose_classes_ids[0].numpy()
                continue
            top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
            semantc_mask[valid_mask] = top_1_propose_class_ids.numpy()
        logger.info("AutoSAM strengthening process end")
        return semantc_mask


    def random_segmentation(self, img: Image.Image) -> Tuple[List[Image.Image], str]:
        logger.info("AutoSAM generating random segmentation for EditAnything")
        img_np = np.array(img.convert("RGB"))
        annotations = self.auto_sam.generate(img_np)
        logger.info(f"AutoSAM generated {len(annotations)} masks")
        H, W, _ = img_np.shape
        color_map = np.zeros((H, W, 3), dtype=np.uint8)
        detected_map_tmp = np.zeros((H, W), dtype=np.uint16)
        for idx, annotation in enumerate(annotations):
            current_seg = annotation['segmentation']
            color_map[current_seg] = np.random.randint(0, 255, (3))
            detected_map_tmp[current_seg] = idx + 1
        detected_map = np.zeros((detected_map_tmp.shape[0], detected_map_tmp.shape[1], 3))
        detected_map[:, :, 0] = detected_map_tmp % 256
        detected_map[:, :, 1] = detected_map_tmp // 256
        try:
            from scripts.processor import HWC3
        except:
            return [], "ControlNet extension not found."
        detected_map = HWC3(detected_map.astype(np.uint8))
        logger.info("AutoSAM generation process end")
        return [self.blend_image_and_seg(img_np, color_map), Image.fromarray(color_map), Image.fromarray(detected_map)], \
            "Random segmentation done. Left above (0) is blended image, right above (1) is random segmentation, left below (2) is Edit-Anything control input."


    def layer_single_image(self, layout_input_image: Image.Image, layout_output_path: str) -> None:
        img_np = np.array(layout_input_image.convert("RGB"))
        annotations = self.auto_sam.generate(img_np)
        logger.info(f"AutoSAM generated {len(annotations)} annotations")
        annotations = sorted(annotations, key=lambda x: x['area'])
        for idx, annotation in enumerate(annotations):
            img_tmp = np.zeros((img_np.shape[0], img_np.shape[1], 3))
            img_tmp[annotation['segmentation']] = img_np[annotation['segmentation']]
            img_np[annotation['segmentation']] = np.array([0, 0, 0])
            img_tmp = Image.fromarray(img_tmp.astype(np.uint8))
            img_tmp.save(os.path.join(layout_output_path, f"{idx}.png"))
        img_np = Image.fromarray(img_np.astype(np.uint8))
        img_np.save(os.path.join(layout_output_path, "leftover.png"))


    def image_layer(self, layout_input_image_or_path, layout_output_path: str) -> str:
        if isinstance(layout_input_image_or_path, str):
            logger.info("Image layer division batch processing")
            all_files = glob.glob(os.path.join(layout_input_image_or_path, "*"))
            for image_index, input_image_file in enumerate(all_files):
                logger.info(f"Processing {image_index}/{len(all_files)} {input_image_file}")
                try:
                    input_image = Image.open(input_image_file)
                    output_directory = os.path.join(layout_output_path, os.path.splitext(os.path.basename(input_image_file))[0])
                    from pathlib import Path
                    Path(output_directory).mkdir(exist_ok=True)
                except:
                    logger.warn(f"File {input_image_file} not image, skipped.")
                    continue
                self.layer_single_image(input_image, output_directory)
        else:
            self.layer_single_image(layout_input_image_or_path, layout_output_path)
        return "Done"


    def semantic_segmentation(self, input_image: Image.Image, annotator_name: str, processor_res: int, 
                              use_pixel_perfect: bool, resize_mode: int, target_W: int, target_H: int) -> Tuple[List[Image.Image], str]:
        if input_image is None:
            return [], "No input image."
        if "seg" in annotator_name:
            try:
                from scripts.processor import uniformer, oneformer_coco, oneformer_ade20k
                from scripts.external_code import pixel_perfect_resolution
                oneformers = {
                    "ade20k": oneformer_ade20k,
                    "coco": oneformer_coco
                }
            except:
                return [], "ControlNet extension not found."
            input_image_np = np.array(input_image)
            if use_pixel_perfect:
                processor_res = pixel_perfect_resolution(input_image_np, resize_mode, target_W, target_H)
            logger.info("Generating semantic segmentation without SAM")
            if annotator_name == "seg_ufade20k":
                original_semantic = uniformer(input_image_np, processor_res)
            else:
                dataset = annotator_name.split('_')[-1][2:]
                original_semantic = oneformers[dataset](input_image_np, processor_res)
            logger.info("Generating semantic segmentation with SAM")
            sam_semantic = self.strengthen_semmantic_seg(np.array(original_semantic), input_image_np)
            output_gallery = [original_semantic, sam_semantic, self.blend_image_and_seg(input_image, original_semantic), self.blend_image_and_seg(input_image, sam_semantic)]
            return output_gallery, f"Done. Left is segmentation before SAM, right is segmentation after SAM."
        else:
            return self.random_segmentation(input_image)


    def categorical_mask_image(self, crop_processor: str, crop_processor_res: int, crop_category_input: List[int], crop_input_image: Image.Image,
                               crop_pixel_perfect: bool, crop_resize_mode: int, target_W: int, target_H: int) -> Tuple[np.ndarray, Image.Image]:
        if crop_input_image is None:
            return "No input image."
        try:
            from scripts.processor import uniformer, oneformer_coco, oneformer_ade20k
            from scripts.external_code import pixel_perfect_resolution
            oneformers = {
                "ade20k": oneformer_ade20k,
                "coco": oneformer_coco
            }
        except:
            return [], "ControlNet extension not found."
        filter_classes = crop_category_input
        if len(filter_classes) == 0:
            return "No class selected."
        try:
            filter_classes = [int(i) for i in filter_classes]
        except:
            return "Illegal class id. You may have input some string."
        crop_input_image_np = np.array(crop_input_image)
        if crop_pixel_perfect:
            crop_processor_res = pixel_perfect_resolution(crop_input_image_np, crop_resize_mode, target_W, target_H)
        crop_input_image_copy = copy.deepcopy(crop_input_image)
        logger.info(f"Generating categories with processor {crop_processor}")
        if crop_processor == "seg_ufade20k":
            original_semantic = uniformer(crop_input_image_np, crop_processor_res)
        else:
            dataset = crop_processor.split('_')[-1][2:]
            original_semantic = oneformers[dataset](crop_input_image_np, crop_processor_res)
        sam_semantic = self.strengthen_semmantic_seg(np.array(original_semantic), crop_input_image_np)
        mask = np.zeros(sam_semantic.shape, dtype=np.bool_)
        from scripts.sam_config import SEMANTIC_CATEGORIES
        for i in filter_classes:
            mask[np.equal(sam_semantic, SEMANTIC_CATEGORIES[crop_processor][i])] = True
        return mask, crop_input_image_copy


    def register_auto_sam(self, sam, 
        auto_sam_points_per_side, auto_sam_points_per_batch, auto_sam_pred_iou_thresh, 
        auto_sam_stability_score_thresh, auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
        auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
        auto_sam_crop_n_points_downscale_factor, auto_sam_min_mask_region_area, auto_sam_output_mode):
        self.auto_sam = SamAutomaticMaskGeneratorHQ(
            sam, auto_sam_points_per_side, auto_sam_points_per_batch, 
            auto_sam_pred_iou_thresh, auto_sam_stability_score_thresh, 
            auto_sam_stability_score_offset, auto_sam_box_nms_thresh, 
            auto_sam_crop_n_layers, auto_sam_crop_nms_thresh, auto_sam_crop_overlap_ratio, 
            auto_sam_crop_n_points_downscale_factor, None, 
            auto_sam_min_mask_region_area, auto_sam_output_mode)
