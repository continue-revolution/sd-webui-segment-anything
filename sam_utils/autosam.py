from typing import List, Tuple, Union, Optional
import os
import glob
import copy
from PIL import Image
import numpy as np
import torch
from segment_anything.modeling import Sam
from thirdparty.fastsam import FastSAM
from thirdparty.sam_hq.automatic import SamAutomaticMaskGeneratorHQ
from sam_utils.segment import Segmentation
from sam_utils.logger import logger
from sam_utils.util import blend_image_and_seg


class AutoSAM:
    """Automatic segmentation."""

    def __init__(self, sam: Segmentation) -> None:
        """AutoSAM initialization.

        Args:
            sam (Segmentation): global Segmentation instance.
        """
        self.sam = sam
        self.auto_sam: Union[SamAutomaticMaskGeneratorHQ, FastSAM] = None
        self.fastsam_conf = None
        self.fastsam_iou = None
    

    def auto_generate(self, img: np.ndarray) -> List[dict]:
        """Generate segmentation.

        Args:
            img (np.ndarray): input image.

        Returns:
            List[dict]: list of segmentation masks.
        """
        return self.auto_sam.generate(img) if type(self.auto_sam) == SamAutomaticMaskGeneratorHQ else \
               self.auto_sam(img, device=self.sam.sam_device, retina_masks=True, imgsz=1024, conf=self.fastsam_conf, iou=self.fastsam_iou)


    def strengthen_semantic_seg(self, class_ids: np.ndarray, img: np.ndarray) -> np.ndarray:
        # TODO: class_ids use multiple dimensions, categorical mask single and batch
        logger.info("AutoSAM strengthening semantic segmentation")
        from sam_utils.util import install_pycocotools
        install_pycocotools()
        import pycocotools.mask as maskUtils
        semantc_mask = copy.deepcopy(class_ids)
        annotations = self.auto_generate(img)
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
        """Random segmentation for EditAnything

        Args:
            img (Image.Image): input image.

        Raises:
            ModuleNotFoundError: ControlNet not installed.

        Returns:
            Tuple[List[Image.Image], str]: List of 3 displayed images and output message.
        """
        logger.info("AutoSAM generating random segmentation for EditAnything")
        img_np = np.array(img.convert("RGB"))
        annotations = self.auto_generate(img_np)
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
            raise ModuleNotFoundError("ControlNet extension not found.")
        detected_map = HWC3(detected_map.astype(np.uint8))
        logger.info("AutoSAM generation process end")
        return [blend_image_and_seg(img_np, color_map), Image.fromarray(color_map), Image.fromarray(detected_map)], \
            "Random segmentation done. Left above (0) is blended image, right above (1) is random segmentation, left below (2) is Edit-Anything control input."


    def layout_single_image(self, input_image: Image.Image, output_path: str) -> None:
        """Single image layout generation.

        Args:
            input_image (Image.Image): input image.
            output_path (str): output path.
        """
        img_np = np.array(input_image.convert("RGB"))
        annotations = self.auto_generate(img_np)
        logger.info(f"AutoSAM generated {len(annotations)} annotations")
        annotations = sorted(annotations, key=lambda x: x['area'])
        for idx, annotation in enumerate(annotations):
            img_tmp = np.zeros((img_np.shape[0], img_np.shape[1], 3))
            img_tmp[annotation['segmentation']] = img_np[annotation['segmentation']]
            img_np[annotation['segmentation']] = np.array([0, 0, 0])
            img_tmp = Image.fromarray(img_tmp.astype(np.uint8))
            img_tmp.save(os.path.join(output_path, f"{idx}.png"))
        img_np = Image.fromarray(img_np.astype(np.uint8))
        img_np.save(os.path.join(output_path, "leftover.png"))


    def layout(self, input_image_or_path: Union[str, Image.Image], output_path: str) -> str:
        """Single or bath layout generation.

        Args:
            input_image_or_path (Union[str, Image.Image]): input imag or path.
            output_path (str): output path.

        Returns:
            str: generation message.
        """
        if isinstance(input_image_or_path, str):
            logger.info("Image layer division batch processing")
            all_files = glob.glob(os.path.join(input_image_or_path, "*"))
            for image_index, input_image_file in enumerate(all_files):
                logger.info(f"Processing {image_index}/{len(all_files)} {input_image_file}")
                try:
                    input_image = Image.open(input_image_file)
                    output_directory = os.path.join(output_path, os.path.splitext(os.path.basename(input_image_file))[0])
                    from pathlib import Path
                    Path(output_directory).mkdir(exist_ok=True)
                except:
                    logger.warn(f"File {input_image_file} not image, skipped.")
                    continue
                self.layout_single_image(input_image, output_directory)
        else:
            self.layout_single_image(input_image_or_path, output_path)
        return "Done"


    def semantic_segmentation(self, input_image: Image.Image, annotator_name: str, processor_res: int, 
                              use_pixel_perfect: bool, resize_mode: int, target_W: int, target_H: int) -> Tuple[List[Image.Image], str]:
        """Semantic segmentation enhanced by segment anything.

        Args:
            input_image (Image.Image): input image.
            annotator_name (str): annotator name. Should be one of "seg_ufade20k"|"seg_ofade20k"|"seg_ofcoco".
            processor_res (int): processor resolution. Support 64-2048.
            use_pixel_perfect (bool): whether to use pixel perfect written by lllyasviel.
            resize_mode (int): resize mode for pixel perfect, should be 0|1|2.
            target_W (int): target width for pixel perfect.
            target_H (int): target height for pixel perfect.

        Raises:
            ModuleNotFoundError: ControlNet not installed.

        Returns:
            Tuple[List[Image.Image], str]: list of 4 displayed images and message.
        """
        assert input_image is not None, "No input image."
        if "seg" in annotator_name:
            try:
                from scripts.processor import uniformer, oneformer_coco, oneformer_ade20k
                from scripts.external_code import pixel_perfect_resolution
                oneformers = {
                    "ade20k": oneformer_ade20k,
                    "coco": oneformer_coco
                }
            except:
                raise ModuleNotFoundError("ControlNet extension not found.")
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
            sam_semantic = self.strengthen_semantic_seg(np.array(original_semantic), input_image_np)
            output_gallery = [original_semantic, sam_semantic, blend_image_and_seg(input_image, original_semantic), blend_image_and_seg(input_image, sam_semantic)]
            return output_gallery, "Done. Left is segmentation before SAM, right is segmentation after SAM."
        else:
            return self.random_segmentation(input_image)


    def categorical_mask_image(self, annotator_name: str, processor_res: int, category_input: List[int], input_image: Image.Image,
                               use_pixel_perfect: bool, resize_mode: int, target_W: int, target_H: int) -> Tuple[np.ndarray, Image.Image]:
        """Single image categorical mask.

        Args:
            annotator_name (str): annotator name. Should be one of "seg_ufade20k"|"seg_ofade20k"|"seg_ofcoco".
            processor_res (int): processor resolution. Support 64-2048.
            category_input (List[int]): category input.
            input_image (Image.Image): input image.
            use_pixel_perfect (bool): whether to use pixel perfect written by lllyasviel.
            resize_mode (int): resize mode for pixel perfect, should be 0|1|2.
            target_W (int): target width for pixel perfect.
            target_H (int): target height for pixel perfect.

        Raises:
            ModuleNotFoundError: ControlNet not installed.
            AssertionError: Illegal class id.

        Returns:
            Tuple[np.ndarray, Image.Image]: mask in resized shape and resized input image.
        """
        assert input_image is not None, "No input image."
        try:
            from scripts.processor import uniformer, oneformer_coco, oneformer_ade20k
            from scripts.external_code import pixel_perfect_resolution
            oneformers = {
                "ade20k": oneformer_ade20k,
                "coco": oneformer_coco
            }
        except:
            raise ModuleNotFoundError("ControlNet extension not found.")
        filter_classes = category_input
        assert len(filter_classes) > 0, "No class selected."
        try:
            filter_classes = [int(i) for i in filter_classes]
        except:
            raise AssertionError("Illegal class id. You may have input some string.")
        input_image_np = np.array(input_image)
        if use_pixel_perfect:
            processor_res = pixel_perfect_resolution(input_image_np, resize_mode, target_W, target_H)
        crop_input_image_copy = copy.deepcopy(input_image)
        logger.info(f"Generating categories with processor {annotator_name}")
        if annotator_name == "seg_ufade20k":
            original_semantic = uniformer(input_image_np, processor_res)
        else:
            dataset = annotator_name.split('_')[-1][2:]
            original_semantic = oneformers[dataset](input_image_np, processor_res)
        sam_semantic = self.strengthen_semantic_seg(np.array(original_semantic), input_image_np)
        mask = np.zeros(sam_semantic.shape, dtype=np.bool_)
        from sam_utils.config import SEMANTIC_CATEGORIES
        for i in filter_classes:
            mask[np.equal(sam_semantic, SEMANTIC_CATEGORIES[annotator_name][i])] = True
        return mask, crop_input_image_copy


    def register(
        self,
        sam_model_name: str, 
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        fastsam_conf: float = 0.4,
        fastsam_iou: float = 0.9) -> None:
        """Register AutoSAM module."""
        self.sam.load_sam_model(sam_model_name)
        assert type(self.sam.sam_model) in [FastSAM, Sam], f"{sam_model_name} does not support auto segmentation."
        if type(self.sam.sam_model) == FastSAM:
            self.fastsam_conf = fastsam_conf
            self.fastsam_iou = fastsam_iou
            self.auto_sam = self.sam.sam_model
        else:
            self.auto_sam = SamAutomaticMaskGeneratorHQ(
                self.sam.sam_model, points_per_side, points_per_batch, pred_iou_thresh,
                stability_score_thresh, stability_score_offset, box_nms_thresh,
                crop_n_layers, crop_nms_thresh, crop_overlap_ratio, crop_n_points_downscale_factor, None, 
                min_mask_region_area, output_mode)
