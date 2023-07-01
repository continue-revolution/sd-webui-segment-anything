from typing import List
import os
import torch
import numpy as np
from ultralytics import YOLO
from modules import shared
from modules.safe import unsafe_torch_load, load
from modules.devices import get_device_for, cpu
from modules.paths import models_path
from scripts.sam_state import sam_extension_dir
from scripts.sam_log import logger
from sam_utils.util import ModelInfo
from sam_hq.build_sam_hq import sam_model_registry
from sam_hq.predictor import SamPredictorHQ
from mam.m2m import SamM2M


class Segmentation:

    def __init__(self) -> None:
        self.sam_model_info = {
            "sam_vit_h_4b8939.pth"  : ModelInfo("SAM", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "Meta", "2.56GB"),
            "sam_vit_l_0b3195.pth"  : ModelInfo("SAM", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "Meta", "1.25GB"),
            "sam_vit_b_01ec64.pth"  : ModelInfo("SAM", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "Meta", "375MB"),
            "sam_hq_vit_h.pth"      : ModelInfo("SAM-HQ", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth", "SysCV", "2.57GB"),
            "sam_hq_vit_l.pth"      : ModelInfo("SAM-HQ", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth", "SysCV", "1.25GB"),
            "sam_hq_vit_b.pth"      : ModelInfo("SAM-HQ", "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth", "SysCV", "379MB"),
            "mobile_sam.pt"         : ModelInfo("SAM", "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt", "KHU", "39MB"),
            "FastSAM-x.pt"          : ModelInfo("SAM-YOLO", "https://huggingface.co/conrevo/SAM4WebUI-Extension-Models/resolve/main/FastSAM-x.pt", "CASIA-IVA-Lab", "138MB"),
        }
        self.check_model_availability()
        self.sam_model = None
        self.sam_model_name = ""
        self.sam_model_wrapper = None
        self.sam_device = get_device_for("sam")
        self.sam_m2m = SamM2M()


    def check_model_availability(self) -> List[str]:
        # retrieve all models in all the model directories
        user_sam_model_dir = shared.opts.data.get("sam_model_path", "")
        sd_sam_model_dir = os.path.join(models_path, "sam")
        scripts_sam_model_dir = os.path.join(sam_extension_dir, "models/sam")
        sd_yolo_model_dir = os.path.join(models_path, "ultralytics")
        sam_model_dirs = [sd_sam_model_dir, scripts_sam_model_dir]
        if user_sam_model_dir != "":
            sam_model_dirs.append(user_sam_model_dir)
        if shared.opts.data.get("use_yolo_models", False):
            sam_model_dirs.append(sd_yolo_model_dir)
        for dir in sam_model_dirs:
            if os.path.isdir(dir):
                # for each model inside sam_model_info, update its information
                sam_model_names = [name for name in os.listdir(dir) if (".pth" in name or ".pt" in name)]
                for name in sam_model_names:
                    if name in self.sam_model_info.keys():
                        self.sam_model_info[name].local_path(os.path.join(dir, name))
                    elif shared.opts.data.get("use_yolo_models", False):
                        logger.warn(f"Model {name} not found in support list, default to use YOLO as initializer")
                        self.sam_model_info[name] = ModelInfo("YOLO", os.path.join(dir, name), "?", "?", "downloaded")
        return [val.get_info(key) for key, val in self.sam_model_info.items()]


    def load_sam_model(self, sam_checkpoint_name: str) -> None:
        sam_checkpoint_name = sam_checkpoint_name.split(" ")[0]
        if self.sam_model is None or self.sam_model_name != sam_checkpoint_name:
            if sam_checkpoint_name not in self.sam_model_info.keys():
                error_msg = f"{sam_checkpoint_name} not found and cannot be auto downloaded"
                logger.error(f"{error_msg}")
                raise Exception(error_msg)
            if "http" in self.sam_model_info[sam_checkpoint_name].url:
                sam_url = self.sam_model_info[sam_checkpoint_name].url
                user_dir = shared.opts.data.get("sam_model_path", "")
                sd_dir = os.path.join(models_path, "sam")
                scripts_dir = os.path.join(sam_extension_dir, "models/sam")
                sam_model_dir = user_dir if user_dir != "" else (sd_dir if os.path.isdir(sd_dir) else scripts_dir)
                logger.info(f"Downloading segmentation model {sam_checkpoint_name} from {sam_url} to {sam_model_dir}")
                try:
                    torch.hub.download_url_to_file(sam_url, sam_model_dir)
                except:
                    error_msg = f"Cannot connect to {sam_url}. Set terminal proxy or download via browser to {sam_model_dir}"
                    logger.error(f"{error_msg}")
                    raise Exception(error_msg)
            device_name = "GPU" if "cuda" in str(self.sam_device).lower() else "CPU"
            logger.info(f"Initializing {sam_checkpoint_name} to {device_name}")
            model_type = self.sam_model_info[sam_checkpoint_name].model_type
            model_path = self.sam_model_info[sam_checkpoint_name].url
            torch.load = unsafe_torch_load
            if "YOLO" not in model_type:
                logger.info(f"Loading SAM model from {model_path}")
                self.sam_model = sam_model_registry[sam_checkpoint_name](checkpoint=model_path)
                self.sam_model_wrapper = SamPredictorHQ(self.sam_model, 'HQ' in model_type)
            else:
                logger.info(f"Loading YOLO model from {model_path}")
                self.sam_model = YOLO(model_path)
                self.sam_model_wrapper = self.sam_model
            self.sam_model_name = sam_checkpoint_name
            torch.load = load
        self.sam_model.to(self.sam_device)
    

    def change_device(self, use_cpu: bool) -> None:
        self.sam_device = cpu if use_cpu else get_device_for("sam")


    def __call__(self, 
            input_image: np.ndarray,
            point_coords: List[List[int]]=None,
            point_labels: List[List[int]]=None, 
            boxes: torch.Tensor=None, 
            multimask_output=True, 
            use_mam=False):
        pass


    def sam_predict(self, 
            input_image: np.ndarray,
            positive_points: List[List[int]]=None,
            negative_points: List[List[int]]=None, 
            boxes_coords: torch.Tensor=None,
            boxes_labels: List[bool]=None,
            multimask_output=True,
            merge_point_and_box=True,
            point_with_box=False,
            use_mam=False,
            use_mam_for_each_infer=False,
            mam_guidance_mode: str="mask") -> np.ndarray:
        """Run segmentation inference with models based on segment anything.

        Args:
            input_image (np.ndarray): input image, expect shape HW3.
            positive_points (List[List[int]], optional): positive point prompts. Defaults to None.
            negative_points (List[List[int]], optional): negative point prompts. Defaults to None.
            boxes_coords (torch.Tensor, optional): bbox inputs, expect shape xyxy. Defaults to None.
            boxes_labels (List[bool], optional): bbox labels, support positive & negative bboxes. Defaults to None.
            multimask_output (bool, optional): output 3 masks or not. Defaults to True.
            merge_point_and_box (bool, optional): if True, output point masks || bbox masks; otherwise, output point masks && bbox masks. Valid only if point_with_box is False. Defaults to True.
            point_with_box (bool, optional): always send bboxes and points to the model at the same time. Defaults to False.
            use_mam (bool, optional): use Matting-Anything. Defaults to False.
            use_mam_for_each_infer (bool, optional): use Matting-Anything for each SAM inference. Valid only if use_mam is True. Defaults to False.
            mam_guidance_mode (str, optional): guidance model for Matting-Anything. Expect "mask" or "bbox". Valid only if use_mam is True. Defaults to "mask".

        Returns:
            np.ndarray: mask output, expect shape 11HW or 31HW.
        """
        masks_for_points, masks_for_boxes, masks = None, None, None
        self.sam_model_wrapper.set_image(input_image)
        if use_mam:
            self.sam_m2m.load_m2m() # TODO: raise exception if network problem
        # When always send bboxes and points to the model at the same time.
        if point_coords is not None and point_labels is not None and boxes_coords is not None and point_with_box:
            point_coords = np.array(positive_points + negative_points)
            point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
            point_labels_neg = np.array([0] * len(positive_points) + [1] * len(negative_points))
            positive_masks, negative_masks, positive_low_res_masks, negative_low_res_masks = [], [], [], []
            # Inference for each positive bbox.
            for box in boxes_coords[boxes_labels]:
                mask, _, low_res_mask = self.sam_model_wrapper.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box.numpy(),
                    multimask_output=multimask_output)
                mask = mask[:, None, ...]
                low_res_mask = low_res_mask[:, None, ...]
                if use_mam:
                    low_res_mask_logits = low_res_mask > self.sam_model_wrapper.model.mask_threshold
                    if use_mam_for_each_infer:
                        mask = self.sam_m2m.forward(
                            self.sam_model_wrapper.features, torch.tensor(input_image), low_res_mask_logits, mask, 
                            self.sam_model_wrapper.original_size, self.sam_model_wrapper.input_size, mam_guidance_mode)
                    else:
                        positive_low_res_masks.append(low_res_mask_logits)
                positive_masks.append(mask)
            positive_masks = np.logical_or(np.stack(positive_masks, 0))
            # Inference for each negative bbox.
            for box in boxes_coords[[not i for i in boxes_labels]]:
                mask, _, low_res_mask = self.sam_model_wrapper.predict(
                    point_coords=point_coords,
                    point_labels=point_labels_neg,
                    box=box.numpy(),
                    multimask_output=multimask_output)
                mask = mask[:, None, ...]
                low_res_mask = low_res_mask[:, None, ...]
                if use_mam:
                    low_res_mask_logits = low_res_mask > self.sam_model_wrapper.model.mask_threshold
                    if use_mam_for_each_infer:
                        mask = self.sam_m2m.forward(
                            self.sam_model_wrapper.features, torch.tensor(input_image), low_res_mask_logits, mask, 
                            self.sam_model_wrapper.original_size, self.sam_model_wrapper.input_size, mam_guidance_mode)
                    else:
                        negative_low_res_masks.append(low_res_mask_logits)
                negative_masks.append(mask)
            negative_masks = np.logical_or(np.stack(negative_masks, 0))
            masks = np.logical_and(positive_masks, ~negative_masks)
            # Matting-Anything inference if not for each inference.
            if use_mam and not use_mam_for_each_infer:
                positive_low_res_masks = np.logical_or(np.stack(positive_low_res_masks, 0))
                negative_low_res_masks = np.logical_or(np.stack(negative_low_res_masks, 0))
                low_res_masks = np.logical_and(positive_low_res_masks, ~negative_low_res_masks)
                masks = self.sam_m2m.forward(
                    self.sam_model_wrapper.features, torch.tensor(input_image), low_res_masks, masks, 
                    self.sam_model_wrapper.original_size, self.sam_model_wrapper.input_size, mam_guidance_mode)
            return masks

        # When separate bbox inference from point inference.    
        if point_coords is not None and point_labels is not None:
            point_coords = np.array(positive_points + negative_points)
            point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
            masks_for_points, _, _ = self.sam_model_wrapper.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=multimask_output)
            masks_for_points = masks_for_points[:, None, ...]
            # TODO: m2m
        if boxes_coords is not None:
            transformed_boxes = self.sam_model_wrapper.transform.apply_boxes_torch(boxes_coords, input_image.shape[:2])
            masks_for_boxes, _, _ = self.sam_model_wrapper.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.sam_device),
                multimask_output=multimask_output)
            masks_for_boxes = masks_for_boxes.permute(1, 0, 2, 3).cpu().numpy()
            # TODO: m2m
        if masks_for_boxes is not None and masks_for_points is not None:
            if merge_point_and_box:
                masks = np.logical_or(masks_for_points, masks_for_boxes)
            else:
                masks = np.logical_and(masks_for_points, masks_for_boxes)
        elif masks_for_boxes is not None:
            masks = masks_for_boxes
        elif masks_for_points is not None:
            masks = masks_for_points
        # TODO: m2m
        return masks
    

    def yolo_predict():
        pass



# check device for each model type
# how to use yolo for auto
# category name dropdown
# yolo model for segmentation and detection
# zoom in and unify box+point
