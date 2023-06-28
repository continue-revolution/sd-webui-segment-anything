from typing import List, Union
import os
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
from modules import shared
from modules.safe import unsafe_torch_load, load
from modules.devices import get_device_for, cpu
from modules.paths import models_path
from scripts.sam_state import sam_extension_dir
from scripts.sam_log import logger
from sam_hq.build_sam_hq import sam_model_registry
from sam_hq.predictor import SamPredictorHQ
from mam.m2m import SamM2M

class Segmentation:

    def __init__(self) -> None:
        self.sam_model_info = {
            "sam_vit_h_4b8939.pth (Meta, 2.56GB)"   : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "sam_vit_l_0b3195.pth (Meta, 1.25GB)"   : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "sam_vit_b_01ec64.pth (Meta, 375MB)"    : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "sam_hq_vit_h.pth (SysCV, 2.57GB)"      : "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth",
            "sam_hq_vit_l.pth (SysCV, 1.25GB)"      : "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth",
            "sam_hq_vit_b.pth (SysCV, 379MB)"       : "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth",
            "FastSAM-x.pt (CASIA-IVA-Lab, 138MB)"   : "https://huggingface.co/conrevo/SAM4WebUI-Extension-Models/resolve/main/FastSAM-x.pt"
        }
        self.check_model_availability(initialization=True)
        self.sam_model = None
        self.sam_model_type = ""
        self.sam_model_wrapper = None
        self.sam_device = get_device_for("sam")
        self.sam_m2m = SamM2M()

    
    def check_model_availability(self, initialization=False) -> None:
        # retrieve all models in all the model directories
        
        # for each model inside sam_model_info, update its information
        pass


    def load_sam_model(self, sam_checkpoint_name: str) -> None:
        if self.sam_model is None or self.sam_model_type != sam_checkpoint_name:
            logger.info(f"[Segment] Initializing {sam_checkpoint_name} to {self.sam_device}")
            model_type = "sam" if "sam_" in sam_checkpoint_name else "ultralytics"
            user_sam_model_dir = shared.opts.data.get(f"{model_type}_model_path", "")
            sd_sam_model_dir = os.path.join(models_path, model_type)
            scripts_sam_model_dir = os.path.join(sam_extension_dir, f"models/{model_type}")
            sam_model_dir = user_sam_model_dir if user_sam_model_dir != "" else (sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir)
            sam_checkpoint_path = os.path.join(sam_model_dir, sam_checkpoint_name)
            if not os.path.exists(sam_checkpoint_path):
                if sam_checkpoint_name in self.sam_model_info.keys():
                    sam_url = self.sam_model_info[sam_checkpoint_name]
                    logger.info(f"[Segment] Downloading segmentation model from {sam_url} to {sam_checkpoint_path}")
                    torch.hub.download_url_to_file(sam_url, sam_model_dir)
                else:
                    error_msg = f"{sam_checkpoint_name} not found and cannot be auto downloaded"
                    logger.error(f"[Segment] {error_msg}")
                    raise Exception(error_msg)
            torch.load = unsafe_torch_load
            if "sam_" in sam_checkpoint_name:
                logger.info(f"[Segment] Loading SAM model from {sam_checkpoint_path}")
                self.sam_model = sam_model_registry[sam_checkpoint_name](checkpoint=sam_checkpoint_path)
                self.sam_model_wrapper = SamPredictorHQ(self.sam_model, 'hq' in sam_checkpoint_name)
            else:
                logger.info(f"[Segment] Loading YOLO model from {sam_checkpoint_path}")
                self.sam_model = YOLO(sam_checkpoint_path)
                self.sam_model_wrapper = self.sam_model
            self.sam_model_type = sam_checkpoint_name
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
            boxes: torch.Tensor=None, 
            multimask_output=True,
            merge_point_and_box=True) -> np.ndarray:
        masks_for_points, masks_for_boxes, masks = None, None, None
        self.sam_model_wrapper.set_image(input_image)
        if point_coords is not None and point_labels is not None:
            point_coords = np.array(positive_points + negative_points)
            point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
            masks_for_points, _, _ = self.sam_model_wrapper.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=multimask_output)
            masks_for_points = masks_for_points[:, None, ...]
        if boxes is not None:
            transformed_boxes = self.sam_model_wrapper.transform.apply_boxes_torch(boxes, input_image.shape[:2])
            masks_for_boxes, _, _ = self.sam_model_wrapper.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.sam_device),
                multimask_output=True)
            masks_for_boxes = masks_for_boxes.permute(1, 0, 2, 3).cpu().numpy()
        if masks_for_boxes is not None and masks_for_points is not None:
            if merge_point_and_box:
                masks = np.logical_or(masks_for_points, masks_for_boxes)
            else:
                masks = np.logical_and(masks_for_points, masks_for_boxes)
        elif masks_for_boxes is not None:
            masks = masks_for_boxes
        elif masks_for_points is not None:
            masks = masks_for_points
        return masks
    

    def yolo_predict():
        pass



# check device for each model type
# how to use yolo for auto
# category name dropdown
# yolo model for segmentation and detection
