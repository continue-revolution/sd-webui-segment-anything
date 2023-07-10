from typing import List
import gc
import os
import torch
import numpy as np
from ultralytics import YOLO
from thirdparty.fastsam import FastSAM, FastSAMPrompt
from modules import shared
from modules.safe import unsafe_torch_load, load
from modules.devices import get_device_for, cpu, torch_gc
from modules.paths import models_path
from scripts.sam_state import sam_extension_dir
from sam_utils.logger import logger
from sam_utils.util import ModelInfo
from thirdparty.sam_hq.build_sam_hq import sam_model_registry
from thirdparty.sam_hq.predictor import SamPredictorHQ
from thirdparty.mam.m2m import SamM2M


class Segmentation:
    """Segmentation related process."""

    def __init__(self) -> None:
        """Initialize segmentation related process."""
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
        """retrieve all models in all the model directories

        Returns:
            List[str]: Model information displayed on the UI
        """
        user_sam_model_dir = shared.opts.data.get("sam_model_path", "")
        sd_sam_model_dir = os.path.join(models_path, "sam")
        scripts_sam_model_dir = os.path.join(sam_extension_dir, "models/sam")
        sd_yolo_model_dir = os.path.join(models_path, "ultralytics")
        sam_model_dirs = [sd_sam_model_dir, scripts_sam_model_dir]
        if user_sam_model_dir != "":
            sam_model_dirs.append(user_sam_model_dir)
        if shared.opts.data.get("sam_use_yolo_models", False):
            sam_model_dirs.append(sd_yolo_model_dir)
        for dir in sam_model_dirs:
            if os.path.isdir(dir):
                # for each model inside sam_model_info, update its information
                sam_model_names = [name for name in os.listdir(dir) if (".pth" in name or ".pt" in name)]
                for name in sam_model_names:
                    if name in self.sam_model_info.keys():
                        self.sam_model_info[name].local_path(os.path.join(dir, name))
                    elif shared.opts.data.get("sam_use_yolo_models", False):
                        logger.warn(f"Model {name} not found in support list, default to use YOLO as initializer")
                        self.sam_model_info[name] = ModelInfo("YOLO", os.path.join(dir, name), "?", "?", "downloaded")
        return [val.get_info(key) for key, val in self.sam_model_info.items()]


    def load_sam_model(self, sam_checkpoint_name: str) -> None:
        """Load segmentation model.

        Args:
            sam_checkpoint_name (str): The model filename. Do not change.

        Raises:
            RuntimeError: Model file not found in either support list or local model directory.
            RuntimeError: Cannot automatically download model from remote server.
        """
        sam_checkpoint_name = sam_checkpoint_name.split(" ")[0]
        if self.sam_model is None or self.sam_model_name != sam_checkpoint_name:
            if sam_checkpoint_name not in self.sam_model_info.keys():
                error_msg = f"{sam_checkpoint_name} not found and cannot be auto downloaded"
                raise RuntimeError(error_msg)
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
            elif "SAM" in model_type:
                logger.info(f"Loading FastSAM model from {model_path}")
                self.sam_model = FastSAM(model_path)
                self.sam_model_wrapper = self.sam_model
            elif shared.opts.data.get("sam_use_yolo_models", False):
                logger.info(f"Loading YOLO model from {model_path}")
                self.sam_model = YOLO(model_path)
                self.sam_model_wrapper = self.sam_model
            else:
                error_msg = f"Unsupported model type {model_type}"
                raise RuntimeError(error_msg)
            self.sam_model_name = sam_checkpoint_name
            torch.load = load
        self.sam_model.to(self.sam_device)
    

    def change_device(self, use_cpu: bool) -> None:
        """Change the device of the segmentation model.

        Args:
            use_cpu (bool): Whether to use CPU for SAM inference.
        """
        self.sam_device = cpu if use_cpu else get_device_for("sam")


    def sam_predict(self, 
        input_image: np.ndarray,
        positive_points: List[List[int]]=None,
        negative_points: List[List[int]]=None, 
        positive_bbox: List[List[float]]=None,
        negative_bbox: List[List[float]]=None,
        merge_positive=True,
        multimask_output=True,
        point_with_box=False,
        use_mam=False,
        mam_guidance_mode: str="mask") -> np.ndarray:
        """Run segmentation inference with models based on segment anything.

        Args:
            input_image (np.ndarray): input image, expect shape HW3.
            positive_points (List[List[int]], optional): positive point prompts, expect N * xy. Defaults to None.
            negative_points (List[List[int]], optional): negative point prompts, expect N * xy. Defaults to None.
            positive_bbox (List[List[float]], optional): positive bbox prompts, expect N * xyxy. Defaults to None.
            negative_bbox (List[List[float]], optional): negative bbox prompts, expect N * xyxy. Defaults to None.
            merge_positive (bool, optional): OR all positive masks. Defaults to True. Valid only if point_with_box is False.
            multimask_output (bool, optional): output 3 masks or not. Defaults to True.
            point_with_box (bool, optional): always send bboxes and points to the model at the same time. Defaults to False.
            use_mam (bool, optional): use Matting-Anything. Defaults to False.
            mam_guidance_mode (str, optional): guidance model for Matting-Anything. Expect "mask" or "bbox". Defaults to "mask". Valid only if use_mam is True.

        Returns:
            np.ndarray: mask output, expect shape 11HW or 31HW.
        """
        has_points = positive_points is not None or negative_points is not None
        has_bbox = positive_bbox is not None or negative_bbox is not None
        assert has_points or has_bbox, "No input provided. Please provide at least one point or one bbox."
        assert type(self.sam_model_wrapper) == SamPredictorHQ, "Incorrect SAM model wrapper. Expect SamPredictorHQ here."
        mask_shape = ((3, 1) if multimask_output else (1, 1)) + input_image.shape[:2]
        self.sam_model_wrapper.set_image(input_image)

        # If use Matting-Anything, load mam model.
        if use_mam:
            try:
                self.sam_m2m.load_m2m()
            except:
                use_mam = False

        # Matting-Anything inference for each SAM inference.
        def _mam_infer(mask: np.ndarray, low_res_mask: np.ndarray) -> np.ndarray:
            low_res_mask_logits = low_res_mask > self.sam_model_wrapper.model.mask_threshold
            if use_mam:
                mask = self.sam_m2m.forward(
                    self.sam_model_wrapper.features, torch.tensor(input_image), low_res_mask_logits, mask, 
                    self.sam_model_wrapper.original_size, self.sam_model_wrapper.input_size, mam_guidance_mode)
            return mask


        # When always send bboxes and points to SAM at the same time.
        if has_points and has_bbox and point_with_box:
            logger.info(f"SAM {self.sam_model_name} inference with "
                        f"{len(positive_points)} positive points, {len(negative_points)} negative points, "
                        f"{len(positive_bbox)} positive bboxes, {len(negative_bbox)} negative bboxes. "
                        f"For each bbox, all point prompts are affective. Masks for each bbox will be merged.")
            point_coords = np.array(positive_points + negative_points)
            point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
            point_labels_neg = np.array([0] * len(positive_points) + [1] * len(negative_points))

            def _box_infer(_bbox, _point_labels):
                _masks = []
                for box in _bbox:
                    mask, _, low_res_mask = self.sam_model_wrapper.predict(
                        point_coords=point_coords,
                        point_labels=_point_labels,
                        box=np.array(box),
                        multimask_output=multimask_output)
                    mask = mask[:, None, ...]
                    low_res_mask = low_res_mask[:, None, ...]
                    _masks.append(_mam_infer(mask, low_res_mask))
                return np.logical_or.reduce(_masks)
            
            mask_bbox_positive = _box_infer(positive_bbox, point_labels) if positive_bbox is not None else np.ones(shape=mask_shape, dtype=np.bool_)
            mask_bbox_negative = _box_infer(negative_bbox, point_labels_neg) if negative_bbox is not None else np.zeros(shape=mask_shape, dtype=np.bool_)

            return mask_bbox_positive & ~mask_bbox_negative

        # When separate bbox inference from point inference.
        if has_points:
            logger.info(f"SAM {self.sam_model_name} inference with "
                        f"{len(positive_points)} positive points, {len(negative_points)} negative points")
            point_coords = np.array(positive_points + negative_points)
            point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
            mask_points_positive, _, low_res_masks_points_positive = self.sam_model_wrapper.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=multimask_output)
            mask_points_positive = mask_points_positive[:, None, ...]
            low_res_masks_points_positive = low_res_masks_points_positive[:, None, ...]
            mask_points_positive = _mam_infer(mask_points_positive, low_res_masks_points_positive)
        else:
            mask_points_positive = np.ones(shape=mask_shape, dtype=np.bool_)

        def _box_infer(_bbox, _character):
            logger.info(f"SAM {self.sam_model_name} inference with {len(positive_bbox)} {_character} bboxes")
            transformed_boxes = self.sam_model_wrapper.transform.apply_boxes_torch(torch.tensor(_bbox), input_image.shape[:2])
            mask, _, low_res_mask = self.sam_model_wrapper.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(self.sam_device),
                multimask_output=multimask_output)
            mask = mask.permute(1, 0, 2, 3).cpu().numpy()
            low_res_mask = low_res_mask.permute(1, 0, 2, 3).cpu().numpy()
            return _mam_infer(mask, low_res_mask)
        mask_bbox_positive = _box_infer(positive_bbox, "positive") if positive_bbox is not None else np.ones(shape=mask_shape, dtype=np.bool_)
        mask_bbox_negative = _box_infer(negative_bbox, "negative") if negative_bbox is not None else np.zeros(shape=mask_shape, dtype=np.bool_)

        if merge_positive:
            return (mask_points_positive | mask_bbox_positive) & ~mask_bbox_negative
        else:
            return (mask_points_positive & mask_bbox_positive) & ~mask_bbox_negative
    

    def fastsam_predict(self,
        input_image: np.ndarray,
        positive_points: List[List[int]]=None,
        negative_points: List[List[int]]=None, 
        positive_bbox: List[List[float]]=None,
        negative_bbox: List[List[float]]=None,
        positive_text: str="",
        negative_text: str="",
        merge_positive=True,
        merge_negative=True,
        conf=0.4, iou=0.9,) -> np.ndarray:
        """Run segmentation inference with models based on FastSAM. (This is a special kind of YOLO model)

        Args:
            input_image (np.ndarray): input image, expect shape HW3.
            positive_points (List[List[int]], optional): positive point prompts, expect N * xy Defaults to None.
            negative_points (List[List[int]], optional): negative point prompts, expect N * xy Defaults to None.
            positive_bbox (List[List[float]], optional): positive bbox prompts, expect N * xyxy. Defaults to None.
            negative_bbox (List[List[float]], optional): negative bbox prompts, expect N * xyxy. Defaults to None.
            positive_text (str, optional): positive text prompts. Defaults to "".
            negative_text (str, optional): negative text prompts. Defaults to "".
            merge_positive (bool, optional): OR all positive masks. Defaults to True.
            merge_negative (bool, optional): OR all negative masks. Defaults to True.
            conf (float, optional): object confidence threshold. Defaults to 0.4.
            iou (float, optional): iou threshold for filtering the annotations. Defaults to 0.9.

        Returns:
            np.ndarray:  mask output, expect shape 11HW. FastSAM does not support multi-mask selection.
        """
        assert type(self.sam_model_wrapper) == FastSAM, "Incorrect SAM model wrapper. Expect FastSAM here."
        logger.info(f"Running FastSAM {self.sam_model_name} inference.")
        annotation = self.sam_model_wrapper(
            input_image, device=self.sam_device, retina_masks=True, imgsz=1024, conf=conf, iou=iou)
        has_points = positive_points is not None or negative_points is not None
        has_bbox = positive_bbox is not None or negative_bbox is not None
        has_text = positive_text != "" or negative_text != ""
        assert has_points or has_bbox or has_text, "No input provided. Please provide at least one point or one bbox or one text."
        logger.info("Post-processing FastSAM inference.")
        prompt_process = FastSAMPrompt(input_image, annotation, device=self.sam_device)
        mask_shape = (1, 1) + input_image.shape[:2]
        mask_bbox_positive = prompt_process.box_prompt(bboxes=positive_bbox) if positive_bbox is not None else np.ones(shape=mask_shape, dtype=np.bool_)
        mask_bbox_negative = prompt_process.box_prompt(bboxes=negative_bbox) if negative_bbox is not None else np.zeros(shape=mask_shape, dtype=np.bool_)

        mask_text_positive = prompt_process.text_prompt(text=positive_text) if positive_text != "" else np.ones(shape=mask_shape, dtype=np.bool_)
        mask_text_negative = prompt_process.text_prompt(text=negative_text) if negative_text != "" else np.zeros(shape=mask_shape, dtype=np.bool_)

        point_coords = np.array(positive_points + negative_points)
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        mask_points_positive = prompt_process.point_prompt(points=point_coords, pointlabel=point_labels) if has_points else np.ones(shape=mask_shape, dtype=np.bool_)

        if merge_positive:
            mask_positive = mask_bbox_positive | mask_text_positive | mask_points_positive
        else:
            mask_positive = mask_bbox_positive & mask_text_positive & mask_points_positive
        if merge_negative:
            mask_negative = mask_bbox_negative | mask_text_negative
        else:
            mask_negative = mask_bbox_negative & mask_text_negative
        return mask_positive & ~mask_negative


    def yolo_predict(self, input_image: np.ndarray, conf=0.4) -> np.ndarray:
        """Run segmentation inference with models based on YOLO.

        Args:
            input_image (np.ndarray): input image, expect shape HW3.
            conf (float, optional): object confidence threshold. Defaults to 0.4.

        Raises:
            RuntimeError: not getting any bbox. Might be caused by high conf or non-detection/segmentation model.

        Returns:
            np.ndarray: mask output, expect shape 11HW. YOLO does not support multi-mask selection.
        """
        assert shared.opts.data.get("sam_use_yolo_models", False), "YOLO models are not enabled. Please enable in settings/Segment Anything."
        assert type(self.sam_model_wrapper) == YOLO, "Incorrect SAM model wrapper. Expect YOLO here."
        logger.info("Running YOLO inference.")
        pred = self.sam_model_wrapper(input_image, conf=conf)
        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        if bboxes.size == 0:
            error_msg = "You are not getting any bbox. There are 2 possible reasons. "\
                        "1. You set up a high conf which means that you should lower the conf. "\
                        "2. You are using a non-detection/segmentation model which means that you should check your model type."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if pred[0].masks is None:
            logger.warn("You are not using a segmentation model. Will use bbox to create masks.")
            masks = []
            for bbox in bboxes:
                mask_shape = (1, 1) + input_image.shape[:2]
                mask = np.zeros(mask_shape, dtype=bool)
                x1, y1, x2, y2 = bbox
                mask[:, :, y1:y2+1, x1:x2+1] = True
            return np.logical_or.reduce(masks, axis=0)
        else:
            return np.logical_or.reduce(pred[0].masks.data, axis=0)


    def clear(self):
        """Clear segmentation model from CPU & GPU."""
        del self.sam_model
        self.sam_model = None
        self.sam_model_name = ""
        self.sam_model_wrapper = None
        self.sam_m2m.clear()


    def unload_model(self):
        """Move all segmentation models to CPU."""
        if self.sam_model is not None:
            self.sam_model.cpu()
        self.sam_m2m.unload_model()


    def __call__(self, 
        sam_checkpoint_name: str, 
        input_image: np.ndarray,
        positive_points: List[List[int]]=None,
        negative_points: List[List[int]]=None, 
        positive_bbox: List[List[float]]=None,
        negative_bbox: List[List[float]]=None,
        positive_text: str="",
        negative_text: str="",
        merge_positive=True,
        merge_negative=True,
        multimask_output=True,
        point_with_box=False,
        use_mam=False,
        mam_guidance_mode: str="mask",
        conf=0.4, iou=0.9,) -> np.ndarray:
        # use_cpu: bool=False,) -> np.ndarray:
        """Entry for segmentation inference. Load model, run inference, unload model if lowvram.

        Args:
            sam_checkpoint_name (str): The model filename. Do not change.
            input_image (np.ndarray): input image, expect shape HW3.
            positive_points (List[List[int]], optional): positive point prompts, expect N * xy. Defaults to None. Valid for SAM & FastSAM.
            negative_points (List[List[int]], optional): negative point prompts, expect N * xy. Defaults to None. Valid for SAM & FastSAM.
            positive_bbox (List[List[float]], optional): positive bbox prompts, expect N * xyxy. Defaults to None. Valid for SAM & FastSAM.
            negative_bbox (List[List[float]], optional): negative bbox prompts, expect N * xyxy. Defaults to None. Valid for SAM & FastSAM.
            positive_text (str, optional): positive text prompts. Defaults to "". Valid for FastSAM.
            negative_text (str, optional): negative text prompts. Defaults to "". Valid for FastSAM.
            merge_positive (bool, optional): OR all positive masks. Defaults to True. Valid for SAM (point_with_box is True) & FastSAM.
            merge_negative (bool, optional): OR all negative masks. Defaults to True. Valid for FastSAM.
            multimask_output (bool, optional): output 3 masks or not. Defaults to True. Valid for SAM.
            point_with_box (bool, optional): always send bboxes and points to the model at the same time. Defaults to False. Valid for SAM.
            use_mam (bool, optional): use Matting-Anything. Defaults to False. Valid for SAM.
            mam_guidance_mode (str, optional): guidance model for Matting-Anything. Expect "mask" or "bbox". Defaults to "mask". Valid for SAM and use_mam is True.
            conf (float, optional): object confidence threshold. Defaults to 0.4. Valid for FastSAM & YOLO.
            iou (float, optional): iou threshold for filtering the annotations. Defaults to 0.9. Valid for FastSAM.
            use_cpu (bool, optional): use CPU for SAM inference. Defaults to False.

        Returns:
            np.ndarray: _description_
        """
        # self.change_device(use_cpu)
        self.load_sam_model(sam_checkpoint_name)
        if type(self.sam_model_wrapper) == SamPredictorHQ:
            masks = self.sam_predict(
                input_image=input_image,
                positive_points=positive_points,
                negative_points=negative_points, 
                positive_bbox=positive_bbox,
                negative_bbox=negative_bbox,
                merge_positive=merge_positive,
                multimask_output=multimask_output,
                point_with_box=point_with_box,
                use_mam=use_mam,
                mam_guidance_mode=mam_guidance_mode)
        elif type(self.sam_model_wrapper) == FastSAM:
            masks = self.fastsam_predict(
                input_image=input_image,
                positive_points=positive_points,
                negative_points=negative_points, 
                positive_bbox=positive_bbox,
                negative_bbox=negative_bbox,
                positive_text=positive_text,
                negative_text=negative_text,
                merge_positive=merge_positive,
                merge_negative=merge_negative,
                conf=conf, iou=iou)
        else:
            masks = self.yolo_predict(
                input_image=input_image,
                conf=conf)
        if shared.cmd_opts.lowvram:
            self.unload_model()
        gc.collect()
        torch_gc()
        return masks


# how to use yolo for auto
# category name dropdown and dynamic ui
# yolo model for segmentation and detection
# zoom in, unify box+point
# make masks smaller
