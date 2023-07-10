from typing import List
import os
from PIL import Image
import torch

from modules import shared
from modules.devices import device
from sam_utils.logger import logger


class Detection:
    """Detection related process.
    """

    def __init__(self) -> None:
        """Initialize detection related process.
        """
        self.dino_model = None
        self.dino_model_type = ""
        from scripts.sam_state import sam_extension_dir
        self.dino_model_dir = os.path.join(sam_extension_dir, "models/grounding-dino")
        self.dino_model_info = {
            "GroundingDINO_SwinT_OGC (694MB)": {
                "checkpoint": "groundingdino_swint_ogc.pth",
                "config": os.path.join(self.dino_model_dir, "GroundingDINO_SwinT_OGC.py"),
                "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
            },
            "GroundingDINO_SwinB (938MB)": {
                "checkpoint": "groundingdino_swinb_cogcoor.pth",
                "config": os.path.join(self.dino_model_dir, "GroundingDINO_SwinB.cfg.py"),
                "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
            },
        }


    def _load_dino_model(self, dino_checkpoint: str, use_pip_dino: bool) -> None:
        """Load GroundignDINO model to device.

        Args:
            dino_checkpoint (str): GroundingDINO checkpoint name.
            use_pip_dino (bool): If True, use pip installed GroundingDINO. If False, use local GroundingDINO.
        """
        logger.info(f"Initializing GroundingDINO {dino_checkpoint}")
        if self.dino_model is None or dino_checkpoint != self.dino_model_type:
            self.clear()
            if use_pip_dino:
                from groundingdino.models import build_model
                from groundingdino.util.slconfig import SLConfig
                from groundingdino.util.utils import clean_state_dict
            else:
                from thirdparty.groundingdino.models import build_model
                from thirdparty.groundingdino.util.slconfig import SLConfig
                from thirdparty.groundingdino.util.utils import clean_state_dict
            args = SLConfig.fromfile(self.dino_model_info[dino_checkpoint]["config"])
            dino = build_model(args)
            checkpoint = torch.hub.load_state_dict_from_url(
                self.dino_model_info[dino_checkpoint]["url"], self.dino_model_dir)
            dino.load_state_dict(clean_state_dict(
                checkpoint['model']), strict=False)
            dino.eval()
            self.dino_model = dino
            self.dino_model_type = dino_checkpoint
        self.dino_model.to(device=device)


    def _load_dino_image(self, image_pil: Image.Image, use_pip_dino: bool) -> torch.Tensor:
        """Transform image to make the image applicable to GroundingDINO.

        Args:
            image_pil (Image.Image): Input image in PIL format.
            use_pip_dino (bool): If True, use pip installed GroundingDINO. If False, use local GroundingDINO.

        Returns:
            torch.Tensor: Transformed image in torch.Tensor format.
        """
        if use_pip_dino:
            import groundingdino.datasets.transforms as T
        else:
            from thirdparty.groundingdino.datasets import transforms as T
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image


    def _get_grounding_output(self, image: torch.Tensor, caption: str, box_threshold: float) -> torch.Tensor:
        """Inference GroundingDINO model.

        Args:
            image (torch.Tensor): transformed input image.
            caption (str): string caption.
            box_threshold (float): bbox threshold.

        Returns:
            torch.Tensor: generated bounding boxes.
        """
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(device)
        with torch.no_grad():
            outputs = self.dino_model(image[None], captions=[caption])
        if shared.cmd_opts.lowvram:
            self.unload_model()
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        return boxes_filt.cpu()


    def dino_predict(self, input_image: Image.Image, dino_model_name: str, text_prompt: str, box_threshold: float) -> List[List[float]]:
        """Exposed API for GroundingDINO inference.

        Args:
            input_image (Image.Image): input image.
            dino_model_name (str): GroundingDINO model name.
            text_prompt (str): string prompt.
            box_threshold (float): bbox threshold.

        Returns:
            List[List[float]]: generated N * xyxy bounding boxes.
        """
        from sam_utils.util import install_goundingdino
        install_success = install_goundingdino()
        logger.info("Running GroundingDINO Inference")
        dino_image = self._load_dino_image(input_image.convert("RGB"), install_success)
        self._load_dino_model(dino_model_name, install_success)

        boxes_filt = self._get_grounding_output(
            dino_image, text_prompt, box_threshold
        )

        H, W = input_image.size[1], input_image.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        return boxes_filt.tolist()
    

    def check_yolo_availability(self) -> List[str]:
        """Check if YOLO models are available. Do not check if YOLO not enabled.

        Returns:
            List[str]: available YOLO models.
        """
        if shared.opts.data.get("sam_use_local_yolo", False):
            from modules.paths import models_path
            sd_yolo_model_dir = os.path.join(models_path, "ultralytics")
            return [name for name in os.listdir(sd_yolo_model_dir) if (".pth" in name or ".pt" in name)]
        else:
            return []
    

    def yolo_predict(self, input_image: Image.Image, yolo_model_name: str, conf=0.4) -> List[List[float]]:
        """Run detection inference with models based on YOLO.

        Args:
            input_image (np.ndarray): input image, expect shape HW3.
            conf (float, optional): object confidence threshold. Defaults to 0.4.

        Raises:
            RuntimeError: not getting any bbox. Might be caused by high conf or non-detection/segmentation model.

        Returns:
            np.ndarray: generated N * xyxy bounding boxes.
        """
        from ultralytics import YOLO
        assert shared.opts.data.get("sam_use_yolo_models", False), "YOLO models are not enabled. Please enable in settings/Segment Anything."
        logger.info("Loading YOLO model.")
        from modules.paths import models_path
        sd_yolo_model_dir = os.path.join(models_path, "ultralytics")
        self.dino_model = YOLO(os.path.join(sd_yolo_model_dir, yolo_model_name)).to(device)
        self.dino_model_type = yolo_model_name
        logger.info("Running YOLO inference.")
        pred = self.dino_model(input_image, conf=conf)
        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        if bboxes.size == 0:
            error_msg = "You are not getting any bbox. There are 2 possible reasons. "\
                        "1. You set up a high conf which means that you should lower the conf. "\
                        "2. You are using a non-detection model which means that you should check your model type."
            raise RuntimeError(error_msg)
        else:
            return bboxes.tolist()


    def __call__(self, input_image: Image.Image, model_name: str, text_prompt: str, box_threshold: float, conf=0.4) -> List[List[float]]:
        """Exposed API for detection inference."""
        if model_name in self.dino_model_info.keys():
            return self.dino_predict(input_image, model_name, text_prompt, box_threshold)
        elif model_name in self.check_yolo_availability():
            return self.yolo_predict(input_image, model_name, conf)
        else:
            raise ValueError(f"Detection model {model_name} not found.")


    def clear(self) -> None:
        """Clear detection model from any memory.
        """
        del self.dino_model
        self.dino_model = None


    def unload_model(self) -> None:
        """Unload detection model from GPU to CPU.
        """
        if self.dino_model is not None:
            self.dino_model.cpu()
