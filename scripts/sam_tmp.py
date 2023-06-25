import os
import torch
from modules import shared
from modules.safe import unsafe_torch_load, load
from modules.devices import get_device_for, cpu
from modules.paths import models_path
from scripts.sam_state import sam_extension_dir
from scripts.sam_log import logger
from sam_hq.build_sam_hq import sam_model_registry
from sam_hq.predictor import SamPredictorHQ

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
        self.sam_model = None
        self.sam_model_type = ""
        self.sam_model_wrapper = None
        self.sam_device = get_device_for("sam")


    def load_sam_model(self, sam_checkpoint_name: str) -> None:
        if sam_checkpoint_name not in self.sam_model_info.keys():
            logger.error(f"Invalid SAM model checkpoint name: {sam_checkpoint_name}")
        elif self.sam_model is None or self.sam_model_type != sam_checkpoint_name:
            logger.info(f"Initializing {sam_checkpoint_name} to {self.sam_device}")
            user_sam_model_dir = shared.opts.data.get("sam_model_path", "")
            sd_sam_model_dir = os.path.join(models_path, "sam")
            scripts_sam_model_dir = os.path.join(sam_extension_dir, "models/sam")
            sam_model_dir = user_sam_model_dir if user_sam_model_dir != "" else (sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir)
            sam_checkpoint_path = os.path.join(sam_model_dir, sam_checkpoint_name)
            if not os.path.exists(sam_checkpoint_path):
                sam_url = self.sam_model_info[sam_checkpoint_name]
                logger.info(f"Downloading SAM model from {sam_url} to {sam_checkpoint_path}")
                torch.hub.download_url_to_file(sam_url, sam_model_dir)
            logger.info(f"Loading SAM model from {sam_checkpoint_path}")
            torch.load = unsafe_torch_load
            self.sam_model = sam_model_registry[sam_checkpoint_name](checkpoint=sam_checkpoint_path).to(self.sam_device)
            torch.load = load
            self.sam_model_type = sam_checkpoint_name
            self.sam_model_wrapper = SamPredictorHQ(self.sam_model, 'hq' in sam_checkpoint_name) if "Fast" not in sam_checkpoint_name else self.sam_model
    

    def change_device(self, use_cpu: bool) -> None:
        self.sam_device = cpu if use_cpu else get_device_for("sam")


    def __call__(self, point_coords=None, point_labels=None, boxes=None, multimask_output=True, global_point=False, use_numpy=False, use_mam=False):
        pass