from typing import Tuple
import os
import gc
from PIL import Image
import torch

from modules import shared
from modules.devices import device, torch_gc
from scripts.sam_log import logger
import local_groundingdino

class Detection:

    def __init__(self) -> None:
        self.dino_model = None
        self.dino_model_type = ""
        from scripts.sam_state import sam_extension_dir
        self.dino_model_dir = os.path.join(sam_extension_dir, "models/grounding-dino")
        self.dino_model_list = ["GroundingDINO_SwinT_OGC (694MB)", "GroundingDINO_SwinB (938MB)"]
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
        self.dino_install_issue_text = "Please permanently switch to local GroundingDINO on Settings/Segment Anything or submit an issue to https://github.com/IDEA-Research/Grounded-Segment-Anything/issues."


    def _install_goundingdino(self) -> bool:
        if shared.opts.data.get("sam_use_local_groundingdino", False):
            logger.info("Using local groundingdino.")
            return False

        def verify_dll(install_local=True):
            try:
                from groundingdino import _C
                logger.info("GroundingDINO dynamic library have been successfully built.")
                return True
            except Exception:
                import traceback
                traceback.print_exc()
                def run_pip_uninstall(command, desc=None):
                    from launch import python, run
                    default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")
                    return run(f'"{python}" -m pip uninstall -y {command}', desc=f"Uninstalling {desc}", errdesc=f"Couldn't uninstall {desc}", live=default_command_live)
                if install_local:
                    logger.warn(f"Failed to build dymanic library. Will uninstall GroundingDINO from pip and fall back to local GroundingDINO this time. {self.dino_install_issue_text}")
                    run_pip_uninstall(
                        f"groundingdino",
                        f"sd-webui-segment-anything requirement: groundingdino")
                else:
                    logger.warn(f"Failed to build dymanic library. Will uninstall GroundingDINO from pip and re-try installing from GitHub source code. {self.dino_install_issue_text}")
                    run_pip_uninstall(
                        f"uninstall groundingdino",
                        f"sd-webui-segment-anything requirement: groundingdino")
                return False

        import launch
        if launch.is_installed("groundingdino"):
            logger.info("Found GroundingDINO in pip. Verifying if dynamic library build success.")
            if verify_dll(install_local=False):
                return True
        try:
            launch.run_pip(
                f"install git+https://github.com/IDEA-Research/GroundingDINO",
                f"sd-webui-segment-anything requirement: groundingdino")
            logger.info("GroundingDINO install success. Verifying if dynamic library build success.")
            return verify_dll()
        except Exception:
            import traceback
            traceback.print_exc()
            logger.warn(f"GroundingDINO install failed. Will fall back to local groundingdino this time. {self.dino_install_issue_text}")
            return False


    def _load_dino_model(self, dino_checkpoint: str, dino_install_success: bool) -> torch.nn.Module:
        logger.info(f"Initializing GroundingDINO {dino_checkpoint}")
        if self.dino_model is None or dino_checkpoint != self.dino_model_type:
            self.clear()
            if dino_install_success:
                from groundingdino.models import build_model
                from groundingdino.util.slconfig import SLConfig
                from groundingdino.util.utils import clean_state_dict
            else:
                from local_groundingdino.models import build_model
                from local_groundingdino.util.slconfig import SLConfig
                from local_groundingdino.util.utils import clean_state_dict
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


    def _load_dino_image(self, image_pil: Image.Image, dino_install_success: bool) -> torch.Tensor:
        if dino_install_success:
            import groundingdino.datasets.transforms as T
        else:
            from local_groundingdino.datasets import transforms as T
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image


    def _get_grounding_output(self, image: torch.Tensor, caption: str, box_threshold: float):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(device)
        with torch.no_grad():
            outputs = self.dino_model(image[None], captions=[caption])
        if shared.cmd_opts.lowvram:
            self.dino_model.cpu()
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        return boxes_filt.cpu()


    def dino_predict(self, input_image: Image.Image, dino_model_name: str, text_prompt: str, box_threshold: float) -> Tuple[torch.Tensor, bool]:
        install_success = self._install_goundingdino()
        logger.info("Running GroundingDINO Inference")
        dino_image = self._load_dino_image(input_image.convert("RGB"), install_success)
        self._load_dino_model(dino_model_name, install_success)
        using_groundingdino = install_success or shared.opts.data.get("sam_use_local_groundingdino", False)

        boxes_filt = self._get_grounding_output(
            dino_image, text_prompt, box_threshold
        )

        H, W = input_image.size[1], input_image.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        gc.collect()
        torch_gc()
        return boxes_filt, using_groundingdino


    def clear(self) -> None:
        del self.dino_model
        self.dino_model = None


    def unload_model(self) -> None:
        if self.dino_model is not None:
            self.dino_model.cpu()
