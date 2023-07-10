from typing import Tuple, List
import os
import numpy as np
import cv2
import copy
from PIL import Image


class ModelInfo:

    def __init__(self, model_type: str, url: str, author: str, size: str, download_info: str="auto download"):
        self.model_type = model_type
        self.url = url
        self.author = author
        self.size = size
        self.download_info = download_info


    def get_info(self, model_name: str):
        return f"{model_name} ({self.size}, {self.author}, {self.model_type}, {self.download_info})"
    

    def local_path(self, path: str):
        self.url = path
        self.download_info = "downloaded"


def show_boxes(image_np: np.ndarray, boxes: np.ndarray, color=(255, 0, 0, 255), thickness=2, show_index=False) -> np.ndarray:
    if boxes is None:
        return image_np
    image = copy.deepcopy(image_np)
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (w, h), color, thickness)
        if show_index:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(idx)
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(image, text, (x, y+textsize[1]), font, 1, color, thickness)
    return image


def show_masks(image_np: np.ndarray, masks: np.ndarray, alpha=0.5) -> np.ndarray:
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)


def dilate_mask(mask: np.ndarray, dilation_amt: int) -> Tuple[Image.Image, np.ndarray]:
    from scipy.ndimage import binary_dilation
    x, y = np.meshgrid(np.arange(dilation_amt), np.arange(dilation_amt))
    center = dilation_amt // 2
    dilation_kernel = ((x - center)**2 + (y - center)**2 <= center**2).astype(np.uint8)
    dilated_binary_img = binary_dilation(mask, dilation_kernel)
    dilated_mask = Image.fromarray(dilated_binary_img.astype(np.uint8) * 255)
    return dilated_mask, dilated_binary_img


def update_mask(mask_gallery, chosen_mask: int, dilation_amt: float, input_image: Image.Image):
    if isinstance(mask_gallery, list):
        mask_image = Image.open(mask_gallery[chosen_mask + 3]['name'])
    else:
        mask_image = mask_gallery
    binary_img = np.array(mask_image.convert('1'))
    if dilation_amt:
        mask_image, binary_img = dilate_mask(binary_img, dilation_amt)
    blended_image = Image.fromarray(show_masks(np.array(input_image), binary_img.astype(np.bool_)[None, ...]))
    matted_image = np.array(input_image)
    matted_image[~binary_img] = np.array([0, 0, 0, 0])
    return [blended_image, mask_image, Image.fromarray(matted_image)]


def create_mask_output(image_np: np.ndarray, masks: np.ndarray, boxes_filt: np.ndarray) -> List[Image.Image]:
    mask_images, masks_gallery, matted_images = [], [], []
    boxes_filt = boxes_filt.astype(int) if boxes_filt is not None else None
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        mask_images.append(Image.fromarray(blended_image))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        matted_images.append(Image.fromarray(image_np_copy))
    return mask_images + masks_gallery + matted_images


def create_mask_batch_output(
    input_image_filename: str, dest_dir: str, 
    image_np: np.ndarray, masks: np.ndarray, boxes_filt: np.ndarray, dilation_amt: float, 
    save_image: bool, save_mask: bool, save_background: bool, save_image_with_mask: bool):
    filename, ext = os.path.splitext(os.path.basename(input_image_filename))
    ext = ".png" # JPEG not compatible with RGBA
    for idx, mask in enumerate(masks):
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        merged_mask = np.any(mask, axis=0)
        if save_background:
            merged_mask = ~merged_mask
        if dilation_amt:
            _, merged_mask = dilate_mask(merged_mask, dilation_amt)
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~merged_mask] = np.array([0, 0, 0, 0])
        if save_image:
            output_image = Image.fromarray(image_np_copy)
            output_image.save(os.path.join(dest_dir, f"{filename}_{idx}_output{ext}"))
        if save_mask:
            output_mask = Image.fromarray(merged_mask)
            output_mask.save(os.path.join(dest_dir, f"{filename}_{idx}_mask{ext}"))
        if save_image_with_mask:
            output_blend = Image.fromarray(blended_image)
            output_blend.save(os.path.join(dest_dir, f"{filename}_{idx}_blend{ext}"))


def blend_image_and_seg(image: np.ndarray, seg: np.ndarray, alpha=0.5) -> Image.Image:
    image_blend = image * (1 - alpha) + np.array(seg) * alpha
    return Image.fromarray(image_blend.astype(np.uint8))


def install_pycocotools():
    # install pycocotools if needed
    from sam_utils.logger import logger
    try:
        import pycocotools.mask as maskUtils
    except:
        logger.warn("pycocotools not found, will try installing C++ based pycocotools")
        try:
            from launch import run_pip
            run_pip(f"install pycocotools", f"AutoSAM requirement: pycocotools")
            import pycocotools.mask as maskUtils
        except:
            import traceback
            traceback.print_exc()
            import sys
            if sys.platform == "win32":
                logger.warn("Unable to install pycocotools, will try installing pycocotools-windows")
                try:
                    run_pip("install pycocotools-windows", "AutoSAM requirement: pycocotools-windows")
                    import pycocotools.mask as maskUtils
                except:
                    error_msg = "Unable to install pycocotools-windows"
                    logger.error(error_msg)
                    traceback.print_exc()
                    raise RuntimeError(error_msg)
            else:
                error_msg = "Unable to install pycocotools"
                logger.error(error_msg)
                traceback.print_exc()
                raise RuntimeError(error_msg)


def install_goundingdino() -> bool:
    """Automatically install GroundingDINO.

    Returns:
        bool: False if use local GroundingDINO, True if use pip installed GroundingDINO.
    """
    from sam_utils.logger import logger
    from modules import shared
    dino_install_issue_text = "Please permanently switch to local GroundingDINO on Settings/Segment Anything or submit an issue to https://github.com/IDEA-Research/Grounded-Segment-Anything/issues."
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
                logger.warn(f"Failed to build dymanic library. Will uninstall GroundingDINO from pip and fall back to local GroundingDINO this time. {dino_install_issue_text}")
                run_pip_uninstall(
                    f"groundingdino",
                    f"sd-webui-segment-anything requirement: groundingdino")
            else:
                logger.warn(f"Failed to build dymanic library. Will uninstall GroundingDINO from pip and re-try installing from GitHub source code. {dino_install_issue_text}")
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
        logger.warn(f"GroundingDINO install failed. Will fall back to local groundingdino this time. {dino_install_issue_text}")
        return False
