from typing import Tuple, List
import os
import numpy as np
import cv2
import copy
from scipy.ndimage import binary_dilation
from PIL import Image


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
