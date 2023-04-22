from typing import Tuple, List, Dict
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from modules import shared


def max_cn_num():
    if shared.opts.data is None:
        return 0
    return int(shared.opts.data.get('control_net_max_models_num', 0))


class SAMInpaintUnit:
    def __init__(self, args: Tuple, is_img2img=False):
        self.is_img2img = is_img2img

        self.inpaint_upload_enable: bool = False
        self.cnet_inpaint_invert: bool = False
        self.cnet_inpaint_idx: int = 0
        self.input_image = None
        self.output_mask_gallery: List[Dict] = None
        self.output_chosen_mask: int = 0
        self.dilation_checkbox: bool = False
        self.dilation_output_gallery: List[Dict] = None
        self.sketch_checkbox: bool = False
        self.inpaint_color_sketch = None
        self.inpaint_mask_alpha: int = 0
        self.init_sam_single_image_process(args)

    
    def init_sam_single_image_process(self, args):
        self.inpaint_upload_enable      = args[0]
        self.cnet_inpaint_invert        = args[1]
        self.cnet_inpaint_idx           = args[2]
        self.input_image                = args[3]
        self.output_mask_gallery        = args[4]
        self.output_chosen_mask         = args[5]
        self.dilation_checkbox          = args[6]
        self.dilation_output_gallery    = args[7]
        if self.is_img2img:
            self.sketch_checkbox        = args[8]
            self.inpaint_color_sketch   = args[9]
            self.inpaint_mask_alpha     = args[10]


    def get_input_and_mask(self, mask_blur):
        image, mask = None, None
        if self.inpaint_upload_enable and self.input_image is not None and self.output_mask_gallery is not None:
            if self.dilation_checkbox and self.dilation_output_gallery is not None:
                mask = Image.open(self.dilation_output_gallery[1]['name']).convert('L')
            elif self.output_mask_gallery is not None:
                mask = Image.open(self.output_mask_gallery[self.output_chosen_mask + 3]['name']).convert('L')
            if mask is not None and self.cnet_inpaint_invert:
                mask = ImageOps.invert(mask)
            if self.is_img2img and self.sketch_checkbox and self.inpaint_color_sketch is not None and mask is not None:
                alpha = np.expand_dims(np.array(mask) / 255, axis=-1)
                image = np.uint8(np.array(self.inpaint_color_sketch) * alpha + np.array(self.input_image) * (1 - alpha))
                mask = ImageEnhance.Brightness(mask).enhance(1 - self.inpaint_mask_alpha / 100)
                blur = ImageFilter.GaussianBlur(mask_blur)
                image = Image.composite(image.filter(blur), self.input_image, mask.filter(blur)).convert("RGB")
            else:
                image = self.input_image
        return image, mask



class SAMProcessUnit:
    def __init__(self, args: Tuple, is_img2img=False):
        self.is_img2img = is_img2img

        if is_img2img:
            sam_inpaint_args = args[:11]
            args = args[11:]
        else:
            sam_inpaint_args = args[:8]
            args = args[8:]
        self.sam_inpaint_unit = SAMInpaintUnit(sam_inpaint_args, is_img2img)
        
        self.cnet_seg_output_gallery: List[Dict] = None
        self.cnet_seg_enable_copy: bool = False
        self.cnet_seg_idx: int = 0
        args = self.init_cnet_seg_process(args)

        self.crop_inpaint_unit = SAMInpaintUnit(args, is_img2img)
        
        
    def init_cnet_seg_process(self, args):
        csp_args    = args[:3]
        self.cnet_seg_output_gallery    = csp_args[0]
        self.cnet_seg_enable_copy       = csp_args[1]
        self.cnet_seg_idx               = csp_args[2]
        return args[3:]

    
    def set_process_attributes(self, p):
        inpaint_mask_blur = getattr(p, "mask_blur", 0)
        inpaint_image, inpaint_mask = self.sam_inpaint_unit.get_input_and_mask(inpaint_mask_blur)
        inpaint_cn_num = self.sam_inpaint_unit.cnet_inpaint_idx
        if inpaint_image is None:
            inpaint_image, inpaint_mask = self.crop_inpaint_unit.get_input_and_mask(inpaint_mask_blur)
            inpaint_cn_num = self.crop_inpaint_unit.cnet_inpaint_idx
        if inpaint_image is not None and inpaint_mask is not None:
            if self.is_img2img:
                p.init_images = [inpaint_image]
                p.image_mask = inpaint_mask
            else:
                self.set_p_value(p, 'control_net_input_image', inpaint_cn_num, 
                                {"image": inpaint_image, "mask": inpaint_mask.convert("L")})
        
        if self.cnet_seg_enable_copy and self.cnet_seg_output_gallery is not None:
            self.set_p_value(p, 'control_net_input_image', self.cnet_seg_idx, 
                             Image.fromarray(self.cnet_seg_output_gallery[1]['name']))


    def set_p_value(self, p, attr: str, idx: int, v):
        value = getattr(p, attr, None)
        if isinstance(value, list):
            value[idx] = v
        else:
            # if value is None, ControlNet uses default value
            value = [value] * max_cn_num()
            value[idx] = v
        setattr(p, attr, value)
