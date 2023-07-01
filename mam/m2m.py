# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import os
import torch
from torch.nn import Module, functional as F

import m2ms
from modules.devices import get_device_for
from scripts.sam_state import sam_extension_dir
from scripts.sam_log import logger
from mam import utils

class SamM2M(Module):

    def __init__(self):
        super(SamM2M, self).__init__()
        self.m2m = None
        self.m2m_device = get_device_for("sam")


    def load_m2m(self, m2m='sam_decoder_deep', ckpt_path: str=None):
        if m2m not in m2ms.__all__:
            raise NotImplementedError(f"Unknown M2M {m2m}")
        self.m2m: Module = m2ms.__dict__[m2m](nc=256)
        if ckpt_path is None:
            ckpt_path = os.path.join(sam_extension_dir, 'models/mam')
        try:
            logger.info(f"Loading mam from path: {ckpt_path}/mam.pth to device: {self.m2m_device}")
            state_dict = torch.load(os.path.join(ckpt_path, 'mam.pth'), map_location=self.m2m_device)
        except:
            mam_url = "https://huggingface.co/conrevo/SAM4WebUI-Extension-Models/resolve/main/mam.pth"
            logger.info(f"Loading mam from url: {mam_url} to path: {ckpt_path}, device: {self.m2m_device}")
            state_dict = torch.hub.load_state_dict_from_url(mam_url, ckpt_path, self.m2m_device)
        self.m2m.load_state_dict(state_dict)
        self.m2m.eval()


    def forward(self, features: torch.Tensor, image: torch.Tensor, 
                low_res_masks: torch.Tensor, masks: torch.Tensor, 
                ori_shape: torch.Tensor, pad_shape: torch.Tensor, guidance_mode: str):
        self.m2m.to(self.m2m_device)
        pred = self.m2m(features, image, low_res_masks)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        alpha_pred_os8 = alpha_pred_os8[..., : pad_shape[0], : pad_shape[1]]
        alpha_pred_os4 = alpha_pred_os4[..., : pad_shape[0], : pad_shape[1]]
        alpha_pred_os1 = alpha_pred_os1[..., : pad_shape[0], : pad_shape[1]]

        alpha_pred_os8 = F.interpolate(alpha_pred_os8, ori_shape, mode="bilinear", align_corners=False)
        alpha_pred_os4 = F.interpolate(alpha_pred_os4, ori_shape, mode="bilinear", align_corners=False)
        alpha_pred_os1 = F.interpolate(alpha_pred_os1, ori_shape, mode="bilinear", align_corners=False)
        
        if guidance_mode == 'mask':
            weight_os8 = utils.get_unknown_tensor_from_mask_oneside(masks, rand_width=10, train_mode=False)
            masks[weight_os8 > 0] = alpha_pred_os8[weight_os8 > 0]
            alpha_pred = masks.clone().detach()
        else:
            weight_os8 = utils.get_unknown_box_from_mask(masks)
            alpha_pred_os8[weight_os8>0] = masks[weight_os8 > 0]
            alpha_pred = alpha_pred_os8.clone().detach()

        weight_os4 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=20, train_mode=False)
        alpha_pred[weight_os4 > 0] = alpha_pred_os4[weight_os4 > 0]
        
        weight_os1 = utils.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=10, train_mode=False)
        alpha_pred[weight_os1 > 0] = alpha_pred_os1[weight_os1 > 0]
       
        alpha_pred = alpha_pred[0][0].cpu().numpy()
        return alpha_pred


    def clear(self):
        del self.m2m
        self.m2m = None


    def unload_model(self):
        if self.m2m is not None:
            self.m2m.cpu()
