# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import os
import torch
from torch.nn import Module

import m2ms
from modules.devices import get_device_for
from scripts.sam_state import sam_extension_dir
from scripts.sam_log import logger

class SamM2M(Module):

    def __init__(self):
        super(SamM2M, self).__init__()
        self.m2m_device = get_device_for("sam")


    def load_m2m(self, m2m='sam_decoder_deep', ckpt_path=None):
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


    def forward(self, feas, image, masks):
        self.m2m.to(self.m2m_device)
        pred = self.m2m(feas, image, masks)
        return pred


    def clear(self):
        del self.m2m
        self.m2m = None


    def unload_model(self):
        if self.m2m is not None:
            self.m2m.cpu()
