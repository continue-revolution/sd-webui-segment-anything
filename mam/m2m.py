# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import os
import torch
from torch.nn import Module

import m2ms
from modules.devices import get_device_for

class SamM2M(Module):
    def __init__(self, m2m='sam_decoder_deep', ckpt_path=None, device=None):
        super(SamM2M, self).__init__()
        if m2m not in m2ms.__all__:
            raise NotImplementedError(f"Unknown M2M {m2m}")
        self.m2m: Module = m2ms.__dict__[m2m](nc=256)
        if ckpt_path is None:
            ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/mam')
        try:
            state_dict = torch.load(os.path.join(ckpt_path, 'mam.pth'), map_location=device)
        except:
            state_dict = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/conrevo/Matting-Anything-diff/resolve/main/mam.pth", ckpt_path, device)
        self.m2m.load_state_dict(state_dict)
        self.m2m.eval()
        self.m2m_device = get_device_for("sam") if device is None else device


    def forward(self, feas, image, masks):
        self.m2m.to(self.m2m_device)
        pred = self.m2m(feas, image, masks)
        return pred


    def unload_model(self):
        self.m2m.to('cpu')
