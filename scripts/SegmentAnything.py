import modules.scripts as scripts
from modules import script_callbacks, shared
from scripts.sam_ui.sam_ui_group import SamUiGroup
import gradio as gr


from scripts import (
    global_var,
    sam_process
)

print(sam_process.max_cn_num())
global_var.update_sam_models()

class Script(scripts.Script):
    
    def __init__(self):
        super().__init__()
        self.name = "Segment Anything"
        self.description = "Segment anything using a segmentation model."
        self.local_groundingdino = False
        self.sam_device = "cpu"
        self.sam_model_name = None
        self.sam_model_list = []
    
    
    def title(self):
        return "SegmentAnything"
    
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        group = SamUiGroup()
        group.render(is_img2img)
        group.register_callbacks(is_img2img)
        return group


def on_ui_settings():
    section = ("segment_anything", "Segment Anything")

    shared.opts.add_option(
        "animatediff_model_path",
        shared.OptionInfo(
            None,
            "Path to save AnimateDiff motion modules",
            gr.Textbox,
            section=section,
        ),
    )
    
script_callbacks.on_ui_settings(on_ui_settings)