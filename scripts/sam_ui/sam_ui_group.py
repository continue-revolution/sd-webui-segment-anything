import gradio as gr
import numpy as np
from PIL import Image
from modules import shared, script_callbacks
from modules.ui import gr_show
from scripts import (
    global_var,
    dino_unit
)

dino_model = dino_unit.DinoUnit()
class ToolButton(gr.Button, gr.components.FormComponent):
  """Small button with single emoji as text, fits inside gradio forms"""

  def __init__(self, **kwargs):
    super().__init__(variant="tool", **kwargs)

  def get_block_name(self):
    return "button"


class SamUiGroup:
    refresh_symbol = "\U0001f504"  # 🔄
    tooltips = {
        '🔄': 'Refresh',
    }
    
    def __init__(self) -> None:
        self.sam_model = None
        self.sam_use_cpu = None
        self.input_image = None
        self.sam_remove_dots = None
        self.sam_dummy_component = None
        self.dino_checkbox = None
        pass

    def render(self, is_img2img) -> None:
        sam_model_list = []
        tab_prefix = ("img2img" if is_img2img else "txt2img") + "_sam_"
        with gr.Accordion('Segment Anything', open=False):
            with gr.Row():
                with gr.Column(scale=10):
                    with gr.Row():
                        # FIXME
                        sam_models_keys = list(global_var.sam_models.keys())
                        self.sam_model = gr.Dropdown(
                            sam_models_keys,
                            label="SAM Model",
                            elem_id=f"sam_model_dropdown",
                            value=sam_models_keys[0] if len(sam_models_keys) > 0 else None
                        )
                        self.refresh_models = ToolButton(
                            value=SamUiGroup.refresh_symbol,
                            elem_id=f"sam_refresh_models",
                            tooltip=SamUiGroup.tooltips[SamUiGroup.refresh_symbol],
                        )

                        # sam_refresh_models.click(refresh_sam_models, sam_model_name, sam_model_name)
                with gr.Column(scale=1):
                    self.sam_use_cpu = gr.Checkbox(value=False, label="Use CPU for SAM")
            with gr.Tabs():
                with gr.TabItem(label="Single Image"):
                    # with gr.Row():
                    gr.HTML(
                        value="<p>Left click the image to add one positive point (black dot). "
                        "Right click the image to add one negative point (red dot). Left click the point to remove it.</p>"
                    )
                    self.sam_input_image = gr.Image(
                        source="upload",
                        label="Image for Segment Anything",
                        type="pil",
                        elem_id=f"{tab_prefix}input_image",
                        image_mode="RGBA"
                    )
                    self.sam_remove_dots = gr.Button(value="Remove all point prompts")
                    self.sam_dummy_component = gr.Label(visible=False)
                    gr.HTML(
                        value="<p>GroundingDINO + Segment Anything can achieve "
                              "[text prompt]->[object detection]->[segmentation]</p>"
                    )
                    self.dino_checkbox = gr.Checkbox(
                        value=False,
                        label="Enable GroundingDINO",
                        elem_id=f"{tab_prefix}dino_enable_checkbox"
                    )
                    # FIXME(ttt-77): add dino part
                    with gr.Column(visible=False) as self.dino_column:
                        gr.HTML(
                            value="<p>Due to the limitation of Segment Anything, when there are point prompts, at most 1 box prompt will be allowed; when there are multiple box prompts, no point prompts are allowed.</p>")
                        self.dino_model_name = gr.Dropdown(
                            label="GroundingDINO Model (Auto download from huggingface)",
                            choices=dino_model.dino_model_list,
                            value=dino_model.dino_model_list[0]
                        )
                        self.dino_text_prompt = gr.Textbox(
                            placeholder="You must enter text prompts to enable groundingdino. "
                                        "Otherwise this extension will fall back to point prompts only.",
                            label="GroundingDINO Detection Prompt",
                            elem_id=f"{tab_prefix}dino_text_prompt"
                        )
                        self.dino_box_threshold = gr.Slider(
                            label="GroundingDINO Box Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.001
                        )
                        self.dino_preview_checkbox = gr.Checkbox(
                            value=False,
                            label="I want to preview GroundingDINO detection result and select the boxes I want.",
                            elem_id=f"{tab_prefix}dino_preview_checkbox"
                        )
                        with gr.Column(visible=False) as self.dino_preview:
                            self.dino_preview_boxes = gr.Image(
                                show_label=False,
                                type="pil",
                                image_mode="RGBA"
                            )
                            self.dino_preview_boxes_button = gr.Button(
                                value="Generate bounding box",
                                elem_id=f"{tab_prefix}dino_run_button"
                            )
                            self.dino_preview_boxes_selection = gr.CheckboxGroup(
                                label="Select your favorite boxes: ",
                                elem_id=f"{tab_prefix}dino_preview_boxes_selection"
                            )
                            self.dino_preview_result = gr.Text(
                                value="",
                                label="GroundingDINO preview status",
                                visible=False
                            )
                    self.sam_output_mask_gallery = gr.Gallery(
                        label = 'Segment Anything Output',
                        columns = 3
                    )
                    self.sam_submit_button = gr.Button(
                        value='Preview Segmentation',
                        elem_id=f'{tab_prefix}run_button'
                    )
                    self.sam_result = gr.Text(
                        value="",
                        label="Segment Anything status"
                    )
    
    
    def register_refresh_all_models(self):
        def refresh_all_models(*inputs):
            global_var.update_sam_models()

            dd = inputs[0]
            if dd in global_var.sam_models:
                selected = dd
            elif len(global_var.sam_models.keys()) > 0:
                selected = list(global_var.sam_models.keys())[0]
            else:
              selected = None

            return gr.Dropdown.update(
                value=selected, 
                choices=list(global_var.sam_models.keys())
            )

        self.refresh_models.click(refresh_all_models, self.sam_model, self.sam_model)
        
        
    def register_change_sam_device(self):
        self.sam_use_cpu.change(
            fn=global_var.change_sam_device, 
            inputs=[self.sam_use_cpu], 
            show_progress=False
        )


    def register_remove_dots(self):
        self.sam_remove_dots.click(
            fn=lambda _: None,
            _js="samRemoveDots",
            inputs=[self.sam_dummy_component],
            outputs=None
        )

    def register_dino_unit(self):
        def dino_predict(input_image, dino_model_name, text_prompt, box_threshold):
            if input_image is None:
                return None, gr.update(), gr.update(visible=True, value=f"GroundingDINO requires input image.")
            if text_prompt is None or text_prompt == "":
                return None, gr.update(), gr.update(visible=True, value=f"GroundingDINO requires text prompt.")
            image_np = np.array(input_image)
            boxes_filt, install_success = dino_model.dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)
            boxes_filt = boxes_filt.numpy()
            boxes_choice = [str(i) for i in range(boxes_filt.shape[0])]
            return_res = (
                Image.fromarray(dino_model.show_boxes(image_np, boxes_filt.astype(int), show_index=True)),
                gr.update(choices=boxes_choice, value=boxes_choice),
                gr.update(visible=False)
                    if install_success
                    else gr.update(visible=True,
                                   value=f"GroundingDINO installment failed. Your process automatically fall back to local groundingdino. "
                                         f"See your terminal for more detail and {dino_model.dino_install_issue_text}")
            )
            return return_res

        self.dino_preview_boxes_button.click(
            fn=dino_predict,
            _js="submit_dino",
            inputs=[self.sam_input_image, self.dino_model_name, self.dino_text_prompt, self.dino_box_threshold],
            outputs=[self.dino_preview_boxes, self.dino_preview_boxes_selection, self.dino_preview_result]
        )

        self.dino_preview_checkbox.change(
            fn=gr_show,
            inputs=[self.dino_preview_checkbox],
            outputs=[self.dino_preview],
            show_progress=False)

        self.dino_checkbox.change(
            fn=gr_show,
            inputs=[self.dino_checkbox],
            outputs=[self.dino_column],
            show_progress=False
        )

    def register_submit_button(self):
        def predict()
        self.sam_submit_button.click(
            fn=self.predict,
            _js='submit_sam',
            inputs=[sam_model_name, sam_input_image,  # SAM
                    sam_dummy_component, sam_dummy_component,  # Point prompts
                    dino_checkbox, dino_model_name, dino_text_prompt, dino_box_threshold,  # DINO prompts
                    dino_preview_checkbox, dino_preview_boxes_selection],  # DINO preview prompts
            outputs=[sam_output_mask_gallery, sam_result]
        )


    def register_callbacks(self, is_img2img: bool):
        """Register callbacks on the UI elements.

        Args:
            is_img2img: Whether ControlNet is under img2img. False when in txt2img mode.

        Returns:
            None
        """
        
        self.register_refresh_all_models()
        self.register_change_sam_device()
        self.register_remove_dots()