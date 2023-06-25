import gradio as gr

class SamUI:

    def __init__(self) -> None:
        pass

    def render_sam_model(self) -> None:
        with gr.Row():
            with gr.Column(scale=10):
                with gr.Row():
                    sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list, value=sam_model_list[0] if len(sam_model_list) > 0 else None)
                    sam_refresh_models = ToolButton(value=refresh_symbol)
                    sam_refresh_models.click(refresh_sam_models, sam_model_name, sam_model_name)
            with gr.Column(scale=1):
                sam_use_cpu = gr.Checkbox(value=False, label="Use CPU for SAM")
                def change_sam_device(use_cpu=False):
                    global sam_device
                    sam_device = "cpu" if use_cpu else device
                sam_use_cpu.change(fn=change_sam_device, inputs=[sam_use_cpu], show_progress=False)