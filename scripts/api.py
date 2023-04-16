from fastapi import FastAPI, Body
from io import BytesIO
import base64
from pydantic import BaseModel
from typing import Any, Optional
import asyncio
import gradio as gr
import os
from scripts.sam import init_sam_model, dilate_mask, sam_predict, sam_model_list
from scripts.dino import dino_model_list
from PIL import Image, ImageChops
import base64

def sam_api(_: gr.Blocks, app: FastAPI):    
    @app.get("/sam-webui/heartbeat")
    async def heartbeat():
        return {            
            "msg": "Success!"
        }
    
    class MaskRequest(BaseModel):
        image: str #base64 string containing image
        prompt: str
        box_threshold: float
        padding: Optional[int] = 0
        

    def pil_image_to_base64(img: Image.Image) -> str:
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64

    @app.post("/sam-webui/image-mask")
    async def process_image(payload: MaskRequest = Body(...)) -> Any:
        sam_model_name = sam_model_list[0] if len(sam_model_list) > 0 else None
        dino_model_name = dino_model_list[0] if len(dino_model_list) > 0 else None
        # Decode the base64 image string
        img_b64 = base64.b64decode(payload.image)
        input_img = Image.open(BytesIO(img_b64))
        #Run DINO and SAM inference to get masks back
        masks = sam_predict(sam_model_name,
                            input_img,
                            [],
                            [],
                            True,
                            dino_model_name,
                            payload.prompt,
                            payload.box_threshold,
                            None,
                            None,
                            gui=False)[0]
        if payload.padding:
            masks = [dilate_mask(mask, payload.padding)[0] for mask in masks]
        # Convert the final PIL image to a base64 string
        response = [{"image": pil_image_to_base64(mask)} for mask in masks]

        return response

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(sam_api)
except:
    print("SAM Web UI API failed to initialize")