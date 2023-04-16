# Segment Anything for Stable Diffusion WebUI

This extension aim for helping [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) users to use [segment anything](https://github.com/facebookresearch/segment-anything/) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to do stable diffusion inpainting and create LoRA/LyCORIS training set. If you want to cut out images, you are also recommended to use `Batch Process` functionality described [here](#batch-process).

## News

- `2023/04/12`: [Feature] Mask expansion released by [@jordan-barrett-jm](https://github.com/jordan-barrett-jm)!
- `2023/04/15`: [Feature] [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) support released! Check [Note about GroundingDINO](https://github.com/continue-revolution/sd-webui-segment-anything#note-about-groundingdino), [How to Use](#how-to-use) and [Demo](#demo) for more detail.
- `2023/04/15`: [Feature] API support released by [@jordan-barrett-jm](https://github.com/jordan-barrett-jm)! Check [API Support](#api-support) for more detail.

## Plan

Thanks for suggestions from [GitHub Issues](https://github.com/continue-revolution/sd-webui-segment-anything/issues), [reddit](https://www.reddit.com/r/StableDiffusion/comments/12hkdy8/sd_webui_segment_everything/) and [bilibili](https://www.bilibili.com/video/BV1Tg4y1u73r/) to make this extension better.

- [ ] Support color inpainting as mentioned in [#21](https://github.com/continue-revolution/sd-webui-segment-anything/issues/22)
- [ ] Support automatic mask generation for hierarchical image segmentation and SD animation
- [ ] Support semantic segmentation for batch process, ControlNet segmentation and SD animation
- [ ] Connect to [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) inpainting and segmentation
- [ ] Support WebUI older commits (e.g. `a9fed7c364061ae6efb37f797b6b522cb3cf7aa2`)

Not all plans may ultimately be implemented. Some ideas might not work and be abandoned. Support for old commits has low priority, so I would encourage you to update your WebUI as soon as you can.

## Update your WebUI version

If you are unable to add dot, observe [list index out of range](https://github.com/continue-revolution/sd-webui-segment-anything/issues/6) error on your terminal, or any other error, the most probable reason is that your WebUI is outdated (such as you are using this commitment: `a9fed7c364061ae6efb37f797b6b522cb3cf7aa2`).

In most cases, updating your WebUI can solve your problem. Before you submit your issue and before I release support for some old version of WebUI, I ask that you firstly check your version of your WebUI.

## Note about GroundingDINO

We have supported GroundingDINO. It has the following functionality:
- You can use text prompt to automatically generate masks
- You can use point prompts with **ONE mask** to generate masks
- You can go to `Batch Process` tab to cut out images and get LoRA/LyCORIS training set

However, there are some existing problems with GroundingDINO:
- GroundingDINO will be install when you firstly use GroundingDINO features, instead of when you initiate the WebUI. Make sure that your terminal can have access to GitHub. Otherwise you have to download manually.
- Downloading GroundingDINO requires your device to compile C++, which might take a long time and be problematic. I honestly can do very little about such problem. Please go to [Grounded Segment Anything Issue](https://github.com/IDEA-Research/Grounded-Segment-Anything/issues) and submit an issue there. If you submit an issue in my repository, I will redirect your issue there. Despite of this, you can still use this extension for point prompts->segmentation masks even if you cannot install GroundingDINO, don't worry.
- If you want to use point prompts, SAM can at most accept one mask. In this case, my script will check if there are multiple masks. If multiple masks, my script will disgard all point prompts; otherwise all point prompts will be effective. You may always select one mask you want.

For more detail, check [How to Use](#how-to-use) and [Demo](#demo).

## How to Use

### Step 1:

Download this extension to `${sd-webui}/extensions` use whatever way you like (git clone or install from UI)

### Step 2:

Download segment-anything model from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) to `${sd-webui}/models/sam`. **Do not change model name, otherwise this extension may fail due to a bug inside segment anything.**

To give you a reference, [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) is 2.56GB, [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) is 1.25GB, [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) is 375MB. I myself tested vit_h on NVIDIA 3090 Ti which is good. If you encounter VRAM problem, you should switch to smaller models.

### Step 3:

- Launch webui and switch to img2img mode.

#### Single Image
- Upload your image
- Optionally add point prompts on the image. Left click for positive point prompt (black dot), right click for negative point prompt (red dot), left click any dot again to cancel the prompt. You must add point prompt if you do not wish to use GroundingDINO.
- Optionally check `Enable GroundingDINO`, select GroundingDINO model you want, write text prompt and pick a box threshold. You must write text prompt if you do not wish to use point prompts. Note that GroundingDINO models will be automatically downloaded from [HuggingFace](https://huggingface.co/ShilongLiu/GroundingDINO/tree/main). If your terminal cannot visit HuggingFace, please manually download the model and put it under `${sd-webui-sam}/models/grounding-dino`.
- Optionally enable previewing GroundingDINO bounding box and click `Generate bounding box`. You must write text prompt to preview bounding box. After you see the boxes with number marked on the left corner, uncheck all the boxes you do not want. If you uncheck all boxes, you will have to add point prompts to generate masks.
- Click `Preview Segmentation` button. Due to the limitation of SAM, if there are multiple bounding boxes, your point prompts will not take effect when generating masks.
- Choose your favorite segmentation and check `Copy to Inpaint Upload`
- Optionally check `Expand Mask` and specify the amount, then click `Update Mask`
- Click `Switch to Inpaint Upload` button. There is no need to upload another image or mask, just leave them blank. Write your prompt, configurate and click `Generate`.

#### Batch Process
- Choose your SAM model, GroundingDINO model, text prompt, box threshold and mask expansion amount. Enter the source and destination directories of your images. **The source directory should only contain images**.
- `Output per image` gives you a choice on configurating the number of masks per bounding box. I would highly recommend choosing 3, since some mask might be wierd.
- `save mask` gives you a choice to save the black & white mask and `Save original image with mask and bounding box` enables you to save image+mask+bounding_box.
- Click `Start batch process` and wait. If you see "Done" below this button, you are all set.

## Demo
Point prompts demo

https://user-images.githubusercontent.com/63914308/230916163-af661008-5a50-496e-8b79-8be7f193f9e9.mp4

GroundingDINO demo

https://user-images.githubusercontent.com/63914308/232157480-757f6e70-673a-4023-b4ca-df074ed30436.mp4

Batch process image demo

![Configuration Image](https://user-images.githubusercontent.com/63914308/232157562-2f3cc9cc-310c-4b8b-89ba-216d2e014ca2.jpg) 

| Input Image | Output Image | Output Mask | Output Blend |
| --- | --- | --- | --- |
| ![Input Image](https://user-images.githubusercontent.com/63914308/232157678-fcaaf6b6-1805-49fd-91fa-8a722cc01c8a.png) | ![Output Image](https://user-images.githubusercontent.com/63914308/232157721-2754ccf2-b341-4b24-95f2-b75ac5b4fcd2.png) | ![Output Mask](https://user-images.githubusercontent.com/63914308/232157975-05de0b23-1225-4187-89b1-032c731b46eb.png) | ![Output Blend](https://user-images.githubusercontent.com/63914308/232158575-228f687c-8045-4079-bcf5-5a4dd0c8d7bd.png)

## API Support

### API Usage

We have added an API endpoint to allow for automated workflows.

The API utilizes both Segment Anything and GroundingDINO to return masks of all instances of whatever object is specified in the text prompt.

This is an extension of the existing [Stable Diffusion Web UI API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API).

There are 2 endpoints exposed
- GET sam-webui/heartbeat
- POST /sam-webui/image-mask

The heartbeat endpoint can be used to ensure that the API is up.

The image-mask endpoint accepts a payload that includes your base64-encoded image.

Below is an example of how to interface with the API using requests.

### API Example

```
import base64
import requests
from PIL import Image
from io import BytesIO

def image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    return img_base64

payload = {
    "image": image_to_base64("IMAGE_FILE_PATH"),
    "prompt": "TEXT PROMPT",
    "box_threshold": 0.3
}
res = requests.post(url, json=payload)

for dct in res.json():
    image_data = base64.b64decode(dct['image'])
    image = Image.open(BytesIO(image_data))
    image.show()
```

## Contribute

Disclaimer: I have not thoroughly tested this extension, so there might be bugs. Bear with me while I'm fixing them :)

If you encounter a bug, please submit a issue. Please at least provide your WebUI version, your extension version, your browser version, errors on your browser console log if there is any, error on your terminal log if there is any, to make sure that I can find a solution faster.

I welcome any contribution. Please submit a pull request if you want to contribute

## Star History

Give me a star if you like this extension!

[![Star History Chart](https://api.star-history.com/svg?repos=continue-revolution/sd-webui-segment-anything&type=Date)](https://star-history.com/#continue-revolution/sd-webui-segment-anything&Date)

