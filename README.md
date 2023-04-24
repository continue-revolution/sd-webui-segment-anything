# Segment Anything for Stable Diffusion WebUI

This extension aim for helping [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) users to use [segment anything](https://github.com/facebookresearch/segment-anything/) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to do stable diffusion inpainting and create LoRA/LyCORIS training set. If you want to cut out images, you are also recommended to use `Batch Process` functionality described [here](#batch-process).

## News

- `2023/04/10`: [Release] SAM extension released!
- `2023/04/12`: [Feature] Mask expansion released by [@jordan-barrett-jm](https://github.com/jordan-barrett-jm)!
- `2023/04/15`: [Feature] [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) support released!
- `2023/04/15`: [Feature] API support released by [@jordan-barrett-jm](https://github.com/jordan-barrett-jm)!
- `2023/04/18`: [Feature] [ControlNet V1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) inpainting support released! Note that you **must** update [ControlNet extension](https://github.com/Mikubill/sd-webui-controlnet) to the most up-to-date version to use it. ControlNet inpainting has far better performance compared to general-purpose models, and you do not need to download inpainting-specific models anymore.
- `2023/04/24`: [Feature] Automatic segmentation support released! This should hopefully be the final huge update. Note that some functionalities of this update requires you to have [ControlNet extension](https://github.com/Mikubill/sd-webui-controlnet) installed. This update includes support for 
    - [ControlNet V1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) semantic segmentation
    - [Edit-Anything](https://github.com/sail-sg/EditAnything) un-semantic segmentation (Not tested)
    - Image masking with categories (single image + batch process)
    - Image layout generation (single image + batch process)
    - Draw un-masked region for ControlNet inpainting on txt2img panel

## Note about GroundingDINO

We have supported GroundingDINO. It has the following functionalities:
- You can use text prompt to automatically generate bounding boxes. You can separate different category names with `.`. SAM can convert these bounding boxes to masks.
- You can use point prompts with **ONE bounding box** to generate masks
- You can go to `Batch Process` tab to cut out images and get LoRA/LyCORIS training set

However, there are some existing problems with GroundingDINO:
- GroundingDINO will be install when you firstly use GroundingDINO features, instead of when you initiate the WebUI. Make sure that your terminal can have access to GitHub. Otherwise you have to download manually.
- Downloading GroundingDINO requires your device to compile C++, which might take a long time and be problematic. I honestly can do very little about such problem. Please go to [Grounded Segment Anything Issue](https://github.com/IDEA-Research/Grounded-Segment-Anything/issues) and submit an issue there. If you submit an issue in my repository, I will redirect your issue there. Despite of this, you can still use this extension for point prompts->segmentation masks even if you cannot install GroundingDINO, don't worry.
- If you want to use point prompts, SAM can at most accept one bounding box. This extension will check if there are multiple bounding boxes. If multiple bounding boxes, this extension will disgard all point prompts; otherwise all point prompts will be effective. You may always select one bounding box you want.
- If you cannot compile `_C`, it's most probably because you did not install CUDA Toolkit. Follow steps decribed [here](https://github.com/continue-revolution/sd-webui-segment-anything/issues/32#issuecomment-1513873296). DO NOT skip steps.

For more detail, check [How to Use](#how-to-use) and [Demo](#demo).

## How to Use

Download this extension to `${sd-webui}/extensions` use whatever way you like (git clone or install from UI)

Download segment-anything model using link below to `${sd-webui}/models/sam`. **Do not change model name, otherwise this extension may fail due to a bug inside segment anything.**

To give you a reference, [vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) is 2.56GB, [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) is 1.25GB, [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) is 375MB. I myself tested vit_h on NVIDIA 3090 Ti which is good. If you encounter VRAM problem, you should switch to smaller models.

### Single Image
- Upload your image
- Optionally add point prompts on the image. Left click for positive point prompt (black dot), right click for negative point prompt (red dot), left click any dot again to cancel the prompt. You must add point prompt if you do not wish to use GroundingDINO.
- Optionally check `Enable GroundingDINO`, select GroundingDINO model you want, write text prompt (separate different categories with `.`) and pick a box threshold (Too high threshold with result in no bounding box). You must write text prompt if you do not wish to use point prompts. Note that GroundingDINO models will be automatically downloaded from [HuggingFace](https://huggingface.co/ShilongLiu/GroundingDINO/tree/main). If your terminal cannot visit HuggingFace, please manually download the model and put it under `${sd-webui-sam}/models/grounding-dino`.
- Optionally enable previewing GroundingDINO bounding box and click `Generate bounding box`. You must write text prompt to preview bounding box. After you see the boxes with number marked on the left corner, uncheck all the boxes you do not want. If you uncheck all boxes, you will have to add point prompts to generate masks.
- Click `Preview Segmentation` button. Due to the limitation of SAM, if there are multiple bounding boxes, your point prompts will not take effect when generating masks.
- Choose your favorite segmentation.
- Optionally check `Expand Mask` and specify the amount, then click `Update Mask`.

#### txt2img
- You may only copy image and mask to ControlNet inpainting. 
- You may check `ControlNet inpaint not masked`, where the mask will be inverted.
- You should select the correct ControlNet index where you are using inpainting, if you wish to use multi-ControlNet. 

#### img2img
- Update your ControlNet (very important, see [this pull request](https://github.com/Mikubill/sd-webui-controlnet/pull/859)) and check `Allow other script to control this extension` on your settings of ControlNet.
- Check `Copy to Inpaint Uploade & ControlNet Inpainting`. There is no need to select ControlNet index.
- Configurate ControlNet panel. Click `Enable`, preprocessor choose `inpaint_global_harmonious`, model choose `control_v11p_sd15_inpaint [ebff9138]`. There is no need to upload image to the ControlNet inpainting panel.
- Click `Switch to Inpaint Upload` button. There is no need to upload another image or mask, just leave them blank. Write your prompts, configurate A1111 panel and click `Generate`.

### Batch Process
- Choose your SAM model, GroundingDINO model, text prompt, box threshold and mask expansion amount. Enter the source and destination directories of your images.
- Choose `Output per image` to configurate the number of masks per bounding box. I highly recommend 3, since some masks might be wierd.
- Click/unclick several checkboxes to configurate the images you want to save.
- Click `Start batch process` and wait. If you see "Done" below this button, you are all set.

### Auto SAM

#### ControlNet

#### Image Layout

#### Mask by Category

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

url = "http://127.0.0.1:7860/sam-webui/image-mask"

def image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    return img_base64

payload = {
    "image": image_to_base64("IMAGE_FILE_PATH"),
    "prompt": "TEXT PROMPT",
    "box_threshold": 0.3,
    "padding": 30 #Optional param to pad masks
}
res = requests.post(url, json=payload)

for dct in res.json():
    image_data = base64.b64decode(dct['image'])
    image = Image.open(BytesIO(image_data))
    image.show()
```

## FAQ

Thanks for suggestions from [GitHub Issues](https://github.com/continue-revolution/sd-webui-segment-anything/issues), [reddit](https://www.reddit.com/r/StableDiffusion/comments/12hkdy8/sd_webui_segment_everything/) and [bilibili](https://www.bilibili.com/video/BV1Tg4y1u73r/) to make this extension better.

Q: Do you plan to support old commits of Stable Diffusion WebUI, like `a9fed7c364061ae6efb37f797b6b522cb3cf7aa2`?

A: No, because the current version of WebUI is stable, and some integrated package authors have also updated their packages (for example, if you are using the package from [@Akegarasu](https://github.com/Akegarasu), i.e. 秋叶整合包, it has already been updated according to [this video](https://www.bilibili.com/video/BV1iM4y1y7oA)). Also, supporting different versions will be a huge time commitment, during which I can create many more features. Please update your WebUI and it is safe to use.

Q: I cannot install GroundingDINO. What should I do?

A: The most common reason is that `_C` is unable to compile. Please install cuda toolkit from NVIDIA page. Make sure that the version matches your PyTorch version.

Q: Do you plan to support color inpainting?

A: Not at this moment, because gradio wierdly enlarge the input image which makes the user experience extremely bad. If you copy image to color inpainting panel, your browser will be significantly slower, or even crash. I have already implemented this feature, though, but I just made it invisible.

Q: Do you plan to support [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything), [Edit-Anything](https://github.com/sail-sg/EditAnything), or any other popular SD+SAM repositories?

A: [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) and [Edit-Anything](https://github.com/sail-sg/EditAnything) have been supported. For Inpaint-Anything, you may check [this issue](https://github.com/continue-revolution/sd-webui-segment-anything/issues/60) for how to use. For Edit-Anything, this extension has **in-theory** supported, but since they only published diffusers models which probably only work for SD 2.x, I am unable to test at this moment. I will update once they release models in lllyasviel format. I am always open to support any other interesting applications, but I think at this moment, all such applications should have been supported.

Q: Do you plan to continue updating this extension?

A: Yes, but this extension has moved into maintenance phase. I don't think there will be huge updates in the foreseeable future. Despite of this, I will continue to deal with issues, and monitor new research works to see if they are worth supporting. I welcome any community contribution and any feature requests.

Q: I have a job opportunity, are you interested?

A: Yes, please send me an email if you are interested.

Q: I want to sponsor you, how can I do that?

A: Please go to [sponsor](#sponsor) section and scan the corresponding QR code.


## Contribute

Disclaimer: I have not thoroughly tested this extension, so there might be bugs. Bear with me while I'm fixing them :)

If you encounter a bug, please submit a issue. Please at least provide your WebUI version, your extension version, your browser version, errors on your browser console log if there is any, error on your terminal log if there is any, to make sure that I can find a solution faster.

I welcome any contribution. Please submit a pull request if you want to contribute

## Star History

Give me a star if you like this extension!

[![Star History Chart](https://api.star-history.com/svg?repos=continue-revolution/sd-webui-segment-anything&type=Date)](https://star-history.com/#continue-revolution/sd-webui-segment-anything&Date)

## Sponsor

You can sponsor me via WeChat or Alipay.

| WeChat | Alipay |
| --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) |
