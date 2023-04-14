# Segment Anything for Stable Diffusion WebUI

This extension aim for helping [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) users to use [segment anything](https://github.com/facebookresearch/segment-anything/) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to do stable diffusion inpainting.

## News

- `2023/04/12`: [Feature] Mask expansion enabled. Thanks [@jordan-barrett-jm](https://github.com/jordan-barrett-jm) for your great contribution!
- `2023/04/14`: [Feature] [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) support with full feature released in master branch! Check it out and use text prompt to automatically generate masks! Also use `Batch Process` tab to get LoRA/LyCORIS training set! Note that when you firstly initiate WebUI you may need to wait some time for GroundingDINO to be built. Also make sure that you have access to GitHub on your terminal, otherwise you may need to install manually.

## Plan

Thanks for suggestions from [GitHub Issues](https://github.com/continue-revolution/sd-webui-segment-anything/issues), [reddit](https://www.reddit.com/r/StableDiffusion/comments/12hkdy8/sd_webui_segment_everything/) and [bilibili](https://www.bilibili.com/video/BV1Tg4y1u73r/) to make this extension better.

- [ ] [Developing] Support API as mentioned in [#15](https://github.com/continue-revolution/sd-webui-segment-anything/issues/15)
- [ ] Support color inpainting as mentioned in [#21](https://github.com/continue-revolution/sd-webui-segment-anything/issues/22)
- [ ] Support automatic mask generation for hierarchical image segmentation and SD animation
- [ ] Support semantic segmentation for batch process, ControlNet segmentation and SD animation
- [ ] Connect to [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) inpainting and segmentation
- [ ] Support WebUI older commits (e.g. `a9fed7c364061ae6efb37f797b6b522cb3cf7aa2`)

Not all plans may ultimately be implemented. Some ideas might not work and be abandoned. Support for old commits has low priority, so I would encourage you to update your WebUI as soon as you can.

## Update your WebUI version

If you are unable to add dot, observe [list index out of range](https://github.com/continue-revolution/sd-webui-segment-anything/issues/6) error on your terminal, or any other error, the most probable reason is that your WebUI is outdated (such as you are using this commitment: `a9fed7c364061ae6efb37f797b6b522cb3cf7aa2`).

In most cases, updating your WebUI can solve your problem. Before you submit your issue and before I release support for some old version of WebUI, I ask that you firstly check your version of your WebUI.

## How to use

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
- Choose your SAM model, GroundingDINO model, text prompt, box threshold and mask expansion amount. Enter the source and destination directories of your images. The source directory should only contain images.
- `Output per image` gives you a choice on configurating the number of masks per bounding box. I would highly recommend choosing 3, since some mask might be wierd.
- `save mask` gives you a choice to save the black & white mask and `Save original image with mask and bounding box` enables you to save image+mask+bounding_box.
- Click `Start batch process` and wait. If you see "Done" below this button, you are all set.

### Demo
Point prompts demo
https://user-images.githubusercontent.com/63914308/230916163-af661008-5a50-496e-8b79-8be7f193f9e9.mp4

GroundingDINO demo
https://user-images.githubusercontent.com/63914308/232157480-757f6e70-673a-4023-b4ca-df074ed30436.mp4

Batch process image demo
![configuration](https://user-images.githubusercontent.com/63914308/232157562-2f3cc9cc-310c-4b8b-89ba-216d2e014ca2.jpg)

![input image](https://user-images.githubusercontent.com/63914308/232157678-fcaaf6b6-1805-49fd-91fa-8a722cc01c8a.png)

![output image](https://user-images.githubusercontent.com/63914308/232157721-2754ccf2-b341-4b24-95f2-b75ac5b4fcd2.png)

![output mask](https://user-images.githubusercontent.com/63914308/232157975-05de0b23-1225-4187-89b1-032c731b46eb.png)

![output blend](https://user-images.githubusercontent.com/63914308/232158026-d661cfe0-d7fa-4c3b-987b-58c9c1d3686c.png)

## Contribute

Disclaimer: I have not thoroughly tested this extension, so there might be bugs. Bear with me while I'm fixing them :)

If you encounter a bug, please submit a issue. Please at least provide your WebUI version, your extension version, your browser version, errors on your browser console log if there is any, error on your terminal log if there is any, to make sure that I can find a solution faster.

I welcome any contribution. Please submit a pull request if you want to contribute
