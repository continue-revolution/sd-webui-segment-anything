# Segment Anything for Stable Diffusion WebUI

This extension aim for helping [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) users to use [segment anything](https://github.com/facebookresearch/segment-anything/) to do stable diffusion inpainting.

## News

- `2023/04/12`: [Feature] Mask expansion enabled. Thanks [@jordan-barrett-jm](https://github.com/jordan-barrett-jm) for your great contribution!
- `2023/04/13`: [Feature] Preview version of [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) support released in [GroundingDINO](https://github.com/continue-revolution/sd-webui-segment-anything/tree/GroundingDINO) branch. The full featured GroundingDINO support is still under development.

## Plan

Thanks for suggestions from [GitHub Issues](https://github.com/continue-revolution/sd-webui-segment-anything/issues), [reddit](https://www.reddit.com/r/StableDiffusion/comments/12hkdy8/sd_webui_segment_everything/) and [bilibili](https://www.bilibili.com/video/BV1Tg4y1u73r/) to make this extension better.

- [x] Support mask expansion
- [ ] [Developing] Support text->detection->segmentation from [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/), with batch process to support LoRA/LyCORIS training, and merge to master branch
- [ ] [Developing] Support API as mentioned in #15
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
- Upload your image and **add prompts on the image (MUST)**. Left click for positive prompt (black dot), right click for negative prompt (red dot), left click any dot again to cancel the prompt. If you forgot to add prompts, there will be exceptions on your terminal.
- Click `Preview Segmentation` button
- Choose your favorite segmentation and check `Copy to Inpaint Upload`
- Optionally check `Expand Mask` and specify the amount, then click `Update Mask`
- Switch to `Inpaint upload`. There is no need to upload another image or mask, just leave them blank. Write your prompt, configurate and click `Generate`.

### Demo

https://user-images.githubusercontent.com/63914308/230916163-af661008-5a50-496e-8b79-8be7f193f9e9.mp4


## Contribute

Disclaimer: I have not thoroughly tested this extension, so there might be bugs. Bear with me while I'm fixing them :)

If you encounter a bug, please submit a issue. Please at least provide your WebUI version, your extension version, your browser version, errors on your browser console log if there is any, error on your terminal log if there is any, to make sure that I can find a solution faster.

I welcome any contribution. Please submit a pull request if you want to contribute
