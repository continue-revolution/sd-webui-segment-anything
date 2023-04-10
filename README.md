# Segment Everything for Stable Diffusion Webui

This extension aim for helping [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) users to use [segment everything](https://github.com/facebookresearch/segment-anything/) to do stable diffusion inpainting.

## How to use

### Step 1:

Download this extension to `${sd-webui}/extensions` use whatever way you like (git clone or install from UI)

### Step 2:

Download segment-everything model from [here](https://github.com/facebookresearch/segment-anything#model-checkpoints) to `${sd-webui-segment-everything}/models/segment-everything`

### Step 3:

- Launch webui and switch to img2img mode. 
- Upload your image, left click for positive prompt (black dot), right click for negative prompt (red dot), left click any dot again to cancel the prompt. 
- Click `Preview Segmentation` button
- Choose your favorite segmentation and check `Copy to Inpaint Upload`
- Switch to `Inpaint upload`. There is no need to upload another image or mask, just leave them blank. Write your prompt, configurate and click `Generate`.

### Demo

https://user-images.githubusercontent.com/63914308/230916163-af661008-5a50-496e-8b79-8be7f193f9e9.mp4

## Future Plan

I plan to support text->object detection->segmentation from [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/) in the near future. Stay tuned!

## Contribute

Disclaimer: I have not thoroughly tested this extension, so there might be bugs. Bear with me while I'm fixing them :)

If you encounter a bug, please submit a issue. 

I welcome any contribution. Please submit a pull request if you want to contribute
