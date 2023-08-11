# Segment Anything for Stable Diffusion WebUI

This extension aim for connecting [AUTOMATIC1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [Mikubill ControlNet Extension](https://github.com/Mikubill/sd-webui-controlnet) with [segment anything](https://github.com/facebookresearch/segment-anything/) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) to enhance Stable Diffusion/ControlNet inpainting, enhance ControlNet semantic segmentation, automate image matting and create LoRA/LyCORIS training set.

## News

- `2023/04/10`: [v1.0.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.0.0) SAM extension released! You can click on the image to generate segmentation masks.
- `2023/04/12`: [v1.0.1](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.0.1) Mask expansion and API support released by [@jordan-barrett-jm](https://github.com/jordan-barrett-jm)! You can expand masks to overcome edge problems of SAM.
- `2023/04/15`: [v1.1.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.1.0) [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) support released! You can enter text prompts to generate bounding boxes and segmentation masks.
- `2023/04/18`: [v1.2.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.2.0) [ControlNet V1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) inpainting support released! You can copy SAM generated masks to ControlNet to do inpainting. Note that you **must** update [ControlNet extension](https://github.com/Mikubill/sd-webui-controlnet) to use it. ControlNet inpainting has far better performance compared to general-purposed models, and you do not need to download inpainting-specific models anymore.
- `2023/04/24`: [v1.3.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.3.0) Automatic segmentation support released! Functionalities with * require you to have [ControlNet extension](https://github.com/Mikubill/sd-webui-controlnet) installed. This update includes support for 
    - *[ControlNet V1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) semantic segmentation
    - [EditAnything](https://github.com/sail-sg/EditAnything) un-semantic segmentation
    - Image layout generation (single image + batch process)
    - *Image masking with categories (single image + batch process)
    - *Inpaint not masked for ControlNet inpainting on txt2img panel
- `2023/04/29`: [v1.4.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.4.0) API has been completely refactored. You can access all features for **single image process** through API. API documentation has been moved to [wiki](https://github.com/continue-revolution/sd-webui-segment-anything/wiki/API).
- `2023/05/22`: [v1.4.1](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.4.1) [EditAnything](https://github.com/sail-sg/EditAnything) is ready to use! You can generate random segmentation and copy the output to EditAnything ControlNet.
- `2023/05/29`: [v1.4.2](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.4.2) You may now do SAM inference on CPU by checking "Use CPU for SAM". This is for some MAC users who are not able to do SAM inference on GPU. I discourage other users from using this feature because it is significantly slower than CUDA.
- `2023/06/01`: [v1.5.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.5.0) You may now choose to use local GroundingDINO to bypass C++ problem. See [FAQ](#faq)-1 for more detail.
- `2023/06/04`: [v1.5.1](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.5.1) `Upload Mask to ControlNet Inpainting` comes back in response to [ControlNet inpaint improvement](https://github.com/Mikubill/sd-webui-controlnet/discussions/1464). You should see a new tab beside `AutoSAM` after updating the extension. This feature will again be removed once ControlNet extension has its own uploading feature.
- `2023/06/13`: [v1.6.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.6.0) [SAM-HQ](https://github.com/SysCV/sam-hq) supported by [@SpenserCai](https://github.com/SpenserCai) and me. This is an "upgraded" SAM, created by researchers at ETH Zurich & HKUST. However, I cannot guarantee which one is better and you should make your own choice based on your own experiments. Go to [Installation](#installation) to get the link to the models.
- `2023/06/29`: [v1.6.1](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.6.1) [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) supported. This is a tiny version of SAM, created by researchers at Kyung Hee University. Visit [here](https://github.com/continue-revolution/sd-webui-segment-anything/issues/139) for more information.

Note that support for some other variations of SAM, such as [Matting-Anything](https://github.com/SHI-Labs/Matting-Anything) and [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) are still on the way. Support for these models, unlike MobileSAM, are non-trivial, especially FastSAM, which utilize a completely different pipeline, ultralytics/YOLO. Introducing these new works to the current codebase will make the original ugly-enough codebase more ugly. They will be supported once I finish a major refactor of the current codebase.

## FAQ

Thanks for suggestions from [github issues](https://github.com/continue-revolution/sd-webui-segment-anything/issues), [reddit](https://www.reddit.com/r/StableDiffusion/comments/12hkdy8/sd_webui_segment_everything/) and [bilibili](https://www.bilibili.com/video/BV1Tg4y1u73r/) to make this extension better.

There are already at least two great tutorials on how to use this extension. Check out [this video (Chinese)](https://www.bilibili.com/video/BV1Sm4y1B7Pg/) from [@ThisisGameAIResearch](https://github.com/ThisisGameAIResearch/) and [this video (Chinese)](https://www.bilibili.com/video/BV1Hh411j7b2/) from [@OedoSoldier](https://github.com/OedoSoldier). You can also check out my [demo](#demo).

You should know the following before submitting an issue.

1. Due to the overwhelming complaints about GroundingDINO installation and the lack of substitution of similar high-performance text-to-bounding-box library, I decide to modify the source code of GroundingDINO and push to this repository. **Starting from [v1.5.0](https://github.com/continue-revolution/sd-webui-segment-anything/releases/tag/v1.5.0), you can choose to use local GroundingDINO by checking `Use local groundingdino to bypass C++ problem` on `Settings/Segment Anything`.** This change should solve all problems about ninja, pycocotools, _C and any other problems related to C++/CUDA compilation.

    If you did not modify the setting described above, This script will firstly try to install GroundingDINO and check if your environment has successfully built the C++ dynamic library (the annoying `_C`). If so, this script will use the official implementation of GroundingDINO. This is to show respect to the authors of GroundingDINO. If the script failed to install GroundingDINO, it will use local GroundingDINO instead.

    If you'd still like to resolve the install problem of GroundingDINO, I observe some common problems for Windows users:
    - `pycocotool`: [here](https://github.com/cocodataset/cocoapi/issues/415#issuecomment-627313816).
    - `_C`: [here](https://github.com/continue-revolution/sd-webui-segment-anything/issues/32#issuecomment-1513873296). DO NOT skip steps.

    If you are still unable to install GroundingDINO on Windows AND you cannot resolve this problem AFTER searching for issues inside [here](https://github.com/continue-revolution/sd-webui-segment-anything/issues) [here](https://github.com/IDEA-Research/Grounded-Segment-Anything/issues) and [here](https://github.com/IDEA-Research/GroundingDINO/issues), You may refer to [#98](https://github.com/continue-revolution/sd-webui-segment-anything/issues/98) and watch the videos there. Note that I develop on linux, so I cannot guarantee that any video tutorials may or may not work.

2.  If you 
    - cannot use ControlNet/WebUI after installing this extension even when you are/are not using this extension
    - observe problems like [AttributeError: 'bool' object has no attribute 'enabled'](https://github.com/continue-revolution/sd-webui-segment-anything/issues/106) and [TypeError: 'bool' object is not subscriptable](https://github.com/continue-revolution/sd-webui-segment-anything/issues/93) from ControlNet
    - when you disable SAM extension, the problem disappear
    
    The problem is most likely due to some other extensions which might also change the position inside the extension list to control ControlNet. The easiest solution is [here](https://github.com/continue-revolution/sd-webui-segment-anything/issues/93#issuecomment-1546777860). This change will precede SAM extension to be before ControlNet, bypassing the internal preceding code, and will not prevent you from receiving any updates from me. I am not planning to refactor my code to bypass this problem. I did not expect to control ControlNet when I created this extension, but ControlNet indeed grow much faster than my expectation.

3. This extension has almost moved into maintenance phase. Although I don't think there will be huge updates in the foreseeable future, Mikubill ControlNet Extension is still fast developing, and I'm looking forward to more opportunities to connect my extension to ControlNet. Despite of this, I will continue to deal with issues, and monitor new research works to see if they are worth supporting. I welcome any community contribution and any feature requests.

4. You must use gradio>=3.23.0 and WebUI>=`22bcc7be` to use this extension. A1111 WebUI is stable, and some integrated package authors have also updated their packages (for example, if you are using the package from [@Akegarasu](https://github.com/Akegarasu), i.e. 秋叶整合包, it has already been updated according to [this video](https://www.bilibili.com/video/BV1iM4y1y7oA)). Also, supporting different versions of WebUI will be a huge time commitment, during which I can create many more features. Please update your WebUI and it is safe to use. I'm not planning to support some old commits of WebUI, such as `a9fed7c3`.

5. It is impossible to support the following features, at least at this moment, due to gradio/A1111 limitations. I will closely monitor gradio/A1111 update to see if it becomes possible to support them:
    - [color inpainting](https://github.com/continue-revolution/sd-webui-segment-anything/issues/22), because gradio wierdly enlarge the input image which slows down your browser, or even freeze your page. I have already implemented this feature, though, but I made it invisible. Note that ControlNet v1.1 inpainting model is very strong, and you do not need to rely on the traditional inpainting anymore. ControlNet v1.1 does not support color inpainting.
    - edit mask/explicit copy, because gradio [Image](https://gradio.app/docs/#image) component cannot accept image+mask as an output, which is the required way of explicitly copying a masked image to img2img inpaint/inpaint sketch/ControlNet (i.e. you can see the actual masked image on the panel, instead of a mysterious internal copying). Without this, you will not be able to edit mask.

6. [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) and [EditAnything](https://github.com/sail-sg/EditAnything) and A LOT of other popular SAM extensions have been supported. For Inpaint-Anything, you may check [this issue](https://github.com/continue-revolution/sd-webui-segment-anything/issues/60) for how to use. For EditAnything, please check [how to use](#controlnet). I am always open to support any other interesting applications, submit a feature request if you find another interesting one.

7. If you have a job opportunity and think I am a good fit, please feel free to send me an email.

8. If you want to sponsor me, please go to [sponsor](#sponsor) section and scan the corresponding QR code.

## Installation

Download this extension to `${sd-webui}/extensions` via whatever way you like (git clone or install from UI)

Choose one or more of the models below and put them to `${sd-webui}/models/sam` or `${sd-webui-segment-anything}/models/sam` (Choose one, not both. Remove the former folder if you choose to use the latter.). **Do not change model name, otherwise this extension may fail due to a bug inside segment anything.**

We support several variations of segmentation models:

1. [SAM](https://github.com/facebookresearch/segment-anything) from Meta AI.
    - [2.56GB sam_vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
    - [1.25GB sam_vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
    - [375MB sam_vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
    
    I myself tested vit_h on NVIDIA 3090 Ti which is good. If you encounter VRAM problem, you should switch to smaller models.

2. [SAM-HQ](https://github.com/SysCV/sam-hq) from SysCV.
    - [2.57GB sam_hq_vit_h](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth)
    - [1.25GB sam_hq_vit_l](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth)
    - [379MB sam_hq_vit_b](https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth)

3. [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) from Kyung Hee University.
    - [39MB mobile_sam](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt)

We plan to (**NOT supported yet**) support some other variations of segmentation models after a major refactor of the codebase:

4. [Matting-Anything](https://github.com/SHI-Labs/Matting-Anything) from SHI-Labs. This is a post-processing model for any variation of SAM. Put the model under `${sd-webui-segment-anything}/models/sam`
    - [11MB mam](https://huggingface.co/conrevo/SAM4WebUI-Extension-Models/resolve/main/mam.pth)

5. [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) from CASIA-IVA-Lab. This is a YOLO variation of SAM.
    - [145MB FastSAM-x](https://huggingface.co/conrevo/SAM4WebUI-Extension-Models/resolve/main/FastSAM-x.pt)

GroundingDINO packages, GroundingDINO models and ControlNet annotator models will be automatically installed the first time you use them. 

If your network does not allow you to access huggingface via the terminal, download GroundingDINO models from [huggingface](https://huggingface.co/ShilongLiu/GroundingDINO/tree/main) and put them under `${sd-webui-segment-anything}/models/grounding-dino`. Please note that GroundingDINO still need to access huggingface to download bert vocabularies. There is no alternative at this time. Read [here](https://github.com/continue-revolution/sd-webui-segment-anything/issues/138) to find a way to resolve this problem. I will try to find an alternative in the near future.

## GroundingDINO

GroundingDINO has been supported in this extension. It has the following functionalities:
1. You can use text prompt to automatically generate bounding boxes. You can separate different category names with `.`. SAM can convert these bounding boxes to masks
2. You can use point prompts with **ONE bounding box** to generate masks
3. You can go to `Batch Process` tab to do image matting and get LoRA/LyCORIS training set

However, there are some existing problems with GroundingDINO:
1. GroundingDINO will be install when you firstly use GroundingDINO features, instead of when you initiate the WebUI. Make sure that your terminal can have access to GitHub, otherwise you have to install GroundingDINO manually. GroundingDINO models will be automatically downloaded from [huggingface](https://huggingface.co/ShilongLiu/GroundingDINO/tree/main). If your terminal cannot visit HuggingFace, please manually download the model and put it under `${sd-webui-segment-anything}/models/grounding-dino`.
2. **If you want to use local groundingdino to bypass ALL the painful C++/CUDA/ninja/pycocotools problems, please read [FAQ](#faq)-1.** GroundingDINO requires your device to compile C++, which might take a long time and throw tons of exceptions. If you encounter `_C` problem, it's most probably because you did not install CUDA Toolkit. Follow steps decribed [here](https://github.com/continue-revolution/sd-webui-segment-anything/issues/32#issuecomment-1513873296). DO NOT skip steps. Otherwise, please go to [Grounded-SAM Issue Page](https://github.com/IDEA-Research/Grounded-Segment-Anything/issues) and submit an issue there. Despite of this, you can still use this extension for point prompts->segmentation masks even if you cannot install GroundingDINO, don't worry.
3. If you want to use point prompts, SAM can at most accept one bounding box. This extension will check if there are multiple bounding boxes. If multiple bounding boxes, this extension will disgard all point prompts; otherwise all point prompts will be effective. You may always select one bounding box you want.

For more detail, check [How to Use](#how-to-use) and [Demo](#demo).

## AutoSAM

Automatic Segmentation has been supported in this extension. It has the following functionalities:
1. You can use SAM to enhance semantic segmentation and copy the output to control_v11p_sd15_seg
2. You can generate random segmentation and copy the output to [EditAnything](https://github.com/sail-sg/EditAnything) ControlNet
3. You can generate image layout and edit them inside PhotoShop. Both single image and batch process are supported.
4. You can generate masks according to category IDs. This tend to be more accurate compared to purely SAM+GroundingDINO segmentation, if what you want is a large object.

However, there are some existing problems with AutoSAM:
1. You are required to install [Mikubill ControlNet Extension](https://github.com/Mikubill/sd-webui-controlnet) to use functionality 1 and 4. Please do not change the directory name (`sd-webui-controlnet`).
2. You are required to open WebUI via administrative mode the first time you access this feature if you are using Windows. This is because Windows does not allow visitors to create symbolic links via python.
3. You can observe drastic improvement if you combine `seg_ufade20k` and SAM. You may only observe some slight improvement if you combine one of the `Oneformer` preprocessors (`seg_ofade20k`&`seg_ofcoco`). This is because [Oneformer](https://github.com/SHI-Labs/OneFormer) is already very strong, compared to Uniformer, for semantic segmentation. SAM can only improve some details of semantic segmentation instead of showing some categories semantic models cannot show, because SAM is NOT a semantic-recognizable model.
4. Image layout generation has a pretty bad performance for anime images. I discourage you from using this functionality if you are dealing with anime images. I'm not sure about the performance for real images.

## How to Use

If you have previously enabled other copies while using this extension, you may want to click `Uncheck all copies` at the bottom of this extension UI, to prevent other copies from affecting your current page.

### Single Image
1. Upload your image
2. Optionally add point prompts on the image. Left click for positive point prompt (black dot), right click for negative point prompt (red dot), left click any dot again to cancel the prompt. You must add point prompt if you do not wish to use GroundingDINO.
3. Optionally check `Enable GroundingDINO`, select GroundingDINO model you want, write text prompt (separate different categories with `.`) and pick a box threshold (I highly recommend the default setting. High threshold may result in no bounding box). You must write text prompt if you do not wish to use point prompts.
4. Optionally enable previewing GroundingDINO bounding box and click `Generate bounding box`. You must write text prompt to preview bounding box. After you see the boxes with number marked on the top-left corner, uncheck all the boxes you do not want. If you uncheck all boxes, you will have to add point prompts to generate masks.
5. Click `Preview Segmentation` button. Due to the limitation of SAM, if there are multiple bounding boxes, your point prompts will not take effect when generating masks.
6. Choose your favorite segmentation.
7. Optionally check `Expand Mask` and specify the amount, then click `Update Mask`.
8. [**VERY IMPORTANT**] Update your ControlNet and check `Allow other script to control this extension` (**MUST**) on your ControlNet settings.

#### txt2img
1. You may only copy image and mask to ControlNet inpainting. 
2. Optionally check `ControlNet inpaint not masked` to invert mask colors and inpaint regions outside of the mask.
3. Select the correct ControlNet index where you are using inpainting, if you wish to use Multi-ControlNet. 
4. Configurate ControlNet panel. Click `Enable`, preprocessor choose `inpaint_global_harmonious`, model choose `control_v11p_sd15_inpaint [ebff9138]`. There is no need to upload image to the ControlNet inpainting panel.
5. Write your prompts, configurate A1111 panel and click `Generate`.

#### img2img
1. Check `Copy to Inpaint Upload & ControlNet Inpainting`. There is no need to select ControlNet index.
2. Configurate ControlNet panel. Click `Enable`, preprocessor choose `inpaint_global_harmonious`, model choose `control_v11p_sd15_inpaint [ebff9138]`. There is no need to upload image to the ControlNet inpainting panel.
3. Click `Switch to Inpaint Upload` button. There is no need to upload another image or mask, just leave them blank. Write your prompts, configurate A1111 panel and click `Generate`.

### Batch Process
1. Choose your SAM model, GroundingDINO model, text prompt, box threshold and mask expansion amount. Enter the source and destination directories of your images.
2. Choose `Output per image` to configurate the number of masks per bounding box. I highly recommend 3, since some masks might be wierd.
3. Click/unclick several checkboxes to configurate the images you want to save. See [demo](#demo) for what type of images these checkboxes represent.
4. Click `Start batch process` and wait. If you see "Done" below this button, you are all set.

### AutoSAM

1. Install and update [Mikubill ControlNet Extension](https://github.com/Mikubill/sd-webui-controlnet) before using it.
2. Configurate AutoSAM tunnable parameters according to descriptions [here](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L35-L96). Use default if you cannot understand.

#### ControlNet

1. Choose preprocessor.
    - `seg_ufade20k`, `seg_ofade20k` and `seg_ofcoco` are from ControlNet annotators. I highly recommend one of `seg_ofade20k` and `seg_ofcoco` because their performance are far better than `seg_ufade20k`. They are all compatible with `control_v11p_sd15_seg`. Optionally enable [pixel-perfect](https://github.com/Mikubill/sd-webui-controlnet/issues/924) to automatically pick the best preprocessor resolution. Configure your target width and height on txt2img/img2img default panel before preview if you wish to enable pixel perfect. Otherwise you need to manually set a preprocessor resolution. 
    - `random` is for [EditAnything](https://github.com/sail-sg/EditAnything). There is no need to set preprocessor resolution for random preprocessor since it does not contain semantic segmentation, but you need to pick an image from the AutoSeg output gallery to copy to ControlNet. 1 represents random colorization of different mask regions which is reserved for future ControlNet, 2 represents fixed colorization which can be EditAnything ControlNet control image.
2. Click preview segmentation image. For semantic semgentations, you will see 4 images where the left 2 are without SAM and the right 2 are with SAM. For random preprocessor, you will see 3 images where the top-left is the blended image, the top-right is random colorized masks and the down-left is for EditAnything ControlNet.
3. Check `Copy to ControlNet Segmentation` and select the correct ControlNet index where you are using ControlNet segmentation models if you wish to use Multi-ControlNet.
4. Configurate ControlNet panel. Click `Enable`, preprocessor choose `none`, model choose `control_v11p_sd15_seg [e1f51eb9]`. There is no need to upload image to the ControlNet segmentation panel.
5. Write your prompts, configurate A1111 panel and click `Generate`.
6. If you want to use [EditAnything](https://github.com/sail-sg/EditAnything), you need to modify some steps above:
    - In step 1: you need to choose `random` preprocessor.
    - Between step 3 & 4: download
        - SD 1.5 [weight](https://huggingface.co/shgao/edit-anything-v0-4-lllyasviel-format/resolve/main/control_v11p_sd15_sam_v0_4.pth) to `${a1111-webui}/models/ControlNet` or `${sd-webui-controlnet}/models`, [config](https://huggingface.co/shgao/edit-anything-v0-4-lllyasviel-format/resolve/main/control_v11p_sd15_sam_v0_4.yaml) to `${sd-webui-controlnet}/models`
        - SD 2.1 [weight](https://huggingface.co/shgao/edit-anything-v0-4-lllyasviel-format/resolve/main/control_v11p_sd21_sam_v0_4.pth) to `${a1111-webui}/models/ControlNet` or `${sd-webui-controlnet}/models`, [config](https://huggingface.co/shgao/edit-anything-v0-4-lllyasviel-format/resolve/main/control_v11p_sd21_sam_v0_4.yaml) to `${sd-webui-controlnet}/models`
    - In step 4: model choose `control_v11p_sd15_seg [e1f51eb9]`
        

#### Image Layout

1. For single image, simply upload image, enter output path and click generate. You will see a lot of images inside the output directory.
2. For batch process, simply enter source and destination directories and click generate. You will see a lot of images inside `${destination}/{image_filename}` directory.

#### Mask by Category

1. Choose preprocessor similar to [ControlNet step 1](#controlnet). This is pure semantic segmentation so there is no random preprocessor.
2. Enter category IDs separated by `+`. Visit [here](https://github.com/Mikubill/sd-webui-controlnet/blob/main/annotator/oneformer/oneformer/data/datasets/register_ade20k_panoptic.py#L12-L207) for ade20k and [here](https://github.com/Mikubill/sd-webui-controlnet/blob/main/annotator/oneformer/detectron2/data/datasets/builtin_meta.py#L20-L153) for coco to get category->id map. Note that coco jumps some numbers, so the actual ID is line_number - 21. For example, if you want bed+person, your input should be 7+12 for ade20k and 59+0 for coco.
3. For single image, upload image, click preview and configurate copy similar to [here for txt2img](#txt2img) and [here for img2img](#img2img).
4. For batch process, it is similar to [Batch process](#batch-process) step 2-4.

## Demo
Point prompts demo (also so-called Remove/Fill Anything)

https://user-images.githubusercontent.com/63914308/230916163-af661008-5a50-496e-8b79-8be7f193f9e9.mp4

GroundingDINO demo

https://user-images.githubusercontent.com/63914308/232157480-757f6e70-673a-4023-b4ca-df074ed30436.mp4

Batch process demo

![Configuration Image](https://user-images.githubusercontent.com/63914308/232157562-2f3cc9cc-310c-4b8b-89ba-216d2e014ca2.jpg) 

| Input Image | Output Image | Output Mask | Output Blend |
| --- | --- | --- | --- |
| ![Input Image](https://user-images.githubusercontent.com/63914308/232157678-fcaaf6b6-1805-49fd-91fa-8a722cc01c8a.png) | ![Output Image](https://user-images.githubusercontent.com/63914308/232157721-2754ccf2-b341-4b24-95f2-b75ac5b4fcd2.png) | ![Output Mask](https://user-images.githubusercontent.com/63914308/232157975-05de0b23-1225-4187-89b1-032c731b46eb.png) | ![Output Blend](https://user-images.githubusercontent.com/63914308/232158575-228f687c-8045-4079-bcf5-5a4dd0c8d7bd.png)

Semantic segmentation demo

https://user-images.githubusercontent.com/63914308/234080818-0bac3d67-4dfe-4666-b888-4e7d2e3c0cb4.mp4

Mask by Category demo (also so-called Replace Anything)

https://user-images.githubusercontent.com/63914308/234083907-cdb47082-d587-41fc-8da1-324e67e2749a.mp4

Mask by Category batch demo

![image](https://user-images.githubusercontent.com/63914308/234085307-462efd10-465b-488e-91b8-b75e5f814a47.png)

| Input Image | Output Image | Output Mask | Output Blend |
| --- | --- | --- | --- |
| ![1NHa6Wc](https://user-images.githubusercontent.com/63914308/234085498-70ca1d4c-cc5a-44d4-adb2-366630e5ce24.png) | ![1NHa6Wc_0_output](https://user-images.githubusercontent.com/63914308/234085495-0bfc4114-3e81-4ace-81d6-0f0f3186df25.png) | ![1NHa6Wc_0_mask](https://user-images.githubusercontent.com/63914308/234085491-8976f46c-2617-47ee-968e-0a9dd479c63a.png) | ![1NHa6Wc_0_blend](https://user-images.githubusercontent.com/63914308/234085503-7e041373-39cd-4f20-8696-986be517f188.png)

## Contribute

Disclaimer: I have not thoroughly tested this extension, so there might be bugs. Bear with me while I'm fixing them :)

If you encounter a bug, please submit an issue. Please at least provide your WebUI version, your extension version, your browser version, errors on your browser console log if there is any, error on your terminal log if there is any, to save both of our time.

I welcome any contribution. Please submit a pull request if you want to contribute

## Star History

Give me a star if you like this extension!

[![Star History Chart](https://api.star-history.com/svg?repos=continue-revolution/sd-webui-segment-anything&type=Date)](https://star-history.com/#continue-revolution/sd-webui-segment-anything&Date)

## Sponsor

You can sponsor me via WeChat or Alipay.

| WeChat | Alipay |
| --- | --- |
| ![216aff0250c7fd2bb32eeb4f7aae623](https://user-images.githubusercontent.com/63914308/232824466-21051be9-76ce-4862-bb0d-a431c186fce1.jpg) | ![15fe95b4ada738acf3e44c1d45a1805](https://user-images.githubusercontent.com/63914308/232824545-fb108600-729d-4204-8bec-4fd5cc8a14ec.jpg) |
