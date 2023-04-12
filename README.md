# Segment Anything for Stable Diffusion WebUI

This **developing** branch attempts to enable [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), achieving the goal of text prompt->object detection->segment anything->stable diffusion inpainting. 

Currently the extension with GroundingDINO enabled has the following behavior:
- accept point prompts or text prompts or both
- generate boxes and take the first box
- generate masks according to point prompts

The reason why it has such limitation is that SAM has limitation, as stated in [Next Step](https://github.com/continue-revolution/sd-webui-segment-anything/tree/GroundingDINO#next-step).

Warning: You may need a long time to get GroundingDINO and its dependency build, because they have C++.

To use this extension, simply run `git checkout GroundingDINO` on your terminal after `cd` to this extension directory. This branch will be merged to master branch someday, when the features in [Next Step](https://github.com/continue-revolution/sd-webui-segment-anything/tree/GroundingDINO#next-step) have been implemented and the conflict is resolved.

## Next Step:

SAM only support:
- case 1: ==0 point + >=1 box
- case 2: >=1 point + <=1 box

feature 1: automatic 
- when ==0 point, use all box, generate 3 image
- when >=1 point, use 1   box, generate 3 image

feature 2: preview 
- preview boxes and select all user want
- when selected multiple boxes, point prompt no effect
- choose the box the user want to preview segmentation, generate 3 image, point prompt effective
- blindly pair 1-1, 2-2, 3-3 choice from SAM (unless user specified) and generate 3 image

feature 3: batch
- blindly pair 1-1, 2-2, 3-3 choices from SAM and apply all boxes
- user input input path, output path, text prompt, box threshold, expansion amount
- user select
    - generate 1 or 3 masks for each image
    - default save the masked image
    - select whether save mask, whether save image with mask & box
