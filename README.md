# Segment Anything for Stable Diffusion WebUI

**This branch is still developing. I discourage you to use it before I remove this warning from README.**

This branch attemts to enable [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), achieving the goal of text prompt->object detection->segment anything->stable diffusion inpainting.

Warning: You may have a lot trouble using this extension from this branch, due to several incompability of SAM, GroundingDINO, etc. Please follow these steps to bypass potential error.

- If you see `ModuleNotFoundError: No module named 'groundingdino'` on your terminal, restart your webui may help.
- You may need a long time to get GroundingDINO and its dependency build, because they have C++.

To use this extension, simply run `git checkout GroundingDINO` on your terminal after `cd` to this extension directory. This branch will be merged to master branch someday, when I think it is stable enough.

## Help Wanted

If anyone know a good solution of installing `GroundingDINO`, please let me know and submit a pull request. My way of installment (in `install.py`) needs a restart, which is honestly not clean. I have not found another way of installing `GroundingDINO`

## Next Step:

- Find a smooth way to install GroundingDINO without restart
- point + detection, find a best way to satisfy need:
  - 0 point + N detection->box: BHW, mask: BCHW
  - ALL point + 1 detection->3 images
  - Preview detection + select the best


## How to Use

TODO

## Demo

TODO
