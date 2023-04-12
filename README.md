# Segment Anything for Stable Diffusion WebUI

This **developing** branch attempts to enable [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), achieving the goal of text prompt->object detection->segment anything->stable diffusion inpainting.

Warning: You may need a long time to get GroundingDINO and its dependency build, because they have C++.

To use this extension, simply run `git checkout GroundingDINO` on your terminal after `cd` to this extension directory. This branch will be merged to master branch someday, when I think it is stable enough.

## Next Step:

- point + detection, find a best way to satisfy need:
  - 0 point + N detection->box: BHW, mask: BCHW
  - ALL point + 1 detection->3 images
  - Preview detection + select the best

## How to Use

TODO

## Demo

TODO
