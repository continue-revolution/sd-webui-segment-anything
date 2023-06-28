# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling.mask_decoder_hq import MaskDecoderHQ
from .modeling.image_encoder import ImageEncoderViTHQ
from .modeling.tiny_vit import TinyViT
from segment_anything.modeling import PromptEncoder, Sam, TwoWayTransformer, MaskDecoder
from segment_anything import build_sam_vit_h, build_sam_vit_l, build_sam_vit_b


def build_sam_hq_vit_h(checkpoint=None):
    return _build_sam_hq(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_sam_hq_vit_l(checkpoint=None):
    return _build_sam_hq(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_hq_vit_b(checkpoint=None):
    return _build_sam_hq(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


def build_mobile_sam(checkpoint=None):
    return _build_mobile_sam(checkpoint)


sam_model_registry = {
    "sam_vit_h": build_sam_vit_h,
    "sam_vit_l": build_sam_vit_l,
    "sam_vit_b": build_sam_vit_b,
    "sam_hq_vit_h": build_sam_hq_vit_h,
    "sam_hq_vit_l": build_sam_hq_vit_l,
    "sam_hq_vit_b": build_sam_hq_vit_b,
    "mobile_sam": build_mobile_sam,
}


def _load_sam_checkpoint(sam: Sam, checkpoint=None):
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        info = sam.load_state_dict(state_dict, strict=False)
        print(info)
    for _, p in sam.named_parameters():
        p.requires_grad = False
    return sam

def _build_sam_hq(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViTHQ(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_embed_dim,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return _load_sam_checkpoint(sam, checkpoint)


def _build_mobile_sam(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
        image_encoder=TinyViT(
            img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        ),
        prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    return _load_sam_checkpoint(mobile_sam, checkpoint)
