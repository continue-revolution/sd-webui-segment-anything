from .conv_sam import SAM_Decoder_Deep

__all__ = ['sam_decoder_deep']

def sam_decoder_deep(nc, **kwargs):
    model = SAM_Decoder_Deep(nc, [2, 3, 3, 2], **kwargs)
    return model