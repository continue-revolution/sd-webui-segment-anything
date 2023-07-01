import cv2
import torch
import numpy as np


Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_pred(pred: torch.Tensor, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape

    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    uncertain_area[pred>1-1.0/255.0] = 0
    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_
    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).cuda()
    
    return weight

def get_unknown_tensor_from_pred_oneside(pred: torch.Tensor, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    #uncertain_area[pred>1-1.0/255.0] = 0
    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_
    uncertain_area[pred>1-1.0/255.0] = 0
    #weight = np.zeros_like(uncertain_area)
    #weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(uncertain_area).cuda()
    return weight

Kernels_mask = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_mask(mask: torch.Tensor, rand_width=30, train_mode=True):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    N, C, H, W = mask.shape
    mask_c = mask.data.cpu().numpy().astype(np.uint8)

    weight = np.ones_like(mask_c, dtype=np.uint8)

    for n in range(N):
        if train_mode:
            width = np.random.randint(rand_width // 2, rand_width)
        else:
            width = rand_width // 2
        fg_mask = cv2.erode(mask_c[n,0], Kernels_mask[width])
        bg_mask = cv2.erode(1 - mask_c[n,0], Kernels_mask[width])
        weight[n,0][fg_mask==1] = 0
        weight[n,0][bg_mask==1] = 0
    weight = torch.from_numpy(weight).cuda()
    return weight

def get_unknown_tensor_from_mask_oneside(mask: torch.Tensor, rand_width=30, train_mode=True):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    N, C, H, W = mask.shape
    mask_c = mask.data.cpu().numpy().astype(np.uint8)

    weight = np.ones_like(mask_c, dtype=np.uint8)

    for n in range(N):
        if train_mode:
            width = np.random.randint(rand_width // 2, rand_width)
        else:
            width = rand_width // 2
        #fg_mask = cv2.erode(mask_c[n,0], Kernels_mask[width])
        fg_mask = mask_c[n,0]
        bg_mask = cv2.erode(1 - mask_c[n,0], Kernels_mask[width])
        weight[n,0][fg_mask==1] = 0
        weight[n,0][bg_mask==1] = 0
    weight = torch.from_numpy(weight).cuda()
    return weight

def get_unknown_box_from_mask(mask: torch.Tensor):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    N, C, H, W = mask.shape
    mask_c = mask.data.cpu().numpy().astype(np.uint8)

    weight = np.ones_like(mask_c, dtype=np.uint8)
    fg_set = np.where(mask_c[0][0] != 0)
    x_min = np.min(fg_set[1])
    x_max = np.max(fg_set[1])
    y_min = np.min(fg_set[0])
    y_max = np.max(fg_set[0])

    weight[0, 0, y_min:y_max, x_min:x_max] = 0
    weight = torch.from_numpy(weight).cuda()
    return weight