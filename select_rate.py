from utils import *
import copy
from skimage.metrics import structural_similarity as ssim
import lpips
import time
import sys
import torch
import PIL.Image as Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision import transforms as T
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F

from einops import rearrange, repeat
import math
import os

def lbp(image):
    """
    Compute the Local Binary Pattern (LBP) of an input image using PyTorch tensors.

    Args:
        image (torch.Tensor): A 4D tensor representing an RGB image (N x C x H x W),
                              where the values are in the range [0, 1].

    Returns:
        torch.Tensor: A 4D tensor of the same size as input, containing the LBP values as floating point numbers.
    """
    assert len(image.shape) == 4, "Input image must be a 4D tensor (N x C x H x W)."

    # Define the neighbors' offsets for the 8 surrounding pixels
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),  # Top row
        ( 0, -1),          ( 0, 1),  # Middle row (left and right)
        ( 1, -1), ( 1, 0), ( 1, 1)   # Bottom row
    ]

    # Get the batch size, number of channels, height, and width
    N, C, H, W = image.shape

    # Pad the image with zeros to handle borders
    padded_image = torch.nn.functional.pad(image, (1, 1, 1, 1), mode='constant', value=0)

    # Initialize the LBP result tensor
    lbp_result = torch.zeros_like(image, dtype=torch.float32)

    # Compute LBP for each pixel in each channel
    for n in range(N):
        for c in range(C):
            center_pixel = image[n, c]
            for idx, (dy, dx) in enumerate(offsets):
                # Get the neighboring pixels
                neighbor_pixel = padded_image[n, c, 1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
                # Perform binary comparison: 1 if neighbor >= center, else 0
                bit = (neighbor_pixel >= center_pixel).float()  # Binary comparison
                lbp_result[n, c] += (bit * (2 ** idx))  # Accumulate binary pattern in the LBP result

    return lbp_result


def sampling_rate_allocate3_PSNR(SR_x4,img1):
    _,_,H,W=img1.shape
    # print(img1.shape,SR_x4.shape)
    patch_size = 64
    rows = int(W / patch_size)
    cols = int(H / patch_size)
    img1=img1[0]*255
    SR_x4=SR_x4*255
    Confidence_Map = torch.abs(img1 - SR_x4)
    # print(Confidence_Map)
    block_nums = rows * cols
    CM_block_list_nopicked = []

    for i in range(block_nums):
        # CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
        #                    i % rows * patch_size:i % rows * patch_size + patch_size]
        # CM_block_mean=torch.mean(CM_block)

        CM_block_mean=psnr_get(img1[:, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                            i % rows * patch_size:i % rows * patch_size + patch_size].cpu().detach().numpy(),SR_x4[:, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                            i % rows * patch_size:i % rows * patch_size + patch_size].cpu().detach().numpy())

        CM_block_list_nopicked.append(100-CM_block_mean)
    CM_block_list_nopicked=sorted(CM_block_list_nopicked)

    #GMS Downscaling
    alpha=55
    #BICUBIC Downscaling
    # alpha=60

    q2 = np.percentile(CM_block_list_nopicked, 50)
    if q2<alpha:
        alpha=q2
    count = sum(1 for x in CM_block_list_nopicked if x < alpha)
    alpha2=CM_block_list_nopicked[-int(count/3.0+1)]
    # alpha2 = CM_block_list_nopicked[-int(count / 3.1 + 1)]        #4k
    print(count,alpha2)
    print(CM_block_list_nopicked)

    patch_size=64
    rows=int(W/patch_size)
    cols=int(H/patch_size)

    block_nums=rows*cols
    block_x2 = []
    block_x4 = []
    block_x8 = []
    CM_block_list_picked = []
    CM_block_list_nopicked = []
    # alpha=1.5
    # alpha2=2.3
    for i in range(block_nums):
        # CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
        #                    i % rows * patch_size:i % rows * patch_size + patch_size]
        # CM_block_mean=torch.mean(CM_block)

        CM_block_mean = 100-psnr_get(img1[:, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                                 i % rows * patch_size:i % rows * patch_size + patch_size].cpu().detach().numpy(),
                                 SR_x4[:, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                                 i % rows * patch_size:i % rows * patch_size + patch_size].cpu().detach().numpy())

        if CM_block_mean<=alpha:
            block_x8 += block_index_4_trans_1(i,rows)
        else:
            CM_block_list_nopicked.append(i)

    patch_size_d2=int(patch_size/2)
    rows_d2=rows*2
    CM_block_list_nopicked2=[]

    # alpha = alpha/4
    # alpha2 = alpha2/4

    for i in CM_block_list_nopicked:
        trans_4_2_list=block_index_4_trans_2(i,rows)
        # print(i,rows,trans_4_2_list)
        for j in trans_4_2_list:
            # CM_block = Confidence_Map[:, :, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
            #            j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2]
            # CM_block_mean = torch.mean(CM_block)

            CM_block_mean = 100-psnr_get(img1[:, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                       j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2].cpu().detach().numpy(),
                                     SR_x4[:, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                       j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2].cpu().detach().numpy())
            # print(CM_block_mean)
            if CM_block_mean <= alpha:
                # print(j,block_index_4_trans_2(j, rows_d2))
                block_x8 += block_index_4_trans_2(j, rows_d2)
            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4 += block_index_4_trans_2(j, rows_d2)
            else:
                CM_block_list_nopicked2.append(j)

    patch_size_d4 = int(patch_size / 4)
    rows_d4 = rows * 4
    # a = transforms.ToPILImage()(img1)
    # a.show()
    # alpha = alpha / 4
    # alpha2 = alpha2 / 4
    for i in CM_block_list_nopicked2:
        trans_4_2_list = block_index_4_trans_2(i, rows_d2)
        for j in trans_4_2_list:
            # CM_block = Confidence_Map[:, :,
            #            int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
            #            j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4]
            # CM_block_mean = torch.mean(CM_block)

            CM_block_mean = 100-psnr_get(
                img1[:,int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                       j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4].cpu().detach().numpy(),
                SR_x4[:,int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                       j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4].cpu().detach().numpy())
            if CM_block_mean <= alpha:
                block_x8.append(j)
            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4.append(j)
            else:
                block_x2.append(j)

    # a = transforms.ToPILImage()(img1)
    # a.show()

    return block_x2,block_x4,block_x8


from salient.code.network.MINet import MINet_VGG16
def sampling_rate_allocate3_salient(SR_x4,img1):
    chpoint = torch.load(
        "D:\code\Python\project\mechine Learning/block super resolution\multi scales SR\salient\MINet_VGG16.pth")
    salient_net = MINet_VGG16().cuda()
    salient_net.load_state_dict(chpoint)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    MEAN = [-mean / std for mean, std in zip(mean, std)]
    STD = [1 / std for std in std]
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)

    _, _, H, W = img1.shape
    # print(img1.shape,SR_x4.shape)
    patch_size = 64
    rows = int(W / patch_size)
    cols = int(H / patch_size)
    Confidence_Map = torch.abs(img1 * 255 - SR_x4 * 255)
    # with torch.no_grad():
    #     temp = checkpoint(salient_net, norm(img1.cuda()), use_reentrant=True)
    # Confidence_Map = temp.sigmoid().cpu()

    Confidence_Map=lbp(img1).float()/255
    print(Confidence_Map.shape)

    block_nums = rows * cols
    CM_block_list_nopicked = []

    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                   i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean = torch.mean(CM_block)
        CM_block_list_nopicked.append(CM_block_mean)
    CM_block_list_nopicked = sorted(CM_block_list_nopicked)

    # GMS Downscaling
    alpha = 0.6
    # BICUBIC Downscaling
    # alpha=60

    q2 = np.percentile(CM_block_list_nopicked, 50)
    if q2 < alpha:
        alpha = q2
    count = sum(1 for x in CM_block_list_nopicked if x < alpha)
    alpha2 = CM_block_list_nopicked[-int(count / 3.1 + 1)]
    # alpha2 = CM_block_list_nopicked[-int(count / 3.1 + 1)]        #4k
    print(count, alpha2)
    print(CM_block_list_nopicked)

    patch_size = 64
    rows = int(W / patch_size)
    cols = int(H / patch_size)
    # Confidence_Map = torch.abs(img1 * 255 - SR_x4 * 255)
    # print(Confidence_Map)
    img1 = img1[0]
    block_nums = rows * cols
    block_x2 = []
    block_x4 = []
    block_x8 = []
    CM_block_list_picked = []
    CM_block_list_nopicked = []
    # alpha=1.5
    # alpha2=2.3
    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                   i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean = torch.mean(CM_block)
        if CM_block_mean <= alpha:
            block_x8 += block_index_4_trans_1(i, rows)
        else:
            CM_block_list_nopicked.append(i)

    patch_size_d2 = int(patch_size / 2)
    rows_d2 = rows * 2
    CM_block_list_nopicked2 = []

    # alpha = alpha/4
    # alpha2 = alpha2/4

    for i in CM_block_list_nopicked:
        trans_4_2_list = block_index_4_trans_2(i, rows)
        # print(i,rows,trans_4_2_list)
        for j in trans_4_2_list:
            CM_block = Confidence_Map[:, :,
                       int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                       j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2]
            CM_block_mean = torch.mean(CM_block)
            # print(CM_block_mean)
            if CM_block_mean <= alpha:
                # print(j,block_index_4_trans_2(j, rows_d2))
                block_x8 += block_index_4_trans_2(j, rows_d2)
            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4 += block_index_4_trans_2(j, rows_d2)
            else:
                CM_block_list_nopicked2.append(j)

    patch_size_d4 = int(patch_size / 4)
    rows_d4 = rows * 4
    # a = transforms.ToPILImage()(Confidence_Map[0])
    # a.show()
    # alpha = alpha / 4
    # alpha2 = alpha2 / 4
    for i in CM_block_list_nopicked2:
        trans_4_2_list = block_index_4_trans_2(i, rows_d2)
        for j in trans_4_2_list:
            CM_block = Confidence_Map[:, :,
                       int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                       j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4]
            CM_block_mean = torch.mean(CM_block)
            if CM_block_mean <= alpha:
                block_x8.append(j)
            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4.append(j)
            else:
                block_x2.append(j)

    # a = transforms.ToPILImage()(img1)
    # a.show()

    return block_x2, block_x4, block_x8


def sampling_rate_allocate3(SR_x4,img1):
    _,_,H,W=img1.shape
    # print(img1.shape,SR_x4.shape)
    patch_size = 64
    rows = int(W / patch_size)
    cols = int(H / patch_size)
    Confidence_Map = torch.abs(img1 * 255 - SR_x4 * 255)
    # print(Confidence_Map)
    block_nums = rows * cols
    CM_block_list_nopicked = []

    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                           i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean=torch.mean(CM_block)
        CM_block_list_nopicked.append(CM_block_mean)
    CM_block_list_nopicked=sorted(CM_block_list_nopicked)

    #GMS Downscaling
    alpha=1.5
    #BICUBIC Downscaling
    # alpha=60

    q2 = np.percentile(CM_block_list_nopicked, 50)
    if q2<alpha:
        alpha=q2
    count = sum(1 for x in CM_block_list_nopicked if x < alpha)
    # alpha2=CM_block_list_nopicked[-int(count/3.1+1)]
    alpha2 = CM_block_list_nopicked[-int(count / 3.1 + 1)]

    if alpha2==CM_block_list_nopicked[-1]:
        alpha2=CM_block_list_nopicked[-1]*2
    # alpha2 = CM_block_list_nopicked[-int(count / 3.1 + 1)]        #4k
    # print(count,alpha2)
    # print(CM_block_list_nopicked)

    patch_size=64
    rows=int(W/patch_size)
    cols=int(H/patch_size)
    Confidence_Map=torch.abs(img1*255-SR_x4*255)
    # print(Confidence_Map)
    img1=img1[0]
    block_nums=rows*cols
    block_x2 = []
    block_x4 = []
    block_x8 = []
    CM_block_list_picked = []
    CM_block_list_nopicked = []
    # alpha=1.5
    # alpha2=2.3
    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                           i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean=torch.mean(CM_block)
        if CM_block_mean<=alpha:
            block_x8 += block_index_4_trans_1(i,rows)
        else:
            CM_block_list_nopicked.append(i)

    patch_size_d2=int(patch_size/2)
    rows_d2=rows*2
    CM_block_list_nopicked2=[]

    # alpha = alpha/4
    # alpha2 = alpha2/4

    for i in CM_block_list_nopicked:
        trans_4_2_list=block_index_4_trans_2(i,rows)
        # print(i,rows,trans_4_2_list)
        for j in trans_4_2_list:
            CM_block = Confidence_Map[:, :, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                       j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2]
            CM_block_mean = torch.mean(CM_block)
            # print(CM_block_mean)
            if CM_block_mean <= alpha:
                # print(j,block_index_4_trans_2(j, rows_d2))
                block_x8 += block_index_4_trans_2(j, rows_d2)
            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4 += block_index_4_trans_2(j, rows_d2)
            else:
                CM_block_list_nopicked2.append(j)

    patch_size_d4 = int(patch_size / 4)
    rows_d4 = rows * 4
    # a = transforms.ToPILImage()(img1)
    # a.show()
    # alpha = alpha / 4
    # alpha2 = alpha2 / 4
    for i in CM_block_list_nopicked2:
        trans_4_2_list = block_index_4_trans_2(i, rows_d2)
        for j in trans_4_2_list:
            CM_block = Confidence_Map[:, :,
                       int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                       j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4]
            CM_block_mean = torch.mean(CM_block)
            if CM_block_mean <= alpha:
                block_x8.append(j)
            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4.append(j)
            else:
                block_x2.append(j)

    # a = transforms.ToPILImage()(img1)
    # a.show()

    return block_x2,block_x4,block_x8

def block_index_4_trans_1(i,rows):
    n=int(i/rows)*16*rows
    list_change_index=[]
    for j in range(4):
        for k in range(4):
            index=n+(i%rows)*4+rows*j*4+k
            list_change_index.append(index)
    return list_change_index

def block_index_4_trans_2(i,rows):
    n=int(i/rows)*4*rows
    # print(n)
    list_change_index=[]
    for j in range(2):
        for k in range(2):
            index=n+(i%rows)*2+rows*j*2+k
            list_change_index.append(index)
    return list_change_index

def sampling_rate_allocate3_show(SR_x4,img1):
    _,_,H,W=img1.shape
    # print(img1.shape,SR_x4.shape)
    patch_size = 64
    rows = int(W / patch_size)
    cols = int(H / patch_size)
    Confidence_Map = torch.abs(img1 * 255 - SR_x4 * 255)
    # print(Confidence_Map)
    block_nums = rows * cols
    CM_block_list_nopicked = []

    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                           i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean=torch.mean(CM_block)
        CM_block_list_nopicked.append(CM_block_mean)
    CM_block_list_nopicked=sorted(CM_block_list_nopicked)

    #GMS Downscaling
    alpha=1.5
    #BICUBIC Downscaling
    # alpha=60

    q2 = np.percentile(CM_block_list_nopicked, 50)
    if q2<alpha:
        alpha=q2
    count = sum(1 for x in CM_block_list_nopicked if x < alpha)
    alpha2=CM_block_list_nopicked[-int(count/3.0+1)]
    # alpha2 = CM_block_list_nopicked[-int(count / 3.1 + 1)]        #4k
    print(count,alpha2)
    print(CM_block_list_nopicked)

    patch_size=64
    rows=int(W/patch_size)
    cols=int(H/patch_size)
    Confidence_Map=torch.abs(img1*255-SR_x4*255)
    # print(Confidence_Map)
    img1=img1[0]
    block_nums=rows*cols
    block_x2 = []
    block_x4 = []
    block_x8 = []
    CM_block_list_picked = []
    CM_block_list_nopicked = []
    # alpha=1.5
    # alpha2=2.3
    eta = 0.3

    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                           i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean=torch.mean(CM_block)
        if CM_block_mean<=alpha:
            block_x8 += block_index_4_trans_1(i,rows)

            light_blue = torch.tensor([0.5, 1, 1.0])
            img1[:, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
            i % rows * patch_size:i % rows * patch_size + patch_size] = eta * light_blue[:, None, None] + (1 - eta) * img1[:, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
            i % rows * patch_size:i % rows * patch_size + patch_size]

        else:
            CM_block_list_nopicked.append(i)

    patch_size_d2=int(patch_size/2)
    rows_d2=rows*2
    CM_block_list_nopicked2=[]

    a = transforms.ToPILImage()(img1)
    # a.show()
    a.save("./a.png")
    for i in CM_block_list_nopicked:
        trans_4_2_list=block_index_4_trans_2(i,rows)
        # print(i,rows,trans_4_2_list)
        for j in trans_4_2_list:
            CM_block = Confidence_Map[:, :, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                       j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2]
            CM_block_mean = torch.mean(CM_block)
            # print(CM_block_mean)
            if CM_block_mean <= alpha:
                # print(j,block_index_4_trans_2(j, rows_d2))
                block_x8 += block_index_4_trans_2(j, rows_d2)

                light_blue = torch.tensor([0.5, 1, 1.0])
                img1[:, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2] = eta * light_blue[:, None, None] + (
                            1 - eta) * img1[:, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2]

            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4 += block_index_4_trans_2(j, rows_d2)

                light_blue = torch.tensor([1, 1.0, 0.5])
                img1[:, int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2] = eta * light_blue[:, None, None] + (
                            1 - eta) * img1[:,int(j / rows_d2) * patch_size_d2:int(j / rows_d2) * patch_size_d2 + patch_size_d2,
                j % rows_d2 * patch_size_d2:j % rows_d2 * patch_size_d2 + patch_size_d2]
            else:
                CM_block_list_nopicked2.append(j)

    patch_size_d4 = int(patch_size / 4)
    rows_d4 = rows * 4
    a = transforms.ToPILImage()(img1)
    # a.show()
    a.save("./b.png")
    # alpha = alpha / 4
    # alpha2 = alpha2 / 4
    for i in CM_block_list_nopicked2:
        trans_4_2_list = block_index_4_trans_2(i, rows_d2)
        for j in trans_4_2_list:
            CM_block = Confidence_Map[:, :,
                       int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                       j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4]
            CM_block_mean = torch.mean(CM_block)
            if CM_block_mean <= alpha:
                block_x8.append(j)

                light_blue = torch.tensor([0.5, 1, 1.0])
                img1[:, int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4] = eta * light_blue[:, None, None] + (
                        1 - eta) * img1[:, int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4]

                # print('(',i,j,')',end=', ')
            elif alpha2 > CM_block_mean and CM_block_mean > alpha:
                block_x4.append(j)

                light_blue = torch.tensor([1, 1.0, 0.5])
                img1[:, int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4] = eta * light_blue[:, None, None] + (
                        1 - eta) * img1[:, int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4]

            else:
                block_x2.append(j)

                light_blue = torch.tensor([1.0, 0.5, 0.5])
                img1[:, int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4] = eta * light_blue[:, None, None] + (
                        1 - eta) * img1[:, int(j / rows_d4) * patch_size_d4:int(j / rows_d4) * patch_size_d4 + patch_size_d4,
                j % rows_d4 * patch_size_d4:j % rows_d4 * patch_size_d4 + patch_size_d4]

    # for i in range(1, cols*4):
    #     img1[:, i * patch_size_d4 - 1:i * patch_size_d4 + 0, :] = 1
    # for i in range(1, rows_d4):
    #     img1[:, :, i * patch_size_d4 - 1:i * patch_size_d4 + 0] = 1

    a = transforms.ToPILImage()(img1)
    # a.show()
    a.save("./c.png")
    return block_x2,block_x4,block_x8

def sampling_rate_allocate3_ablation_study(SR_x4,img1):
    _,_,H,W=img1.shape
    # print(img1.shape)
    patch_size=64
    rows=int(W/patch_size)
    cols=int(H/patch_size)
    Confidence_Map=torch.abs(img1*255-SR_x4*255)
    # print(Confidence_Map)
    block_nums=rows*cols
    block_x2 = []
    block_x4 = []
    block_x8 = []
    CM_block_list_nopicked=[]
    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                           i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean=torch.mean(CM_block)
        CM_block_list_nopicked.append(CM_block_mean)
    CM_block_list_nopicked=sorted(CM_block_list_nopicked)
    alpha=1.5

    q2 = np.percentile(CM_block_list_nopicked, 50)
    if q2<alpha:
        alpha=q2
    count = sum(1 for x in CM_block_list_nopicked if x < alpha)
    alpha2=CM_block_list_nopicked[-int(count/4+1)]

    # alpha=1.6
    # alpha2=8.9
    patch_size = 16
    rows = int(W / patch_size)
    cols = int(H / patch_size)
    block_nums = rows * cols
    for i in range(block_nums):
        CM_block = Confidence_Map[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
                           i % rows * patch_size:i % rows * patch_size + patch_size]
        CM_block_mean=torch.mean(CM_block)
        if CM_block_mean<=alpha:
            block_x8.append(i)
            # print(block_index_4_trans_1(i,rows))
        elif alpha2 > CM_block_mean and CM_block_mean > alpha:
            block_x4.append(i)
        else:
            block_x2.append(i)

    return block_x2,block_x4,block_x8