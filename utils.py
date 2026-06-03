import copy
import torch
import torchvision
import lpips
import time
from ultralytics import YOLO
import sys
# sys.path.append('D:\code\Python\project\mechine Learning\Jincancan\RCM-master/')

from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch
import PIL.Image as Image
import numpy as np
import cv2
import torchvision.transforms as transforms
# from ops_reuse.OmniSR import vgg19
from torchvision import transforms as T
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
from einops import rearrange, repeat
import math
import os

def deblock(frame):

    # 计算图像梯度
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=3)

    # 计算纹理复杂度
    texture = np.sqrt(gx**2 + gy**2)

    # 计算运动强度
    prev_frame = frame.copy()
    motion = cv2.absdiff(prev_frame, frame)

    # 自适应调整滤波强度
    weight = np.exp(-0.01 * texture) * (1 + 0.05 * motion)

    # 去块滤波
    filtered_frame = cv2.GaussianBlur(frame, (5, 5), 0.1)

    return filtered_frame

def H264(image):
    image = image.astype(np.float64)
    deblocking_image = np.copy(image)
    m, n = deblocking_image.shape
    block_size = 4
    alpha = 50
    beta = 1

    for i in range(block_size, m - block_size + 1, block_size):
        for j in range(block_size, n - block_size + 1, block_size):
            for k in range(block_size):
                p3 = deblocking_image[i, j - 4]
                p2 = deblocking_image[i, j - 3]
                p1 = deblocking_image[i, j - 2]
                p0 = deblocking_image[i, j - 1]
                q0 = deblocking_image[i, j]
                q1 = deblocking_image[i, j + 1]
                q2 = deblocking_image[i, j + 2]
                q3 = deblocking_image[i, j + 3]
                ap = np.abs(p2 - p0)
                aq = np.abs(q2 - q0)

                # Left/upper side
                if ap < beta and np.abs(p0 - q0) < ((alpha >> 2) + 2):
                    deblocking_image[i, j - 1] = np.uint8((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) // 8)
                    deblocking_image[i, j - 2] = np.uint8((p2 + p1 + p0 + q0 + 2) // 4)
                    deblocking_image[i, j - 3] = np.uint8((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) // 8)
                else:
                    deblocking_image[i, j - 1] = np.uint8((2 * p1 + p0 + q1 + 2) // 4)

                if aq < beta and np.abs(p0 - q0) < ((alpha >> 2) + 2):
                    deblocking_image[i, j] = np.uint8((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) // 8)
                    deblocking_image[i, j + 1] = np.uint8((p0 + q0 + q1 + q2 + 2) // 4)
                    deblocking_image[i, j + 2] = np.uint8((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) // 8)
                else:
                    deblocking_image[i, j] = np.uint8((2 * q1 + q0 + p1 + 2) // 4)
    return deblocking_image

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    # return mse
    return 10 * math.log10(255.0 ** 2 / mse)

def check_image_size(x, scales):
    _, h, w = x.size()
    mod_pad_h = (scales - h % scales) % scales
    mod_pad_w = (scales - w % scales) % scales
    # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
    return x
def check_box(box, scales, H, W):
    h = box[3] - box[1]
    w = box[2] - box[0]
    mod_pad_h = (scales - h % scales) % scales
    mod_pad_w = (scales - w % scales) % scales
    box[3] += int(mod_pad_h / 2)
    box[2] += int(mod_pad_w / 2)
    box[1] -= mod_pad_h-int(mod_pad_h / 2)
    box[0] -= mod_pad_w-int(mod_pad_w / 2)

    if box[1]<0:
        box[1]=0
    if box[0] < 0:
        box[0] = 0
    if box[3] > H:
        box[3] = H
    if box[2] > W:
        box[2] = W

    return box


def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)  # 左上角的横坐标
    ymin = np.maximum(y1min, y2min)  # 左上角的纵坐标
    xmax = np.minimum(x1max, x2max)  # 右下角的横坐标
    ymax = np.minimum(y1max, y2max)  # 右下角的纵坐标

    # 计算相交矩形的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w

    # 计算相并面积
    union = s1 + s2 - intersection

    # 计算交并比
    iou = intersection / union
    return iou

def return_max(a,b):
    if a>b:
        return a
    else:
        return b

def return_min(a,b):
    if a>b:
        return b
    else:
        return a

def merge_box(boxs,threshold=0.3):
    i=0
    j=1
    while i<len(boxs)-1:
        flag = 0
        while j<len(boxs):
            iou=box_iou_xyxy(boxs[i],boxs[j])
            if iou>threshold:
                boxs[i][0]=return_min(boxs[i][0],boxs[j][0])
                boxs[i][1] = return_min(boxs[i][1], boxs[j][1])
                boxs[i][2] = return_max(boxs[i][2], boxs[j][2])
                boxs[i][3] = return_max(boxs[i][3], boxs[j][3])
                # del boxs[j]
                boxs=torch.cat((boxs[:j],boxs[j+1:]))
                flag=1
            else:
                j+=1
        if flag==0:
            i+=1
    return boxs

def Cmp(x):
    return x[0]
def get_area(boxs):
    area_list=[]
    for i in boxs:
        area_list.append([(i[2]-i[0])*(i[3]-i[1]),i])

    area_list.sort(key=Cmp,reverse=True)
    # area_list.sort(key=Cmp)
    # print(area_list)
    return area_list