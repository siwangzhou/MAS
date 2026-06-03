import time
from skimage import img_as_float
from niqe import niqe
from scipy.ndimage import gaussian_filter
from scipy.special import gammaln
from scipy.stats import genpareto
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch
import torch_pruning as tp
import PIL.Image as Image
import numpy as np
import cv2
import torchvision.transforms as transforms
# from ops.OmniSR import OmniSR,LR_SR_x8,LR_SR_x4,LR_SR_x3,LR_SR_x2
from network.ops.edsr.edsr import LR_SR_x8, LR_SR_x4, LR_SR_x3, LR_SR_x2

from torchvision import transforms as T
from einops import rearrange
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
from einops import rearrange, repeat
import math
import os

transform = transforms.Compose([
    transforms.ToTensor(),  # 将像素值从[0,255]变为[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化到[-1,1]
    transforms.Lambda(lambda x: torch.stack((x[0], x[1]-x[0].mean(), x[2]-x[0].mean()), dim=0)),  # 将三个通道的数据分开并减去第一个通道的数据均值，得到Y、Cb、Cr通道
])


def psnr_get(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def calculate_niqe(image):
    image = img_as_float(image)
    h, w = image.shape[:2]
    block_size = 32
    strides = 32
    features = []


# net=torch.load("./Net/OSAG_no_p2p1.pt")
# net=torch.load("./Net/OSAG_reuse_no_p2p970.pt")
# vgg=vgg19()
# denoise=torch.load("./Net/denoise_x4_1_16/denoise_x4_1_16_2000.pt")
# net=torch.load("./Net/Inv_Omni_p2p/Inv_Omni_p2p_135.pt").cuda()

kwards = {'upsampling': 4,
              'res_num': 5,
              'block_num': 1,
              'bias': True,
              'block_script_name': 'OSA',
              'block_class_name': 'OSA_Block',
              'window_size': 8,
              'pe': True,
              'ffn_bias': True}
net_sr = LR_SR_x4().cuda()
# net_sr = LR_SR_x4().cuda()
# net = InvArbEDRS(4).cuda()
chpoint = torch.load("D:\code\Python\project\mechine Learning/block_rescaling_32\SRNet\EDSR_P2P_quant_x4_norm.pt")
# chpoint = torch.load("D:\code\Python\project\mechine Learning/block_rescaling_v3/new_SR_model_train/Net\enhance_edsr_fuse_Y\enhance_edsr_fuse_Y_194.pt")
net_sr.load_state_dict(chpoint["net"])

# net_enhance = net_LMAR_enhance().cuda()
# chpoint = torch.load("D:\code\Python\project\mechine Learning/block_rescaling_v3/new_SR_model_train\enhance_model/base_model.bin")
# net_enhance.load_state_dict(chpoint["state_dict"])

# net_enhance=net_sr.net_enhance
# net_enhance = CIDNet().cuda()
# net_enhance.load_state_dict(torch.load('D:\code\Python\project\mechine Learning/block_rescaling_v3/new_SR_model_train\CIDNet/best_PSNR.pth', map_location=lambda storage, loc: storage))

# from torchsummary import summary
# summary(net.cuda(), input_size=(3, 64, 64), batch_size=1)
# net.eval()
# net.cpu()
# net1=torch.load("D:\Code\Python\code\机器学习\超分辨率\SR_NSR/Net/SR_2K_x2_epoch9.0.pt")
# net1.cpu()
# params = list(net.parameters())
# k = 0

# from torchstat import stat

# str='D:\code\Python\project\mechine Learning\Data\DIV2K\DIV2K_valid_HR'
# str='D:\code\Python\project\mechine Learning\Data\Set5\Set5\original'
str='D:\code\Python\project\mechine Learning\Data\Set14\Set14\original'
test_list=os.listdir(str)
# test_list2=os.listdir(str2)
# test_list_lr=os.listdir(str_lr)
psnr1_sum=0
psnr2_sum=0
niqo_sum=0
scales=32
sizea=0
sizec=0
time_sum=0
k=0
# nor=T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# for i,j in zip(test_list,test_list_lr):
for i in test_list:
    print(i)
    photo_str=str+'/'+i
    img1=Image.open(photo_str).convert('RGB')

    size=(np.array(img1.size)/scales).astype(int)
    w=size[0]
    h=size[1]
    img1 = img1.crop((0, 0,w*scales,h*scales))
    # img1=transforms.RandomCrop((h*scales,w*scales))(img1)
    img_LR=img1.resize(size*8, Image.BICUBIC)
    img_SR=img_LR.resize((w*scales,h*scales), Image.BICUBIC)

    img1=transforms.ToTensor()(img1).unsqueeze(0).cuda()
    img_LR=transforms.ToTensor()(img_LR).unsqueeze(0).cuda()
    img_SR = transforms.ToTensor()(img_SR).unsqueeze(0).cuda()


    t1=time.time()
    # lr,_, source=checkpoint(net_sr,img1.cuda(),use_reentrant=True)
    # source = checkpoint(net_enhance, img_SR.cuda(), use_reentrant=True)
    # lr = F.interpolate(img1, scale_factor=1 / 4, mode='bicubic')

    _,source = checkpoint(net_sr, img1.cuda()*2-1, use_reentrant=True)
    # source = F.interpolate(source, scale_factor=4, mode='bicubic')
    # lr=lr/2+0.5
    source=source/2+0.5

    i1=source.cpu().detach().numpy()[0]

    i2 = img1.cpu().detach().numpy()[0]

    i1 = 65.481 * i1[0, :, :] + 128.553 * i1[1, :, :] + 24.966 * i1[2, :, :] + 16
    i2 = 65.481 * i2[0, :, :] + 128.553 * i2[1, :, :] + 24.966 * i2[2, :, :] + 16
    psnr1 = psnr_get(i1, i2)
    # psnr2 = psnr_get(high_photo_img1[1], source_ycbcr[1])
    # psnr3 = psnr_get(high_photo_img1[2], source_ycbcr[2])
    # print(psnr1,psnr2,psnr2)

    psnr1_sum+=psnr1
    k+=1
    print(psnr1, psnr1_sum / k)


# print('PSNR:',psnr1_sum/len(test_list),'time:',time_sum)
print('PSNR:',psnr1_sum/20,'time:',time_sum)
# print('PSNR:',psnr1_sum)
