import sys
sys.path.append('D:\code\Python\project\mechine Learning/block_rescaling_32/network')
sys.path.append('D:\code\Python\project\mechine Learning/block_rescaling_32/method2/uniontrain/')
from utils import *
import copy
from skimage.metrics import structural_similarity as ssim
import lpips
import time
from thop import profile
from select_rate import sampling_rate_allocate3,sampling_rate_allocate3_ablation_study
import sys
# from network.hat_model.hat_arch import HAT
# sys.path.append('D:\code\Python\project\mechine Learning\Jincancan\RCM-master/')
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch
from network.ops.OmniSR import OmniSR,LR_SR_x8,LR_SR_x4,LR_SR_x2
from network.ops.OSAG import OSAG
# from method2.uniontrain.method2_union_net import OmniSR_P2P_union2
from method2.uniontrain.method2_union_net_val import OmniSR_P2P_union2_val
import PIL.Image as Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision import transforms as T
from tqdm import tqdm
import torch.nn.functional as F
import torch_pruning as tp
from einops import rearrange, repeat
import math
import os


def cmp(x):
    return x[0]

lpips_model = lpips.LPIPS(net="alex").cuda()

SRnet2 = LR_SR_x2(kwards={'upsampling': 2,'res_num': 5,'block_num': 1,'bias': True,'block_script_name': 'OSA','block_class_name': 'OSA_Block','window_size': 8,
                     'pe': True,
                     'ffn_bias': True}).cuda()
chpoint= torch.load("D:/code/Python/project/mechine Learning/block super resolution/multi scales SR/SRNet/Omni_P2P_quant_x2_norm_blockzero.pt")
SRnet2.load_state_dict(chpoint['net'])
SRnet2.eval()

SRnet4 = LR_SR_x4(kwards={'upsampling': 4,'res_num': 5,'block_num': 1,'bias': True,'block_script_name': 'OSA','block_class_name': 'OSA_Block','window_size': 8,
                     'pe': True,
                     'ffn_bias': True}).cuda()
chpoint= torch.load("D:/code/Python/project/mechine Learning/block super resolution/multi scales SR/SRNet/Omni_P2P_quant_x4_norm_blockzero.pt")
SRnet4.load_state_dict(chpoint['net'])
SRnet4.eval()

down=SRnet4.layer1
up=SRnet4.layer2


SRnet8 = LR_SR_x8(kwards={'upsampling': 8,'res_num': 5,'block_num': 1,'bias': True,'block_script_name': 'OSA','block_class_name': 'OSA_Block','window_size': 8,
                     'pe': True,
                     'ffn_bias': True}).cuda()
chpoint= torch.load("D:/code/Python/project/mechine Learning/block super resolution/multi scales SR/SRNet/Omni_P2P_quant_x8_norm_blockzero.pt")
SRnet8.load_state_dict(chpoint['net'])
SRnet8.eval()

for name, param in SRnet2.named_parameters():
    param.requires_grad = False
for name, param in SRnet4.named_parameters():
    param.requires_grad = False
for name, param in SRnet8.named_parameters():
    param.requires_grad = False

union=OmniSR_P2P_union2_val().cuda()
chpoint=torch.load('D:\code\Python\project\mechine Learning/block_rescaling_32\method2/uniontrain/net/Omni_P2P_union2/Omni_P2P_union2.pt')
union.load_state_dict(chpoint['net'])
union.eval()

# Set5
# photo_path = 'D:\code\Python\project\mechine Learning\Data\Set5\Set5\original'
# photo_path='D:\code\Python\project\mechine Learning\Data\Set14\Set14\original'
# photo_path = 'D:\code\Python\project\mechine Learning\Data\BSDS100\original'
# str = 'D:\code\Python\project\mechine Learning\Data/Urban100/HR'
# str='D:\code\Python\project\mechine Learning\Data\BSD68\BSD68'
# photo_path = 'D:\code\Python\project\mechine Learning\Data\DIV2K\DIV2K_valid_HR'
photo_path = 'D:\code\Python\project\mechine Learning\Data/test2k'
# photo_path = 'D:\code\Python\project\mechine Learning\Data/test4k'


test_list = os.listdir(photo_path)
psnr1_sum = 0
ssim1_sum = 0
psnr2_sum = 0
niqo_sum = 0
distance = 0
psnr = 0
patch_size=16
patch_size_x2=int(patch_size/2)
patch_size_x4=int(patch_size/4)
patch_size_x8=int(patch_size/8)
alpha_x8=1.8
alpha_x2=8.2
sizea = 0
sizec = 0
HR_datasize_sum=0
SR_datasize_sum=0

time_sum=0

flops_sum=0

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
MEAN = [-mean / std for mean, std in zip(mean, std)]
STD = [1 / std for std in std]
denormalizer = transforms.Normalize(mean=MEAN, std=STD)
# nor=T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# for i,j in zip(test_list,test_list_lr):
# f=open("D:\code\Python\project\mechine Learning\Jincancan\VOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\ImageSets\Segmentation/val.txt",'r')
# test_list = f.readlines()[:100]
block_imgnums=0
t = time.time()
# test_list=test_list[17:18]  #24-2k
# test_list=test_list[35:36]  #40-2k

for imgname in tqdm(test_list):
    # for imgname in test_list:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    photo_str = photo_path + '/' + imgname
    # photo_str = photo_path + '/' + imgname[:-1]+'.jpg'
    # print(photo_str)

    img = Image.open(photo_str).convert('RGB')

    W, H = img.size
    # img = img.resize((int(W / 2), int(H / 2)), Image.BICUBIC)
    # W, H = img.size
    patch_size_crop=64
    img = img.crop((0, 0, int(W / patch_size_crop) * patch_size_crop, int(H / patch_size_crop) * patch_size_crop,))
    img1 = transforms.ToTensor()(img).unsqueeze(0)
    H, W = img1.shape[2:]

    H = int(H / patch_size) * patch_size
    W = int(W / patch_size) * patch_size
    rows = int(W / patch_size)

    block_nums = int(H / patch_size * W / patch_size)
    img = transforms.ToPILImage()(img1[0])
    h, w = img1.shape[2:]
    # print(img1.shape)

    # img_LR = img.crop((0,0,256,256)).resize((int(w / scales), int(h / scales)), Image.BICUBIC)
    img_LR2 = img.resize((int(w / 2), int(h / 2)), Image.BICUBIC)
    img_LR4 = img.resize((int(w / 4), int(h / 4)), Image.BICUBIC)
    img_LR8 = img.resize((int(w / 8), int(h / 8)), Image.BICUBIC)

    img_BICUBIC = img_LR4.resize((w, h), Image.BICUBIC)

    img_BICUBIC_x4 = img_LR4.resize((int(w), int(h)), Image.BICUBIC)
    img_BICUBIC_x4 = transforms.ToTensor()(img_BICUBIC_x4).cuda()
    img_BICUBIC_x8 = img_LR8.resize((int(w), int(h)), Image.BICUBIC)
    img_BICUBIC_x8 = transforms.ToTensor()(img_BICUBIC_x8).cuda()

    img_LR4 = transforms.ToTensor()(img_LR4).unsqueeze(0).cuda()
    img_LR2 = transforms.ToTensor()(img_LR2).unsqueeze(0).cuda()
    img_LR8 = transforms.ToTensor()(img_LR8).unsqueeze(0).cuda()
    img_BICUBIC = transforms.ToTensor()(img_BICUBIC).unsqueeze(0)[:, :, :H, :W]
    # img_BICUBIC_x4 = copy.deepcopy(img_BICUBIC)
    # img_BICUBIC_x2 = copy.deepcopy(img_BICUBIC)
    # img_BICUBIC_x8 = copy.deepcopy(img_BICUBIC)

    torch.cuda.empty_cache()
    t1 = time.time()
    # img_LR4_noblock,temp_x4=checkpoint(SRnet4, (img1.cuda()-0.5)*2, use_reentrant=True)
    # img_LR4_noblock=img_LR4_noblock/2+0.5
    # img_blocks=torch.cat([img1[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size, i % rows * patch_size:i % rows * patch_size + patch_size] for i in range(block_nums)], 0)
    # torch.cuda.empty_cache()
    #
    #
    # torch.cuda.empty_cache()
    # temp_LR4, temp_x4 = checkpoint(SRnet4, (img1.cuda() - 0.5) * 2, use_reentrant=True)
    # for i in range(block_nums):
    #     # img_LR4[:, :, int(i / rows) * 32:int(i / rows) * 32 + 32, i % rows * 32:i % rows * 32 + 32] = temp_LR4[:, :, int(i / rows) * 32:int(i / rows) * 32 + 32, i % rows * 32:i % rows * 32 + 32] / 2 + 0.5
    #     img_BICUBIC_x4[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size, i % rows * patch_size:i % rows * patch_size + patch_size] = \
    #     temp_x4[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size, i % rows * patch_size:i % rows * patch_size + patch_size] / 2 + 0.5
    #
    # torch.cuda.empty_cache()
    # temp_LR2, temp_x2 = checkpoint(SRnet2, (img1.cuda() - 0.5) * 2, use_reentrant=True)
    # for i in range(block_nums):
    #     # img_LR2[:, :, int(i / rows) * 64:int(i / rows) * 64 + 64, i % rows * 64:i % rows * 64 + 64] = temp_LR2[:, :, int(i / rows) * 64:int(i / rows) * 64 + 64, i % rows * 64:i % rows * 64 + 64] / 2 + 0.5
    #     img_BICUBIC_x2[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size, i % rows * patch_size:i % rows * patch_size + patch_size] = \
    #     temp_x2[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size, i % rows * patch_size:i % rows * patch_size + patch_size] / 2 + 0.5
    #
    # torch.cuda.empty_cache()
    # temp_LR8, temp_x8 = checkpoint(SRnet8, (img1.cuda() - 0.5) * 2, use_reentrant=True)
    # for i in range(block_nums):
    #     # img_LR8[:, :, int(i / rows) * 16:int(i / rows) * 16 + 16, i % rows * 16:i % rows * 16 + 16] = temp_LR8[:, :, int(i / rows) * 16:int(i / rows) * 16 + 16, i % rows * 16:i % rows * 16 + 16] / 2 + 0.5
    #     img_BICUBIC_x8[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size, i % rows * patch_size:i % rows * patch_size + patch_size] = \
    #     temp_x8[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size, i % rows * patch_size:i % rows * patch_size + patch_size] / 2 + 0.5

#采样率分配
    # block_x2,block_x4,block_x8=sampling_rate_allocate2(img_BICUBIC_x2, img_BICUBIC_x4, img_BICUBIC_x8, img1, patch_size, rows, block_nums, alpha_x2,
    #                        alpha_x8)
    # print(block_x4,sep='\n')
    # t1=time.time()
    block_x2, block_x4, block_x8 = sampling_rate_allocate3((img_BICUBIC_x4).cpu(), img1)
    # block_x2, block_x4, block_x8 = sampling_rate_allocate3_ablation_study((temp_x4/2+0.5).cpu(), img1)
    # print(len(block_x2),len(block_x4),len(block_x8),time.time()-t1)
    with torch.no_grad():
        _ = checkpoint(SRnet4.layer1, img1.cuda())
        _ = checkpoint(SRnet2.layer1, img1.cuda())
        _ = checkpoint(SRnet8.layer1, img1.cuda())

    time_sum+=time.time()-t1
    # continue

    HR_datasize = h * w
    scale_datasize=0
    for i in block_x2:
        scale_datasize+=patch_size_x2**2
    for i in block_x4:
        scale_datasize+=patch_size_x4**2
    for i in block_x8:
        scale_datasize+=patch_size_x8**2

    scale_factor = math.sqrt(HR_datasize / scale_datasize)
    print(scale_factor)
    HR_datasize_sum += HR_datasize
    SR_datasize_sum += scale_datasize

    source = torch.zeros_like(img_BICUBIC)

    #get LRimage
    HR2 = torch.zeros_like(source).cuda()+0.5
    img1=img1.cuda()
    for i in block_x2:
        if int(i / rows) * patch_size<0:
            temp1=0
        else:
            temp1=int(i / rows) * patch_size
        if i % rows * patch_size <0:
            temp2=0
        else:
            temp2=i % rows * patch_size
        HR2[:, :, temp1:int(i / rows) * patch_size + patch_size,
        temp2:i % rows * patch_size + patch_size] = img1[:, :, temp1:int(
            i / rows) * patch_size + patch_size, temp2:i % rows * patch_size + patch_size]
    img_LR2=checkpoint(SRnet2.layer1,HR2*2-1)
    img_LR2 = checkpoint(SRnet2.layer2, img_LR2)/2+0.5

    HR4 = torch.zeros_like(source).cuda()+0.5
    for i in block_x4:
        if int(i / rows) * patch_size<0:
            temp1=0
        else:
            temp1=int(i / rows) * patch_size
        if i % rows * patch_size <0:
            temp2=0
        else:
            temp2=i % rows * patch_size
        HR4[:, :, temp1:int(i / rows) * patch_size + patch_size,
        temp2:i % rows * patch_size + patch_size] = img1[:, :, temp1:int(
            i / rows) * patch_size + patch_size, temp2:i % rows * patch_size + patch_size]
    img_LR4 = checkpoint(SRnet4.layer1, HR4*2-1)
    img_LR4 = checkpoint(SRnet4.layer2, img_LR4)/2+0.5

    HR8 = torch.zeros_like(source).cuda()+0.5
    for i in block_x8:
        if int(i / rows) * patch_size<0:
            temp1=0
        else:
            temp1=int(i / rows) * patch_size
        if i % rows * patch_size <0:
            temp2=0
        else:
            temp2=i % rows * patch_size
        HR8[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
        temp2:i % rows * patch_size + patch_size] = img1[:, :, int(i / rows) * patch_size:int(
            i / rows) * patch_size + patch_size, temp2:i % rows * patch_size + patch_size]
    img_LR8 = checkpoint(SRnet8.layer1, HR8*2-1)
    img_LR8 = checkpoint(SRnet8.layer2, img_LR8)/2+0.5

    # l2=transforms.ToPILImage()(img_LR2[0])
    # l2.show()
    # l4 = transforms.ToPILImage()(img_LR4[0])
    # l4.show()
    # l8 = transforms.ToPILImage()(img_LR8[0])
    # l8.show()
    # flops_temp1, params = profile(SRnet4, inputs=(img1.cuda(),))
    # flops_temp2, params = profile(SRnet2, inputs=(img1.cuda(),))
    # flops_temp3, params = profile(SRnet8, inputs=(img1.cuda(),))
    # flops_sum+=flops_temp1+flops_temp2+flops_temp3
    # continue

    if len(block_x2) >= 0:
        # 分块超分
        # x2
        LR2 = torch.zeros_like(img_LR2)+0.5
        # LR2 = img_LR2
        torch.cuda.empty_cache()
        for i in block_x2:
            LR2[:, :, int(i / rows) * patch_size_x2:int(i / rows) * patch_size_x2 + patch_size_x2, i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2]=img_LR2[:, :, int(i / rows) * patch_size_x2:int(i / rows) * patch_size_x2 + patch_size_x2, i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2]

        # x4
        LR4 = torch.zeros_like(img_LR4)+0.5
        # LR4 = img_LR4
        torch.cuda.empty_cache()
        for i in block_x4:
            LR4[:, :, int(i / rows) * patch_size_x4:int(i / rows) * patch_size_x4 + patch_size_x4,
            i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4] = img_LR4[:, :,
                                                                                 int(i / rows) * patch_size_x4:int(
                                                                                     i / rows) * patch_size_x4 + patch_size_x4,
                                                                                 i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4]

        # x8
        LR8 = torch.zeros_like(img_LR8)+0.5
        # LR8 = img_LR8
        torch.cuda.empty_cache()
        for i in block_x8:
            LR8[:, :, int(i / rows) * patch_size_x8:int(i / rows) * patch_size_x8 + patch_size_x8,
            i % rows * patch_size_x8:i % rows * patch_size_x8 + patch_size_x8] = img_LR8[:, :,
                                                                                 int(i / rows) * patch_size_x8:int(
                                                                                     i / rows) * patch_size_x8 + patch_size_x8,
                                                                                 i % rows * patch_size_x8:i % rows * patch_size_x8 + patch_size_x8]

        source,_,_,_= checkpoint(union,(HR2-0.5)*2, (HR4-0.5)*2,(HR8-0.5)*2, [block_x2,block_x4,block_x8], use_reentrant=True)
        source=source/2+0.5
    else:
    #     # 整张超分
    #     # continue
        torch.cuda.empty_cache()
        _,source = checkpoint(SRnet4, (img1-0.5)*2, use_reentrant=True)
        source=source/2+0.5
    # LR = torch.cat([img_LR2[:, :, int(i / rows) * 16:int(i / rows) * 16 + 16, i % rows * 16:i % rows * 16 + 16] for i in range(block_nums)], 0)
    # HR = checkpoint(SRnet2.layer3, (LR-0.5)*2, use_reentrant=True)/2+0.5
    # j = 0
    # for i in range(block_nums):
    #     source[:, :, int(i / rows) * 32:int(i / rows) * 32 + 32, i % rows * 32:i % rows * 32 + 32] = HR[j]
    #     j += 1

    # source = source[:, :, :H, :W]
    # img1 = img1[:, :, :H, :W]


    # 展示图片
    # a = transforms.ToPILImage()(source[0])
    # a.show()

    # 处理lr
    i1 = source.cpu().detach().numpy()[0]
    i2 = img1.cpu().detach().numpy()[0]
    i1 = np.clip(i1, 0.0, 1.0)
    i2 = np.clip(i2, 0.0, 1.0)

    i1 = np.transpose(i1, (1, 2, 0))
    i2 = np.transpose(i2, (1, 2, 0))

    i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2BGR)
    # i2 = cv2.cvtColor(i2, cv2.COLOR_RGB2BGR)


    # cv2.namedWindow('asd', 0)
    # cv2.resizeWindow('asd', 640, 640)
    # cv2.imshow(winname='asd', mat=i1/255)
    # cv2.waitKey(2000)
    # i1*=255
    # index = 6433  # 24-test2k
    index = 16480  # 40-test2k
    # cv2.imwrite('F:/BMDS_output/contrast/' + "OmniSR_P2P_" + imgname,
    #             (i1 * 255)[int(index / rows) * 16:int(index / rows) * 16 + 256,
    #             index % rows * 16:index % rows * 16 + 256, :])

    # cv2.imwrite('F:\BMDS_output\MultiLevelBMDSR\OmniSR_P2P_test4k/' + imgname, i1*255)
    i1 = cv2.cvtColor(i1, cv2.COLOR_RGB2BGR)

    i1 = np.dot(i1, [65.481, 128.553, 24.966]) + 16  # RGB
    i2 = np.dot(i2, [65.481, 128.553, 24.966]) + 16

    # cv2.namedWindow('asd', 0)
    # cv2.resizeWindow('asd', 640, 640)
    # cv2.imshow(winname='asd', mat=i1/255)
    # cv2.waitKey(100000)


    if len(block_x2) >= 0:
       print(imgname,"PSNR:", psnr_get(i1, i2))
        # print(imgname, "SSIM:", ssim(i1, i2, channel_axis=2, data_range=255))
    block_imgnums += 1
    ssim1_sum += ssim(i1, i2, channel_axis=1, data_range=255)
    psnr += psnr_get(i1, i2)

    distance1 = checkpoint(lpips_model, source.cuda(), img1.cuda(), use_reentrant=True)
    distance += distance1.cpu().item()


# print('PSNR:', psnr1_sum / len(test_list), 'niqo:')
# print('SSIM:', ssim1_sum / len(test_list), 'niqo:')
print("time:",time_sum/100)
print('flops:',flops_sum/100/(10**9))
print('LPIPS:', distance / block_imgnums)
print('SSIM:', ssim1_sum / block_imgnums)
print('PSNR:%.3f/%.3f' % (psnr / block_imgnums, time.time() - t))
print('total scale: ',math.sqrt(HR_datasize_sum / SR_datasize_sum))
print(block_imgnums)

# distance1 = lpips_model(img / 255, img2 / 255)
# distance += distance1.cpu().item()
