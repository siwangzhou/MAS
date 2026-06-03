import sys
sys.path.append('D:\code\Python\project\mechine Learning/block_rescaling_32/network')
sys.path.append('D:\code\Python\project\mechine Learning/block_rescaling_32/method2/uniontrain/')
from utils import *
import copy
from skimage.metrics import structural_similarity as ssim
import lpips
import time
from select_rate import sampling_rate_allocate3_salient,sampling_rate_allocate3_show,sampling_rate_allocate3,sampling_rate_allocate3_ablation_study
import sys
# from network.hat_model.hat_arch import HAT
# sys.path.append('D:\code\Python\project\mechine Learning\Jincancan\RCM-master/')
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch
from network.ops.edsr.edsr import EDSR
from network.ops.OSAG import OSAG
# from method2.uniontrain.method2_union_net import OmniSR_P2P_union2
from method2.uniontrain.method2_union_net_val import EDSR_union2_val,EDSR_union2_wo_db_val
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

SRnet2 = EDSR(2).cuda()
chpoint = torch.load("D:/code/Python/project/mechine Learning/block_rescaling_32/SRNet/EDSR_quant_x2_norm.pt")
SRnet2.load_state_dict(chpoint['net'])

SRnet4 = EDSR(4).cuda()
chpoint = torch.load("D:/code/Python/project/mechine Learning/block_rescaling_32/SRNet/EDSR_quant_x4_norm.pt")
SRnet4.load_state_dict(chpoint['net'])

SRnet8 = EDSR(8).cuda()
chpoint = torch.load("D:/code/Python/project/mechine Learning/block_rescaling_32/SRNet/EDSR_quant_x8_norm.pt")
SRnet8.load_state_dict(chpoint['net'])

for name, param in SRnet4.named_parameters():
    param.requires_grad = False

# union=EDSR_union2_wo_db_val().cuda()
# chpoint=torch.load('D:\code\Python\project\mechine Learning/block_rescaling_32\method2/uniontrain/net/EDSR_union2_wo_db/EDSR_union2_wo_db.pt')
union=EDSR_union2_val().cuda()
chpoint=torch.load('D:\code\Python\project\mechine Learning/block_rescaling_32\method2/uniontrain/net/EDSR_union2/EDSR_union2.pt')

union.load_state_dict(chpoint['net'])
union.eval()

# Set5
# photo_path = 'D:\code\Python\project\mechine Learning\Data\Set5\Set5\original'
# photo_path='D:\code\Python\project\mechine Learning\Data\Set14\Set14\original'
# photo_path = 'D:\code\Python\project\mechine Learning\Data\BSDS100\original'
# str = 'D:\code\Python\project\mechine Learning\Data/Urban100/HR'
# str='D:\code\Python\project\mechine Learning\Data\BSD68\BSD68'
photo_path = 'D:\code\Python\project\mechine Learning\Data\DIV2K\DIV2K_valid_HR'
photo_path = 'D:\code\Python\project\mechine Learning\Data/test2k'
# photo_path = 'D:\code\Python\project\mechine Learning\Data/test4k'


params = list(SRnet4.parameters())
k = 0
for i in params:
    l = 1
    # print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    # print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k/1024/1024))

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
time_sum=0
time_temp=0
t = time.time()
# test_list=test_list[55:56]      #Multi-level
# test_list=test_list[47:48]      #Multi-level
# test_list=test_list[43:44]       #Multi-level
# test_list=test_list[39:40]       #Multi-level

# test_list=test_list[47:48]       #block-artificial
# test_list=test_list[15:16]       #block-artificial

# test_list=test_list[38:39]
print(test_list)

from thop import profile
# test_list=test_list[55:56]
# test_list=test_list[47:48]
# test_list=test_list[40:41]
# test_list=test_list[48:49]
# test_list=test_list[52:53]
# test_list =["C:/Users\Admin\Desktop/0004.png"]
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

    # img_BICUBIC_x4 = copy.deepcopy(img_BICUBIC)
    # img_BICUBIC_x2 = copy.deepcopy(img_BICUBIC)
    img_BICUBIC_x4 = img_LR4.resize((int(w), int(h)), Image.BICUBIC)
    img_BICUBIC_x4 = transforms.ToTensor()(img_BICUBIC_x4).cuda()
    img_BICUBIC_x8 = img_LR8.resize((int(w), int(h)), Image.BICUBIC)
    img_BICUBIC_x8=transforms.ToTensor()(img_BICUBIC_x8).cuda()

    img_BICUBIC = img_LR4.resize((w, h), Image.BICUBIC)

    img_LR4 = transforms.ToTensor()(img_LR4).unsqueeze(0).cuda()
    img_LR2 = transforms.ToTensor()(img_LR2).unsqueeze(0).cuda()
    img_LR8 = transforms.ToTensor()(img_LR8).unsqueeze(0).cuda()
    img_BICUBIC = transforms.ToTensor()(img_BICUBIC).unsqueeze(0)[:, :, :H, :W]

    # flops_temp, params = profile(union_net, inputs=([block_x2, block_x4, block_x8], (LR2 - 0.5) * 2, (LR4 - 0.5) * 2,
    #                                 (LR8 - 0.5) * 2, int(block_nums / rows),
    #                                 rows, ))


    torch.cuda.empty_cache()
    t1 = time.time()
    # temp_x4=checkpoint(SRnet4, (img_LR4.cuda()-0.5)*2, use_reentrant=True)
    # temp_x4=temp_x4/2+0.5
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
    # break
    # block_x2, block_x4, block_x8 = sampling_rate_allocate3_ablation_study((img_BICUBIC_x4).cpu(), img1)
    print(len(block_x2),len(block_x4),len(block_x8),time.time()-t1)

    # get LRimage

    HR_datasize = h * w
    scale_datasize=0
    for i in block_x2:
        scale_datasize+=patch_size_x2**2
    for i in block_x4:
        scale_datasize+=patch_size_x4**2
    for i in block_x8:
        scale_datasize+=patch_size_x8**2
    #
    scale_factor = math.sqrt(HR_datasize / scale_datasize)
    # print(scale_factor)
    HR_datasize_sum += HR_datasize
    SR_datasize_sum += scale_datasize

    time_sum+=time.time()-t1
    # print(time_sum)
    # continue

    source = torch.zeros_like(img_BICUBIC)


    if len(block_x2) >= 0:
        # 分块超分
        # x2
        LR2 = torch.zeros_like(img_LR2)
        # LR2 = img_LR2
        torch.cuda.empty_cache()
        for i in block_x2:
            LR2[:, :, int(i / rows) * patch_size_x2:int(i / rows) * patch_size_x2 + patch_size_x2, i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2]=img_LR2[:, :, int(i / rows) * patch_size_x2:int(i / rows) * patch_size_x2 + patch_size_x2, i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2]

        # x4
        LR4 = torch.zeros_like(img_LR4)
        # LR4 = img_LR4
        torch.cuda.empty_cache()
        for i in block_x4:
            LR4[:, :, int(i / rows) * patch_size_x4:int(i / rows) * patch_size_x4 + patch_size_x4,
            i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4] = img_LR4[:, :,
                                                                                 int(i / rows) * patch_size_x4:int(
                                                                                     i / rows) * patch_size_x4 + patch_size_x4,
                                                                                 i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4]
        # x8
        LR8 = torch.zeros_like(img_LR8)
        torch.cuda.empty_cache()
        for i in block_x8:
            LR8[:, :, int(i / rows) * patch_size_x8:int(i / rows) * patch_size_x8 + patch_size_x8,
            i % rows * patch_size_x8:i % rows * patch_size_x8 + patch_size_x8] = img_LR8[:, :,
                                                                                 int(i / rows) * patch_size_x8:int(
                                                                                     i / rows) * patch_size_x8 + patch_size_x8,
                                                                                 i % rows * patch_size_x8:i % rows * patch_size_x8 + patch_size_x8]
        LR4_t = copy.deepcopy(LR4).cpu()
        LR2=LR2.cpu()
        LR8=LR8.cpu()
        ti1=time.time()
        # a=model(b)
        # print('%.20f' % (time.time()-ti1))

        # LR2_4=transforms.ToPILImage()(LR4[0])
        # LR2_4 = LR2_4.resize((int(w / 4), int(h / 4)), Image.BICUBIC)
        # LR2_4=transforms.ToTensor()(LR2_4).unsqueeze(0)
        #
        # LR8=transforms.ToPILImage()(LR8[0])
        # LR8 = LR8.resize((int(w / 4), int(h / 4)), Image.BICUBIC)
        # LR8_4=transforms.ToTensor()(LR8).unsqueeze(0)

        LR2_4 = F.interpolate(LR2, scale_factor=1/2, mode='bicubic')
        LR8_4 = F.interpolate(LR8, scale_factor=2, mode='bicubic')
        # print(LR2_4.shape,LR8_4.shape,LR4.shape)
        # LR4_t = copy.deepcopy(LR4).cpu()
        # a=1+1
        LR4_t+=LR8_4+LR2_4
        # for i in block_x4:
        #     LR4_t[:, :, int(i / rows) * patch_size_x2:int(i / rows) * patch_size_x2 + patch_size_x2,
        #     i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2] = LR2_4[:, :,
        #                                                                          int(i / rows) * patch_size_x2:int(
        #                                                                              i / rows) * patch_size_x2 + patch_size_x2,
        #                                                                          i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2]
        # for i in block_x8:
        #     LR4_t[:, :, int(i / rows) * patch_size_x2:int(i / rows) * patch_size_x2 + patch_size_x2, i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2]=LR8_4[:, :, int(i / rows) * patch_size_x2:int(i / rows) * patch_size_x2 + patch_size_x2, i % rows * patch_size_x2:i % rows * patch_size_x2 + patch_size_x2]

        # for i in block_x2:
        #     LR4_t[:, :, int(i / rows) * patch_size_x4:int(i / rows) * patch_size_x4 + patch_size_x4,
        #     i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4] = LR2_4[:, :,
        #                                                                          int(i / rows) * patch_size_x4:int(
        #                                                                              i / rows) * patch_size_x4 + patch_size_x4,
        #                                                                          i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4]
        # for i in block_x8:
        #     LR4_t[:, :, int(i / rows) * patch_size_x4:int(i / rows) * patch_size_x4 + patch_size_x4,
        #     i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4] = LR8_4[:, :,
        #                                                                          int(i / rows) * patch_size_x4:int(
        #                                                                              i / rows) * patch_size_x4 + patch_size_x4,
        #                                                                          i % rows * patch_size_x4:i % rows * patch_size_x4 + patch_size_x4]
        time_temp+=time.time()-ti1
        print('%.20f' % (time.time()-ti1))
        # LR4_t = transforms.ToPILImage()(img_LR4[0])
        # LR4_t = LR4_t.resize((int(w / 2), int(h / 2)), Image.BICUBIC)
        # LR4_t = transforms.ToTensor()(LR4_t).unsqueeze(0)


        i1 = LR4_t.cpu().detach().numpy()[0]
        i2 = img_LR4.cpu().detach().numpy()[0]

        i1 = np.clip(i1, 0.0, 1.0)
        i2 = np.clip(i2, 0.0, 1.0)

        i1 = np.transpose(i1, (1, 2, 0))
        i2 = np.transpose(i2, (1, 2, 0))

        i1 = np.dot(i1, [65.481, 128.553, 24.966]) + 16  # RGB
        i2 = np.dot(i2, [65.481, 128.553, 24.966]) + 16

        psnr += psnr_get(i1, i2)
        block_imgnums += 1
        print("PSNR:", psnr_get(i1, i2))
        continue
        #without union
        # with torch.no_grad():
        #     SR2 = SRnet2((LR2 - 0.5) * 2) / 2 + 0.5
        #     SR4 = SRnet4((LR4 - 0.5) * 2) / 2 + 0.5
        #     SR8 = SRnet8((LR8 - 0.5) * 2) / 2 + 0.5
        # for i in block_x8:
        #     source[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
        #     i % rows * patch_size:i % rows * patch_size + patch_size] = SR8[:, :,
        #                                                                          int(i / rows) * patch_size:int(
        #                                                                              i / rows) * patch_size + patch_size,
        #                                                                          i % rows * patch_size:i % rows * patch_size + patch_size]
        # for i in block_x4:
        #     source[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
        #     i % rows * patch_size:i % rows * patch_size + patch_size] = SR4[:, :,
        #                                                                          int(i / rows) * patch_size:int(
        #                                                                              i / rows) * patch_size + patch_size,
        #                                                                          i % rows * patch_size:i % rows * patch_size + patch_size]
        # for i in block_x2:
        #     source[:, :, int(i / rows) * patch_size:int(i / rows) * patch_size + patch_size,
        #     i % rows * patch_size:i % rows * patch_size + patch_size] = SR2[:, :,
        #                                                                          int(i / rows) * patch_size:int(
        #                                                                              i / rows) * patch_size + patch_size,
        #                                                                          i % rows * patch_size:i % rows * patch_size + patch_size]

        #Unoin
        source= checkpoint(union,(LR2-0.5)*2, (LR4-0.5)*2,(LR8-0.5)*2, [block_x2,block_x4,block_x8], use_reentrant=True)
        source=source/2+0.5
    else:
    #     # 整张超分
    #     # continue
        torch.cuda.empty_cache()
        source = checkpoint(SRnet4, (img_LR4-0.5)*2, use_reentrant=True)/2+0.5
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
    # i1=2*torch.abs(img_BICUBIC_x4.cuda()-img1.cuda()).cpu().detach().numpy()[0]      #Confidence Map
    # i1 = SR8.cpu().detach().numpy()[0]                                           #LR_Blocks

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

    # index=1000    # Multi-level 2k-59 && 52
    # index = 9000  # Multi-level 2k-51 &&
    # index = 9020  # Multi-level 2k-48 &&
    # index = 16000  # Multi-level 2k-44 &&
    index = 9050  # Block-artificial 2k-51 &&
    # index = 12920  # Block-artificial 2k-21 &&

    # cv2.imwrite('F:/temp/ConfidenceMap_' + imgname, i1*255)
    # cv2.imwrite('F:/temp/' + "noJoint_noDB"+imgname, (i1 * 255)[int(index / rows) * 16:int(index / rows) * 16 + 256, index % rows * 16:index % rows * 16 + 256,:])
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
    # block_imgnums += 1
    # ssim1_sum += ssim(i1, i2, channel_axis=2, data_range=255)
    psnr += psnr_get(i1, i2)


# print('PSNR:', psnr1_sum / len(test_list), 'niqo:')
# print('SSIM:', ssim1_sum / len(test_list), 'niqo:')
# print('LPIPS:', distance / block_imgnums)
# print('SSIM:', ssim1_sum / block_imgnums)
print('total scale: ',math.sqrt(HR_datasize_sum / SR_datasize_sum))
print("time:",time_sum/100)
print("time:",time_temp/100)
# time_temp
print('PSNR:%.3f/%.3f' % (psnr / block_imgnums, time.time() - t))
print('total scale: ',math.sqrt(HR_datasize_sum / SR_datasize_sum))
print(block_imgnums)

# distance1 = lpips_model(img / 255, img2 / 255)
# distance += distance1.cpu().item()
