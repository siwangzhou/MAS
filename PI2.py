import os
import torch
import pyiqa
from tqdm import tqdm


class NIQE_metric(object):
    def __init__(self) -> None:
        device = torch.device("cuda")
        self.niqe_metric = pyiqa.create_metric('pi', device=device)

    def __call__(self, img_path):
        ''' calculate NIQE value
        Args:
            img_path (str): image dir
        '''
        niqe_value = self.niqe_metric(img_path)  # crop_border=0
        # niqe_value = calculate_niqe(img, crop_border=0)
        return niqe_value.item()


# photo_path = 'F:\BMDS_output\OmniSR_union\BMS-Omni-2k'
photo_path = 'F:\BMDS_output\MultiLevelBMDSR\EDSR_P2P_test4k'
test_list = os.listdir(photo_path)

cul_NIQE=NIQE_metric()
NIQE_sum=0
for imgname in tqdm(test_list):
    # cul_NIQE = NIQE_metric()
    # for imgname in test_list:
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    photo_str = photo_path + '/' + imgname
    torch.cuda.empty_cache()
    NIQE=cul_NIQE(photo_str)
    NIQE_sum+=NIQE
    print(NIQE)
    # del cul_NIQE
print(photo_path,'   PI: ',NIQE_sum/len(test_list))