import numpy as np
import math
import os
import cv2
import torch
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from lpips_pytorch import LPIPS, lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
import pytorch_ssim
from argparse import ArgumentParser
from tqdm import tqdm
import pdb


import pyiqa
# def psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2 )
#     if mse == 0:
#         return 100
#     return 20 * math.log10(255.0 / math.sqrt(mse))
def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    
    #pdb.set_trace()
    return 10. * np.log10(255.*255.0 /mse)

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
def calculate_ssim(img1, img2):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
 
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)


    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

def main(path1, path2, type="all"):
    loss_1 = []
    loss_2 = []
    loss_3 = [] 
    loss_4 = []
    length = 0
    
    if type=="lpips" or type == "all":
        lpips = LPIPS().cuda()
    if type=="ssim" or type == "all":
        ssim = pytorch_ssim.SSIM(window_size = 11).cuda()

    #iqa_metric = pyiqa.create_metric(type, test_y_channel=False, color_space='ycbcr').cuda()
    lpips_metric = pyiqa.create_metric('lpips').cuda()
    ssim_metric = pyiqa.create_metric('ssim').cuda()
    musiq_metric = pyiqa.create_metric('musiq').cuda()

    for idx ,img in tqdm(enumerate(os.listdir(path1)),total=len(os.listdir(path1))):
        imgpath1 = os.path.join(path1,img)
        imgpath2 = os.path.join(path2,img)
        imgpath2 = imgpath2[:-3]+'png'
        
        #print(imgpath1)
        #img1 = cv2.imread(imgpath1).astype(np.float64)
      
        
        #img2 = cv2.imread(imgpath2).astype(np.float64)


        # mean_l = []
        # std_l = []
        # for j in range(3):
        #     mean_l.append(np.mean(img2[:, :, j]))
        #     std_l.append(np.std(img2[:, :, j]))
        # for j in range(3):
        #     # correct twice
        #     mean = np.mean(img1[:, :, j])
        #     img1[:, :, j] = img1[:, :, j] - mean + mean_l[j]
        #     std = np.std(img1[:, :, j])
        #     img1[:, :, j] = img1[:, :, j] / std * std_l[j]

        #     mean = np.mean(img1[:, :, j])
        #     img1[:, :, j] = img1[:, :, j] - mean + mean_l[j]
        #     std = np.std(img1[:, :, j])
        #     img1[:, :, j] = img1[:, :, j] / std * std_l[j]
        # img1 = cv2.resize(img1,(256,256))
        # img2 = cv2.resize(img2,(256,256))
        # if img1.shape != img2.shape:
        #     if img1.shape[0]< img2.shape[0]:
        #         img2 = cv2.resize(img2,img1.shape[:2])
        #     else:
        #         img1 = cv2.resize(img1,img2.shape[:2])
        if type=="psnr":
            psnr_score = iqa_metric(imgpath1,imgpath2)
            # loss_1 += psnr(img1,img2,data_range=255.0)
            loss_1.append(psnr_score.cpu().numpy())
        elif type=="lpips":
            lpips_score = lpips_metric(imgpath1,imgpath2)
            loss_2.append(lpips_score.cpu().numpy())
        elif type=="ssim":
            ssim_score = ssim_metric(imgpath1,imgpath2)
            loss_3.append(ssim_score.cpu().numpy())
        elif type=="musiq":
            musiq_score = musiq_metric(imgpath1)
            loss_4.append(musiq_score.cpu().numpy())
        elif type == "all":
            loss_1 += psnr(img1,img2)
            loss_2 += lpips(transforms.ToTensor()(img1).cuda(),transforms.ToTensor()(img2).cuda())
            loss_3 += ssim(transforms.ToTensor()(img1).unsqueeze(0).cuda(),transforms.ToTensor()(img2).unsqueeze(0).cuda())
        # loss += criterion(transforms.ToTensor()(img1).cuda(),transforms.ToTensor()(img2).cuda())
        # loss += criterion(transforms.ToTensor()(img1).unsqueeze(0).cuda(),transforms.ToTensor()(img2).unsqueeze(0).cuda())
        length +=1
    if type=="psnr" or type == "all":
        print("psnr↑",np.mean(loss_1))
    if type=="lpips" or type == "all":
        print("lpips↓",np.mean(loss_2))
    if type=="ssim" or type == "all":
        print("ssim↑",np.mean(loss_3))
    if type=="musiq" or type == "all":
        print("musiq↑",np.mean(loss_4))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input1", type=str, required=True)
    parser.add_argument("--input2", type=str, required=True)
    parser.add_argument("--type", type=str, default="all")
    args = parser.parse_args()
    main(args.input1, args.input2, args.type)


'''
DiffBIR
psnr↑ 31.14
lpips↓ 0.2063
ssim↑ 0.6731

midd
psnr↑ 30.87
lpips↓ 0.2046
ssim↑ 0.6719

final
psnr↑ 31.17
lpips↓ 0.2248
ssim↑ 0.7220

'''