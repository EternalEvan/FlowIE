from typing import Sequence, Dict, Union
import math
import time
import torch
import numpy as np
import cv2
from PIL import Image,ImageDraw
import torch.utils.data as data
import pdb
from utils.file import load_file_list,list_image_files
from utils.image import center_crop_arr, augment, random_crop_arr
from utils.degradation import (
    random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression
)

from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from os.path import join,isfile
import utils.process as process
from utils.utils import loadmat


import rawpy

def brush_stroke_mask(img, color=(255,255,255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        #pdb.set_trace()
     
        if img is not None: mask = img
        np.random.seed()
        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            #print(mask.szie)
            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

class CodeformerDataset(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        if min(*pil_img.size)!= self.out_size:
            pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)

class CodeformerDatasetLQ(data.Dataset):
    
    def __init__(
        self,
        hq_list: str,
        lq_list:str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDatasetLQ, self).__init__()
        self.hq_list = hq_list
        self.lq_list = lq_list
        self.hq_paths = load_file_list(hq_list)
        self.lq_paths = load_file_list(lq_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.hq_paths[index]
        lq_path = self.lq_paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        
        for _ in range(3):
            try:
                lq_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        if min(*pil_img.size)!= self.out_size:
            pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if self.crop_type == "center":
        #     pil_img_gt = center_crop_arr(pil_img, self.out_size)
        # elif self.crop_type == "random":
        #     pil_img_gt = random_crop_arr(pil_img, self.out_size)
        # else:
        #     pil_img_gt = np.array(pil_img)
        #     assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        if self.crop_type == "center":
            pil_img,lq_img = center_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = center_crop_arr(lq_img, self.out_size)
        elif self.crop_type == "random":
            pil_img,lq_img = random_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = random_crop_arr(lq_img, self.out_size)
            
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if min(*lq_img.size)!= self.out_size:
        #     lq_img = lq_img.resize((self.out_size,self.out_size),resample=Image.BOX)        
        img_gt = np.array(pil_img)
        img_lq = np.array(lq_img)
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        # img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # # BGR to RGB, [-1, 1]
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        img_lq = img_lq.permute((1,2,0)).numpy()
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
        
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        
        # # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.hq_paths)



class CodeformerDatasetLQ_from_dir(data.Dataset):
    
    def __init__(
        self,
        hq_list: str,
        lq_list:str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDatasetLQ_from_dir, self).__init__()
        self.hq_list = hq_list
        self.lq_list = lq_list
        self.hq_paths = load_file_list(hq_list)
        self.lq_paths = load_file_list(lq_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.hq_paths[index]
        lq_path = self.lq_paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        
        for _ in range(3):
            try:
                lq_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        if min(*pil_img.size)!= self.out_size:
            pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if self.crop_type == "center":
        #     pil_img_gt = center_crop_arr(pil_img, self.out_size)
        # elif self.crop_type == "random":
        #     pil_img_gt = random_crop_arr(pil_img, self.out_size)
        # else:
        #     pil_img_gt = np.array(pil_img)
        #     assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        if self.crop_type == "center":
            pil_img,lq_img = center_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = center_crop_arr(lq_img, self.out_size)
        elif self.crop_type == "random":
            pil_img,lq_img = random_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = random_crop_arr(lq_img, self.out_size)
            
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if min(*lq_img.size)!= self.out_size:
        #     lq_img = lq_img.resize((self.out_size,self.out_size),resample=Image.BOX)        
        img_gt = np.array(pil_img)
        img_lq = np.array(lq_img)
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        # img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # # BGR to RGB, [-1, 1]
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        img_lq = img_lq.permute((1,2,0)).numpy()
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
        
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        
        # # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.hq_paths)


class CodeformerDataset_Gray(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        color_jitter_prob: float,
        color_jitter_shift: float,
        color_jitter_pt_prob: float,
        gray_prob: float
    ) -> "CodeformerDataset":
        super(CodeformerDataset_Gray, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        
        self.color_jitter_prob = color_jitter_prob
        self.color_jitter_pt_prob = color_jitter_pt_prob
        self.color_jitter_shift = color_jitter_shift
        self.color_jitter_shift /= 255.
        # to gray
        self.gray_prob = gray_prob
    
    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img
    
    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img
    
    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        if min(*pil_img.size)!= self.out_size:
            pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        if self.crop_type == "center":
            pil_img_gt = center_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "random":
            pil_img_gt = random_crop_arr(pil_img, self.out_size)
        else:
            pil_img_gt = np.array(pil_img)
            assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        img_gt = (pil_img_gt[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        kernel = random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
            
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
            
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness =(0.5, 1.5)
            contrast =  (0.5, 1.5)
            saturation = ( 0, 1.5)
            hue = (-0.1, 0.1)
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        img_lq = img_lq.permute((1,2,0)).numpy()
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
        # resize to original size
        
        
        #pdb.set_trace()
        #print(img_lq.shape)
        #assert False
        
        
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        #img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
        #img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
        
        source = img_lq[..., ::-1].astype(np.float32)
        
        
            
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)



class CodeformerDataset_Mask(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset_Mask, self).__init__()
        self.file_list = file_list
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        img_gt = pil_img
        if pil_img.shape[0]!= self.out_size:
            img_gt = cv2.resize(pil_img,(self.out_size,self.out_size))
            #pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
       
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        #img_gt = img_gt[..., ::-1].astype(np.float32)
        
        # random horizontal flip
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        #print(img_gt.shape)

        img_lq = img_gt
        img_lq = np.asarray(brush_stroke_mask(Image.fromarray(img_gt,mode='RGB')))/255.0
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] /255.0 * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.paths)
    
    

class CodeformerDataset_Derain(data.Dataset):
    
    def __init__(
        self,
       
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        is_val=False
    ) -> "CodeformerDataset":
        super(CodeformerDataset_Derain,self).__init__()
        lq_paths = []
        hq_paths = []
        num = 0
        if is_val:
            img_dir = "/home/user001/zwl/data/Derain/Rain100L"
            lq_paths_0 = list_image_files(join(img_dir,'rainy'))
            for lq_path in lq_paths_0:
                hq_path = lq_path[:-18]+'no'+lq_path[-12:]
                lq_paths.append(lq_path)
                hq_paths.append(hq_path)
                print(hq_path)
        else:
            for img_dir in ['/home/user001/zwl/data/Derain/Rain12600','/home/user001/zwl/data/Derain/RainTrainH','/home/user001/zwl/data/Derain/RainTrainL']:
                lq_paths_0 = list_image_files(join(img_dir,'rainy_image'))
                for lq_path in lq_paths_0:
                    if img_dir[-5:] == '12600':
                        hq_path = lq_path.split('_')[0]+'_'+lq_path.split('_')[1]+'.jpg'
                        hq_path = hq_path.replace('rainy_image','ground_truth')
                        
                    else:
                    
                        hq_path = lq_path.replace('rainy_image','ground_truth')
                        hq_path = hq_path.split('-')[0][:-4]+'norain'+'-'+hq_path.split('-')[1]
                        #print(hq_path)
                        #assert False
                        
                    if isfile(hq_path):
                        lq_paths.append(lq_path)
                        hq_paths.append(hq_path)
                        num += 1
                        if is_val and num>100:
                            break
                    
        self.hq_paths = hq_paths
        self.lq_paths = lq_paths

        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.hq_paths[index]
        lq_path = self.lq_paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        
        for _ in range(3):
            try:
                lq_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if self.crop_type == "center":
        #     pil_img_gt = center_crop_arr(pil_img, self.out_size)
        # elif self.crop_type == "random":
        #     pil_img_gt = random_crop_arr(pil_img, self.out_size)
        # else:
        #     pil_img_gt = np.array(pil_img)
        #     assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        if self.crop_type == "center":
            pil_img,lq_img = center_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = center_crop_arr(lq_img, self.out_size)
        elif self.crop_type == "random":
            pil_img,lq_img = random_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = random_crop_arr(lq_img, self.out_size)
            
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if min(*lq_img.size)!= self.out_size:
        #     lq_img = lq_img.resize((self.out_size,self.out_size),resample=Image.BOX)        
        img_gt = np.array(pil_img)
        img_lq = np.array(lq_img)
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        # img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # # BGR to RGB, [-1, 1]
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        img_lq = img_lq.permute((1,2,0)).numpy()
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
        
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        
        # # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.hq_paths)



def pack_raw_bayer(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    white_point = 16383
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2,G1[1][0]:W:2],
                    im[B[0][0]:H:2,B[1][0]:W:2],
                    im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    
    return out

def compute_expo_ratio(input_fn, target_fn):        
    in_exposure = float(input_fn.split('_')[-1][:-5])
    gt_exposure = float(target_fn.split('_')[-1][:-5])
    ratio = min(gt_exposure / in_exposure, 300)
    return ratio

def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns

class CodeformerDataset_Enlight(data.Dataset):
    
    def __init__(
        self,

        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        is_val = False
    ) -> "CodeformerDataset":
        super(CodeformerDataset_Enlight, self).__init__()
     
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.pack_raw = pack_raw_bayer
        self.stage_out = 'srgb'
        
        lq_paths = []
        hq_paths = []
        self.pairs_id = read_paired_fns('/home/user001/zwl/data/Sony/Sony_train.txt')
        self.img_dir = "/home/user001/zwl/data/Sony"
        num = 0
        if is_val:
            
            for idx, pair in enumerate(self.pairs_id):
                lq_path = join(self.img_dir,'short',pair[0])
                hq_path = join(self.img_dir,'long',pair[1])
                lq_paths.append(lq_path)
                hq_paths.append(hq_path)
                #print(hq_path)
                num += 1
                if num>100:
                    break
        else:
            for idx, pair in enumerate(self.pairs_id):
                print(pair)
                lq_path = join(self.img_dir,'short',pair[0])
                hq_path = join(self.img_dir,'long',pair[1])
                lq_paths.append(lq_path)
                
                hq_paths.append(hq_path)
                print(hq_path)
             
                    
        self.hq_paths = hq_paths
        self.lq_paths = lq_paths
        
        self.target_dict = {}
        self.target_dict_aux = {}
        self.input_dict = {}

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.hq_paths[index]
        lq_path = self.lq_paths[index]
        success = False
        target_fn = gt_path.split('/')[-1]
        input_fn = lq_path.split('/')[-1]
        ratio = compute_expo_ratio(input_fn, target_fn)
        CRF = None
        ##############33
        # print('33')
        ''' rawData = open(lq_path ,'rb').read()
        
        if target_fn not in self.target_dict:
            with rawpy.imread(gt_path) as raw_target: 
                #target_image = self.pack_raw(raw_target)    
                #wb, ccm = process.read_wb_ccm(raw_target)
                if self.stage_out == 'srgb':
                    #for item in dir(raw_target.raw_image):
                    #    print(item)
                    #target_image = process.raw2rgb(target_image, raw_target, CRF)
                    target_image = raw_target.postprocess(
                    use_camera_wb=True,  # 是否使用拍摄时的白平衡值
                    use_auto_wb=False,
                    # half_size=True,  # 是否输出一半大小的图像，通过将每个2x2块减少到一个像素而不是进行插值来
                    exp_shift=3  # 修改后光线会下降，所以需要手动提亮，线性比例的曝光偏移。可用范围从0.25（变暗2级）到8.0（变浅3级）。
                    )

                self.target_dict[target_fn] = target_image
                #self.target_dict_aux[target_fn] = (wb, ccm)
            
        if input_fn not in self.input_dict:
            with rawpy.imread(lq_path) as raw_input:
                #input_image = self.pack_raw(raw_input) * ratio
                if self.stage_out == 'srgb':
                    #if self.gt_wb:
                    #    wb, ccm = self.target_dict_aux[target_fn]
                    #    input_image = process.raw2rgb_v2(input_image, wb, ccm, CRF)
                    #else:
                    input_image = raw_input.postprocess(
                    use_camera_wb=True,  # 是否使用拍摄时的白平衡值
                    use_auto_wb=False,
                    # half_size=True,  # 是否输出一半大小的图像，通过将每个2x2块减少到一个像素而不是进行插值来
                    exp_shift=3  # 修改后光线会下降，所以需要手动提亮，线性比例的曝光偏移。可用范围从0.25（变暗2级）到8.0（变浅3级）。
                    )
                self.input_dict[input_fn] = input_image'''
        ####
        for _ in range(3):
            try:
                
                if target_fn not in self.target_dict:
                    with rawpy.imread(gt_path) as raw_target: 
                        target_image = self.pack_raw(raw_target)    
                        #wb, ccm = process.read_wb_ccm(raw_target)
                        if self.stage_out == 'srgb':
                            target_image = raw_target.postprocess(
                                use_camera_wb=True,  # 是否使用拍摄时的白平衡值
                                use_auto_wb=False,
                                # half_size=True,  # 是否输出一半大小的图像，通过将每个2x2块减少到一个像素而不是进行插值来
                                exp_shift=3  # 修改后光线会下降，所以需要手动提亮，线性比例的曝光偏移。可用范围从0.25（变暗2级）到8.0（变浅3级）。
                                )
                        self.target_dict[target_fn] = Image.fromarray(target_image,mode='RGB')
                        #self.target_dict_aux[target_fn] = (wb, ccm)
                    
                if input_fn not in self.input_dict:
                    with rawpy.imread(lq_path) as raw_input:
                        input_image = self.pack_raw(raw_input) * ratio
                        if self.stage_out == 'srgb':
                            #if self.gt_wb:
                            #    wb, ccm = self.target_dict_aux[target_fn]
                            #    input_image = process.raw2rgb_v2(input_image, wb, ccm, CRF)
                            #else:
                            input_image = raw_input.postprocess(
                                use_camera_wb=True,  # 是否使用拍摄时的白平衡值
                                use_auto_wb=False,
                                # half_size=True,  # 是否输出一半大小的图像，通过将每个2x2块减少到一个像素而不是进行插值来
                                exp_shift=3  # 修改后光线会下降，所以需要手动提亮，线性比例的曝光偏移。可用范围从0.25（变暗2级）到8.0（变浅3级）。
                                )
                        self.input_dict[input_fn] = Image.fromarray(input_image,mode='RGB')
                
                #pil_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        
        lq_img = self.input_dict[input_fn]
        pil_img = self.target_dict[target_fn]
        #print(lq_img.size)
        #print(type(lq_img))
        #(wb, ccm) = self.target_dict_aux[target_fn]
            
            #pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
       
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
            
        #img_gt = img_gt[..., ::-1].astype(np.float32)
        if self.crop_type == "center":
            pil_img,lq_img = center_crop_arr(pil_img, self.out_size,lq_image=lq_img)
        # random horizontal flip
        
        elif self.crop_type == "random":
            pil_img,lq_img = random_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            
        img_gt = np.array(pil_img)
        img_lq = np.array(lq_img)
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        #print(img_gt.shape)
      
        #img_lq = np.asarray(brush_stroke_mask(Image.fromarray(img_gt,mode='RGB')))/255.0
        # BGR to RGB, [-1, 1]
        target = (img_gt[..., ::-1] /255.0 * 2 - 1).astype(np.float32)
        # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        
        return dict(jpg=target, txt="", hint=source,imgname=target_fn)

    def __len__(self) -> int:
        return len(self.lq_paths)



class CodeformerDataset_Dehaze(data.Dataset):
    def __init__(
        self,
       
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int],
        is_val=False
    ) -> "CodeformerDataset":
        super(CodeformerDataset_Dehaze,self).__init__()
        lq_paths = []
        hq_paths = []
        num = 0
        if is_val:
            img_dir = "/home/user001/zwl/data/RESIDE/test/test/"
            lq_paths_0 = list_image_files(join(img_dir,'hazy'))
            for lq_path in lq_paths_0:
                hq_path = lq_path.replace('hazy','GT')
                lq_paths.append(lq_path)
                hq_paths.append(hq_path)
                print(lq_path)
                print(hq_path)
                num += 1
                if num>100:
                    break
        else:
            img_dir = "/home/user001/zwl/data/RESIDE/train/train/"
            lq_paths_0 = list_image_files(join(img_dir,'hazy'))
            for lq_path in lq_paths_0:
                hq_path = lq_path.replace('hazy','GT')
                lq_paths.append(lq_path)
                hq_paths.append(hq_path)
                
              
                    
        self.hq_paths = hq_paths
        self.lq_paths = lq_paths

        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        self.use_hflip = use_hflip
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
     
        gt_path = self.hq_paths[index]
        lq_path = self.lq_paths[index]
        success = False
        imgname = gt_path.split('/')[-1]

        for _ in range(3):
            try:
                pil_img = Image.open(gt_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        
        for _ in range(3):
            try:
                lq_img = Image.open(lq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {gt_path}"
        #if min(*pil_img.size)!= self.out_size:
        #    pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if self.crop_type == "center":
        #     pil_img_gt = center_crop_arr(pil_img, self.out_size)
        # elif self.crop_type == "random":
        #     pil_img_gt = random_crop_arr(pil_img, self.out_size)
        # else:
        #     pil_img_gt = np.array(pil_img)
        #     assert pil_img_gt.shape[:2] == (self.out_size, self.out_size)
        if self.crop_type == "center":
            pil_img,lq_img = center_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = center_crop_arr(lq_img, self.out_size)
        elif self.crop_type == "random":
            pil_img,lq_img = random_crop_arr(pil_img, self.out_size,lq_image=lq_img)
            #lq_img = random_crop_arr(lq_img, self.out_size)
            
        # if min(*pil_img.size)!= self.out_size:
        #     pil_img = pil_img.resize((self.out_size,self.out_size),resample=Image.BOX)
        # if min(*lq_img.size)!= self.out_size:
        #     lq_img = lq_img.resize((self.out_size,self.out_size),resample=Image.BOX)        
        img_gt = np.array(pil_img)
        img_lq = np.array(lq_img)
        img_gt = (img_gt[..., ::-1] / 255.0).astype(np.float32)
        img_lq = (img_lq[..., ::-1] / 255.0).astype(np.float32)
        
        # random horizontal flip
        # img_gt = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=False)
        h, w, _ = img_gt.shape
        
        # ------------------------ generate lq image ------------------------ #
        # blur
        # kernel = random_mixed_kernels(
        #     self.kernel_list,
        #     self.kernel_prob,
        #     self.blur_kernel_size,
        #     self.blur_sigma,
        #     self.blur_sigma,
        #     [-math.pi, math.pi],
        #     noise_range=None
        # )
        # img_lq = cv2.filter2D(img_gt, -1, kernel)
        # # downsample
        # scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        # img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # # noise
        # if self.noise_range is not None:
        #     img_lq = random_add_gaussian_noise(img_lq, self.noise_range)
        # # jpeg compression
        # if self.jpeg_range is not None:
        #     img_lq = random_add_jpg_compression(img_lq, self.jpeg_range)
        
        # # resize to original size
        # img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # # BGR to RGB, [-1, 1]
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.
        img_lq = img_lq.permute((1,2,0)).numpy()
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR)
        
        target = (img_gt[..., ::-1] * 2 - 1).astype(np.float32)
        
        # # BGR to RGB, [0, 1]
        source = img_lq[..., ::-1].astype(np.float32)
        return dict(jpg=target, txt="", hint=source,imgname=imgname)

    def __len__(self) -> int:
        return len(self.hq_paths)