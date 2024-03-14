import cv2  
import numpy as np
import os
from os.path import join
import pdb
def compute_rainy_layer(derained_image_path, groundtruth_image_path, output_image_dir,filename):  
    # Load derained and groundtruth images  
    derained_image = cv2.imread(derained_image_path, cv2.IMREAD_GRAYSCALE) 
    
    gt_image = cv2.imread(groundtruth_image_path, cv2.IMREAD_GRAYSCALE)
    w,h = derained_image.shape[0],derained_image.shape[1]
    gt_image = cv2.resize(gt_image,(h,w))

    # Apply thresholding to groundtruth image  
   # Calculate the difference between the GT and result images  
    difference_image = gt_image - derained_image
    #print(difference_image[:9,:9])
    #pdb.set_trace()
    # Apply a threshold to obtain the rainy layer  
    threshold = 20  # You can adjust this value based on your application 
    is_rainy =  ((difference_image > threshold) * (difference_image <200))
    rainy_layer = np.where(is_rainy, 1, 0)
    # Save the rainy layer  
    output_image_path = join(output_image_dir,filename)
    cv2.imwrite(output_image_path, rainy_layer * 255)
    print(output_image_path)
    #cv2.imwrite(output_image_path, normalized_difference)

if __name__ == "__main__":  
    derained_image_path = "/home/user001/zwl/zyx/Diffbir/outputs/swin_derain/rain-001.png"
    #derained_image_path = "/home/user001/zwl/zyx/Pretrained-IPT/experiment/results/ipt/results-DIV2K/rain-001_x1_SR.png"  
    groundtruth_image_path = "/home/user001/zwl/data/Derain/Rain100L/rainy/rain-001.png"  
    output_image_dir ="/home/user001/zwl/zyx/RCDNet-master/RCDNet_code/for_syn/experiment/RCDNet_test/results//rainy" # "/home/user001/zwl/zyx/Pretrained-IPT/experiment/results/ipt/results-DIV2K/rainy/"
    os.makedirs(output_image_dir,exist_ok=True)
    derained_image_dir = "/home/user001/zwl/zyx/RCDNet-master/RCDNet_code/for_syn/experiment/RCDNet_test/results/"
    groundtruth_image_dir = "/home/user001/zwl/data/Derain/Rain100L/rainy"
    for filename in os.listdir(derained_image_dir):
        if filename[-3:] != 'png':
            continue
        derain_image_path = join(derained_image_dir,filename)
        groundtruth_image_path = join(groundtruth_image_dir,filename)
        compute_rainy_layer(derain_image_path, groundtruth_image_path, output_image_dir,filename)  
