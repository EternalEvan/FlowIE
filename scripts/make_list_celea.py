import sys
sys.path.append(".")
import os
from argparse import ArgumentParser
import pandas as pd

s_img = '/data1/zyx/CelebAMask-HQ/CelebA-HQ-img'



train_count = 0
test_count = 0
val_count = 0
image_list = pd.read_csv('/data1/zyx/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt', delim_whitespace=True, header=None)
train_paths = []
val_paths = []
test_paths = []

for idx, x in enumerate(image_list.loc[:, 1]):
    print (idx, x)
    if idx == 0:
        continue
   
    x = int(x)
    if x >= 162771 and x < 182638:
        img_path = os.path.join(s_img, str(idx-1)+'.jpg')
        val_paths.append(img_path) 
        val_count += 1

    elif x >= 182638:
        img_path = os.path.join(s_img, str(idx-1)+'.jpg')
        test_paths.append(img_path)
        test_count += 1 
    else:
        img_path = os.path.join(s_img, str(idx-1)+'.jpg')
        train_paths.append(img_path)
        train_count += 1  

print (train_count + test_count + val_count)




save_folder = '/data1/zyx/FFHQ512'

# with open(os.path.join(save_folder, "train.list"), "a") as fp:
#     for file_path in train_paths:
#         fp.write(f"{file_path}\n")

# with open(os.path.join(save_folder, "val.list"), "a") as fp:
#     for file_path in val_paths:
#         fp.write(f"{file_path}\n")

with open(os.path.join(save_folder, "test.list"), "w") as fp:
    for file_path in test_paths:
        fp.write(f"{file_path}\n")