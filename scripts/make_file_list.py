import sys
sys.path.append(".")
import os
from argparse import ArgumentParser

from utils.file import list_image_files
import cv2
import pdb

parser = ArgumentParser()
parser.add_argument("--img_folder", type=str, required=True)
parser.add_argument("--val_size", type=int, default=0)
parser.add_argument("--save_folder", type=str, required=True)
parser.add_argument("--follow_links", action="store_true")
args = parser.parse_args()

files = list_image_files(
    args.img_folder, exts=(".jpg", ".png", ".jpeg"), follow_links=args.follow_links,
    log_progress=True, log_every_n_files=10000
)

print(f"find {len(files)} images in {args.img_folder}")
assert args.val_size < len(files)

# val_files = files[:args.val_size]
# train_files = files[args.val_size:]
val_files = files
print(len(val_files))
valid_files = []
for i,path in enumerate(val_files):
    #img = cv2.imread(path)
    #h,w,_ = img.shape
    #if h>=512 and w>=512:
    
    #print(path)
    # if i <69000:
    #     continue
    valid_files.append(path)
    #pdb.set_trace()

print('Total files:{}'.format(len(valid_files)))


os.makedirs(args.save_folder, exist_ok=True)

# with open(os.path.join(args.save_folder, "train.list"), "w") as fp:
#     for file_path in train_files:
#         fp.write(f"{file_path}\n")

with open(os.path.join(args.save_folder, "1024.list"), "w") as fp:
    for file_path in valid_files:
        fp.write(f"{file_path}\n")
