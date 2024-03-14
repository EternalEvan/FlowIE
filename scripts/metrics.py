import pyiqa
import torch

# list all available metrics
print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



# For FID metric, use directory or precomputed statistics as inputs
# refer to clean-fid for more details: https://github.com/GaParmar/clean-fid
fid_metric = pyiqa.create_metric('fid')
score = fid_metric('/home/zyx/CodeFormer-master/results/celeba_512_validation_lq_0.5/restored_faces/', '/data1/zyx/FFHQ512/FFHQ_512/')
print(score)
#score = fid_metric('./ResultsCalibra/dist_dir/', dataset_name="FFHQ", dataset_res=1024, dataset_split="trainval70k")