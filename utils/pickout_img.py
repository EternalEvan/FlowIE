import os
import shutil

if __name__ == "__main__":
    gttest = "outputs/gtcelebamaskhq"
    os.makedirs(gttest,exist_ok=True)
    with open("/data1/zyx/FFHQ512/test.list", 'r') as f:
        lines = f.readlines()
    for line in lines:
        path = line[:-1]
        name = path.split('/')[-1]
        shutil.copy(path,os.path.join(gttest,name))
    
