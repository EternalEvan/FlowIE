# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
from argparse import ArgumentParser
import cv2
import os
#os.environ['CUDA_VISIBLE_DEVICE'] = '7'
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from utils.common import instantiate_from_config, load_state_dict
from torchvision.utils import save_image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def main() -> None:

    # dist.init_process_group(backend='nccl', init_method='env://')
    # rank = torch.cuda.set_device(dist.get_rank()) 
    # device = torch.device('cuda', rank)

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--watch_step", action='store_true')
    # parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
    model_config = OmegaConf.load(config.model.config)
    #print(model_config)
    model_config['params']['output'] = args.output
    
    data_module = instantiate_from_config(config.data)
    data_module.setup(stage="fit")
    model = instantiate_from_config(model_config)
    if config.model.get("resume"):
        model_dict = torch.load(config.model.resume, map_location="cpu")
        model_dict = model_dict['state_dict'] if 'state_dict' in model_dict.keys() else model_dict
        a,b = model.load_state_dict(model_dict, strict=False)
        print("missing_keys:",a)
        print("unexpected_keys:",b)
        print("{} model has been loaded!".format(config.model.resume))
        load_state_dict(model.preprocess_model, torch.load('/home/user001/zwl/data/flowir_work_dirs/swin_inp0/lightning_logs/version_0/checkpoints/step=39999.ckpt', map_location="cpu"), strict=True)
        #swin_dict = torch.load(config.model.resume, map_location="cpu")
    #torch.cuda.empty_cache()
    # model.to(device)
    #model.cuda()
    #model = DistributedDataParallel(model,device_ids=[rank],output_device=rank)
    model.eval()


    save_path = args.output
    # final_path = os.path.join(save_path,"final")
    # midd_path = os.path.join(save_path,"midd")
    # os.makedirs(final_path,exist_ok=True)
    # os.makedirs(midd_path,exist_ok=True)

    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)

    testloader = data_module.val_dataloader()
    # for batch_idx, batch in tqdm(enumerate(testloader),total=len(testloader)):
    #     batch['jpg'] = batch['jpg'].to(device)
    #     batch['hint'] = batch['hint'].to(device)
    #     imgname_batch = batch['imgname']
        # log = model.module.validation_step(batch,args.watch_step)
    trainer.test(model,test_dataloaders=testloader)
        #assert False
        #images = log['samples']
        #images_midd = log['samples_3']
        # save_batch(images=images,
        #            imgname_batch=imgname_batch,
        #            save_path=final_path,
        #            watch_step=False)
        # save_batch(images=images_midd,
        #            imgname_batch=imgname_batch,
        #            save_path=midd_path,
        #            watch_step=False)

def save_batch(images,imgname_batch, save_path, watch_step=False):
    if watch_step:
        for list_idx, img_list in enumerate(images):
            for img_idx, img in enumerate(img_list):
                imgname = str(list_idx)+"_"+imgname_batch[img_idx]
                save_img = os.path.join(save_path,imgname)
                save_image(img,save_img)
    else:   
        for img_idx, img in enumerate(images):
            imgname = imgname_batch[img_idx]
            save_img = os.path.join(save_path,imgname)
            save_image(img,save_img)


if __name__ == "__main__":
    main()

'''
CUDA_VISIBLE_DEVICES=0 \
python3 \
test.py \
--config configs/test_cldm.yaml \
--output /home/zyx/DiffBIR-main/outputs/celebamaskhq_reflow_nolora/

CUDA_VISIBLE_DEVICES=6 \
python test.py \
--config configs/test_cldm.yaml \
--output outputs/reflow_celeba_lq_than_lq
'''