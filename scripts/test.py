from model.enc_dec import setup_codec
from utils.data import ImagesDataloader #, tomat, topil, cv2pil
from torch.utils.data import DataLoader
import cv2
from pathlib import Path
from PIL import Image

from model.pl_module import TrainingModule 
from utils.data import ImagesDataloader, convert_image
import torch
import pytorch_lightning as pl
import yaml
import argparse
from glob import glob
import json

save_format = "img%03d"

# python -m compressai.utils.find_close jpeg ~/picture.png 35 --metric psnr --save

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_path = "../data/test"

def test_run(ckpt_path, opts_path, test_path):
    model_path = Path(ckpt_path)
    
    model = TrainingModule.load_from_checkpoint(
        checkpoint_path=model_path,
        map_location=device,
    )
    
    trainer = pl.Trainer(accelerator="gpu",
            devices=[0],)
    test_dataloader = DataLoader(ImagesDataloader(test_path), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    res = trainer.test(model, test_dataloader)
    res_ex = model.test_dict
    res_ex.update(res)
    res_ex.update({'psnr_mean' : torch.tensor(res_ex['psnr']).mean().item(), 'bpp_mean' : torch.tensor(res_ex['bpp']).mean().item()})

    test_path = f"{test_path}/kodim04.png"
    x = Image.open(test_path)
    x = convert_image(x, 'pil', '[-1, 1]').unsqueeze(0)
    
    if model.use_residual:
        y, _ = model(x)
    else:
        y = model(x)
        
    y = convert_image(y.squeeze(0), '[-1, 1]', 'pil')
    y.save(f"{model_path.parent}/pic.png")
    return res_ex

        # sbs = cv2.hconcat([im_dist, im_gt])
        # cv2.imwrite(str(out_path / f"{save_format % i}_gt.png"), im_gt) 
        # cv2.imwrite(str(out_path / f"{save_format % i}_dist.png"), im_dist) 

def parse_arguments():

    parser = argparse.ArgumentParser(description="Train script")

    parser.add_argument('--model_path', type=str, help="Validation folder")

    return parser.parse_args()

def main():
    args = parse_arguments()

    res = test_run(
        glob(f"{args.model_path}/epoch=*")[0],
        f"{args.model_path}/opts.yml",
        "../data/test",
    )
    json.dump(res, open(f"{args.model_path}/res.json", 'w'), indent=4)


if __name__ == "__main__":
    main()