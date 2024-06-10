import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.pl_module import TrainingModule
from utils.data import ImagesDataloader
from pathlib import Path
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from CompressAI.compressai.transforms.transforms import RGB2YCbCr
import argparse
import yaml
import shutil, os

GPU_USE = torch.cuda.is_available()
NUM_WORKERS = 16

def start_train(train_path : str | Path, valid_path : str | Path, opts : dict):
    crop_size = opts['crop_size']
    BATCH_SIZE = opts['lr_params']['batch_size']

    tp = f"runs/down=" \
         f"{opts['is_simple']}" \
         f"@sc={opts['scaling_factor']}" \
         f"@cod1={opts['codec_params']['quality']}{opts['res_codec_params']['quality']}" \
         f"@res={opts['use_residual']}" \
         f"@gan={opts['use_adv']}" \
         f"@vgg={opts['use_vgg']}" 

    print(f"MODEL SAVED: {tp}")
    os.makedirs(tp, exist_ok=True)

    with open(f'{tp}/opts.yml', 'w') as fp:
        yaml.dump(opts, fp, default_flow_style=False)
    
    transform = transforms.Compose([
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomCrop((crop_size, crop_size)),
        # RGB2YCbCr()
    ])

    epoches = opts['lr_params']['epoches']
    train_dataloader = DataLoader(ImagesDataloader(train_path, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=GPU_USE)
    valid_dataloader = DataLoader(ImagesDataloader(valid_path, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=GPU_USE)

    MyTrainingModuleCheckpoint = ModelCheckpoint(
        dirpath=tp,
        filename="{epoch}-{psnr:.3f}",
        monitor="psnr",
        mode="max",
        save_top_k=1,
    )
    MyEarlyStopping = EarlyStopping(monitor="psnr", mode="max", patience=15, verbose=True)

    if GPU_USE:
        trainer = pl.Trainer(
            max_epochs=epoches,
            accelerator="gpu",
            devices=[0],
            callbacks=[MyTrainingModuleCheckpoint, MyEarlyStopping],
            log_every_n_steps=1,
            strategy="ddp_find_unused_parameters_true",
        )
    else:
        trainer = pl.Trainer(
            max_epochs=epoches,
            accelerator="cpu",
            devices="auto",
            callbacks=[MyTrainingModuleCheckpoint, MyEarlyStopping],
            log_every_n_steps=1
        )
    
    training_module = TrainingModule(opts, tp)

    trainer.fit(training_module, train_dataloader, valid_dataloader)

def parse_arguments():

    parser = argparse.ArgumentParser(description="Train script")

    # Folder mode opitons
    parser.add_argument('--train_path', type=str, default='../data/train_oi', help="Train folder")
    parser.add_argument('--valid_path', type=str, default='../data/valid_oi',help="Validation folder")
    parser.add_argument('--opts_path', type=str,  default='./scripts/opts.yml', help="Validation folder")

    parser.add_argument('--l_mse', type=float, default=0.0)
    parser.add_argument('--l_adv', type=float, default=0.0)
    parser.add_argument('--l_vgg', type=float, default=0.0)
    parser.add_argument('--l_mse_part', type=float, default=1.0)
    
    parser.add_argument('--is_simple', action='store_true')

    parser.add_argument('--scaling_factor', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)

    return parser.parse_args()

def main():
    args = parse_arguments()
    opts = Path(args.opts_path)
    train_path = Path(args.train_path)
    valid_path = Path(args.valid_path)

    opts = yaml.safe_load(open(opts))

    if args.l_mse:
        opts['lr_params']['l_mse'] = args.l_mse
    if args.l_adv:
        opts['lr_params']['l_adv'] = args.l_adv
    if args.l_vgg:
        opts['lr_params']['l_vgg'] = args.l_vgg
    if args.l_mse_part:
        opts['lr_params']['l_mse_part'] = args.l_mse_part
    if args.is_simple:
        opts['is_simple'] = args.is_simple
    if args.scaling_factor:
        opts['scaling_factor'] = args.scaling_factor
    if args.crop_size:
        opts['crop_size'] = args.crop_size

    opts['use_vgg'] = opts['lr_params']['l_vgg'] > 0.0
    opts['use_adv'] = opts['lr_params']['l_adv'] > 0.0
    opts['use_mse_part'] = opts['lr_params']['l_mse_part'] > 0.0
    opts['use_residual'] = opts['lr_params']['l_mse'] > 0.0
    opts['gen_params']['scaling_factor'] = opts['scaling_factor']
    opts['down_params']['scaling_factor'] = opts['scaling_factor']

    start_train(train_path, valid_path, opts)

if __name__ == "__main__":
    main()