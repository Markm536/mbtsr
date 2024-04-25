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

GPU_USE = False
BATCH_SIZE = 2
NUM_WORKERS = 1

def start_train(train_path : str | Path, valid_path : str | Path):
    transform = transforms.Compose([
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # RGB2YCbCr()
    ])

    epoches = 5
    train_dataloader = DataLoader(ImagesDataloader(train_path, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=GPU_USE)
    valid_dataloader = DataLoader(ImagesDataloader(valid_path, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=GPU_USE)

    MyTrainingModuleCheckpoint = ModelCheckpoint(
        dirpath="runs/pl_classifier",
        filename="{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=2,
    )
    MyEarlyStopping = EarlyStopping(monitor="val_loss", mode="min", patience=7, verbose=True)

    if GPU_USE:
        trainer = pl.Trainer(
            max_epochs=epoches,
            accelerator="gpu",
            devices=[1],
            callbacks=[MyTrainingModuleCheckpoint, MyEarlyStopping],
            log_every_n_steps=1,
            strategy="ddp",
        )
    else:
        trainer = pl.Trainer(
            max_epochs=epoches,
            accelerator="cpu",
            devices="auto",
            callbacks=[MyTrainingModuleCheckpoint, MyEarlyStopping],
            log_every_n_steps=1
        )
    training_module = TrainingModule(
        learning_rate=      1e-3, 
        optimizer=          'adam',
        regularization=     5e-4, 
        sheduler_patience=  3,
        sheduler_factor=    0.2
    )

    trainer.fit(training_module, train_dataloader, valid_dataloader)

def parse_arguments():

    parser = argparse.ArgumentParser(description="Train script")

    # parser.add_argument('--skip_tlk', action='store_true', help='If set do not make toloka images')

    # Folder mode opitons
    parser.add_argument('--train_path', type=str, help="Train folder")
    parser.add_argument('--valid_path', type=str, help="Validation folder")

    return parser.parse_args()

def main():
    args = parse_arguments()

    train_path = Path(args.train_path)
    valid_path = Path(args.valid_path)

    start_train(train_path, valid_path)

if __name__ == "__main__":
    main()