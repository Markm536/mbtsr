import pytorch_lightning as pl
import torch
import torch.nn as nn


from .model import SRCompress

class TrainingModule(pl.LightningModule):
    def __init__(self, 
        learning_rate=      1e-3, 
        optimizer=          'adam',
        regularization=     5e-4, 
        sheduler_patience=  5,
        sheduler_factor=    0.2
        ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SRCompress()                            # <----------------------------------------------- super
        self.loss = nn.MSELoss()
        self.train_loss = []
        
        self.learning_rate = learning_rate
        self.optimizer = optimizer          
        self.regularization = regularization    

        self.sheduler_patience = sheduler_patience  
        self.sheduler_factor = sheduler_factor     

    def training_step(self, batch, batch_idx):
        x = batch

        y_rec = self.model(x)

        loss = self.loss(y_rec, x)

        metrics = {"train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.regularization)

        if self.sheduler_patience or self.sheduler_factor:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.sheduler_factor,
                patience=self.sheduler_patience,
                verbose=True,
            )
            lr_dict = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            }

            return [optimizer], [lr_dict]
        else:
            return optimizer

    def validation_step(self, batch, batch_idx):
        x = batch

        y_rec = self.model(x)
        loss = self.loss(y_rec, x)

        metrics = {"val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        return metrics

    def test_step(self, batch, batch_idx):
        x = batch

        y_down = self.model.down(x)
        y_coded = self.model.codec.compress(y_down)
        y_dec = self.model.codec.decompress(**y_coded)
        y_rec = self.model.ups(y_dec)

        test_loss = self.loss(y_rec, x)
        self.log("test_loss", test_loss)

        return test_loss