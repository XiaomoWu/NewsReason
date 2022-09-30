import pytorch_lightning as pl
import torch

from pytorch_lightning.demos.boring_classes import BoringDataModule, BoringModel
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy

class BoringModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(768, 16)

        # validation metrics from torchmetrics
        self.val_acc = MulticlassAccuracy(num_classes=16, average='micro', multidim_average='samplewise', ignore_index=-100)
        print(f'{self.val_acc=}')

    def configure_optimizers(self):
        return torch.optim.Adam(self.layer.parameters())

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_idx: int):
        preds = self.layer(batch).transpose(1, 2)  # (N, C, S)
        target = torch.randint(0, 16, (preds.shape[0], preds.shape[2]), dtype=torch.long) \
            .to(preds.device)  # (N, L)

        loss = torch.nn.functional.cross_entropy(preds, target)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):

        preds = self.layer(batch).transpose(1, 2)  # (N, C, D)
        target = torch.randint(0, 16, (preds.shape[0], preds.shape[2]), dtype=torch.long) \
            .to(preds.device)  # (N, L)

        # print(f'{preds.shape=}')
        # print(f'{target.shape=}')

        # update validation metrics
        self.val_acc.update(preds, target)
        # self.log('val/acc', self.val_acc, on_step=False, on_epoch=True)
    
    def validation_epoch_end(self, outputs):
        self.log('val/acc', self.val_acc.compute())
        self.val_acc.reset()

# shape:
#     N: number of samples
#     C: number of classes
#     L: sequence length per sample
#     D: embedding dimension

#     input: (N, L, D) where N=64, L=32, D=768

#     preds: (N, C, L) where N=64, C=16, L=32
#     target: (N, C) where N=64, C=16

# init dataloader
dataset = TensorDataset(torch.rand(64, 32, 768))[0]
dataloader = DataLoader(dataset, batch_size=8, num_workers=0)

# init model
model = BoringModel()

# init trainer
trainer = Trainer(max_epochs=10, accelerator='gpu', devices=2, strategy='ddp', check_val_every_n_epoch=1)
trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=dataloader)

