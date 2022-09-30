import hydra
import itertools
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import transformers

from collections import Counter
from einops import rearrange
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from transformers import AutoModel


class NrSpanModel(pl.LightningModule):
    def __init__(self, pretrained_model, d_model, datamodule_cfg, **kwargs):
        super().__init__()
        self.n_unique_labels = datamodule_cfg['n_unique_labels']

        # read id_to_label
        self.id_to_label = torch.load('/home/yu/OneDrive/NewsReason/local-dev/code/model/src/id_to_label.pt')

        # init encoder
        # suppress "Some weights are not initialized"
        transformers.logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        transformers.logging.set_verbosity_info()

        self.encoder.train()

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, self.n_unique_labels))

    def forward(self, batch):
        y = self.get_y(batch)
        return y

    def get_y(self, batch):
        # get input_ids, attention_mask
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        y = self.classifier(output)
        return y


class NrClassModel(pl.LightningModule):
    def __init__(self, pretrained_model, d_model, datamodule_cfg, **kwargs):
        super().__init__()
        self.n_unique_labels = datamodule_cfg['n_unique_labels']

        # suppress "Some weights are not initialized"
        transformers.logging.set_verbosity_error()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        # suppress "Some weights are not initialized"
        transformers.logging.set_verbosity_info()

        self.encoder.train()

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, self.n_unique_labels))

    def forward(self, batch):
        y = self.get_y(batch)
        return y

    def get_y(self, batch):
        # get input_ids, attention_mask
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        y = self.classifier(output)
        return y


class Model(pl.LightningModule):
    def __init__(self,
                 model_cfg,
                 datamodule_cfg,
                 optimizer_cfg,
                 trainer_cfg):
        super().__init__()
        self.save_hyperparameters()

        # ----------------------
        # init forward model
        # ----------------------
        self.model = hydra.utils.instantiate(
            model_cfg,
            datamodule_cfg=datamodule_cfg,
            _recursive_=False)

        # ----------------------
        # init loss functions
        # ----------------------
        self.model_type = model_cfg.model_type
        self.n_unique_labels = datamodule_cfg['n_unique_labels']

        if self.model_type == 'NrSpan':
            self.ignore_index = datamodule_cfg['ignore_index']

            self.train_cross_entropy = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index)

            # `val_acc`: "unbalanced" version (the default as NrClass)
            self.val_acc = MulticlassAccuracy(
                num_classes=self.n_unique_labels, ignore_index=self.ignore_index, multidim_average='samplewise', average='micro')

        elif self.model_type == 'NrClass':
            self.train_cross_entropy = nn.CrossEntropyLoss()
            self.val_acc = MulticlassAccuracy(
                num_classes=self.n_unique_labels, average='micro')

    def forward(self, batch):
        y = self.model(batch)
        return y

    def predict(self, batch):
        y = self.model(batch)
        return y

    def training_step(self, batch, batch_idx):
        t = self.get_t(batch)
        y_logits = self.model(batch)  # (N, S, C)

        if self.model_type == 'NrSpan':
            # since NrSpan is "multi-dimensinal & multi-label", need to reshape
            y_logits = rearrange(y_logits, 'n s c -> n c s')

        # compute loss
        loss = self.train_cross_entropy(y_logits, t)

        return loss

    def validation_step(self, batch, batch_idx):
        y_logits = self.model(batch)
        t = self.get_t(batch)

        if self.model_type == 'NrSpan':
            # since NrSpan is "multi-dimensinal & multi-label", need to reshape
            y_logits = rearrange(y_logits, 'n s c -> n c s')

            # first, compute "token-level" accuracy
            self.val_acc.update(y_logits, t)

            '''
            # get "headline-level" prediction
            headline_class_preds = self.get_headline_class_preds(
                batch, y_logits)

            return headline_class_preds
            '''

        elif self.model_type == 'NrClass':
            # since I only log val metrics on the epoch end, use `.update` to avoid extra computation
            self.val_acc.update(y_logits, t)

    def validation_epoch_end(self, outputs):
        if self.model_type == 'NrSpan':
            # ---- get "token-level" accuracy ----
            val_acc = self.val_acc.compute().mean()*100

            self.log('val/acc', val_acc, sync_dist=True)

            # (for debugging) print metrics
            if self.global_rank == 0:
                print(f'val/acc={val_acc: .2f}%')

            self.val_acc.reset()

            '''
            # ---- get "headline-level" accuracy ----
            outputs = list(itertools.chain(*outputs))
            
            headline_acc = sum(outputs)/len(outputs)*100

            # (for debugging) print metrics
            if self.global_rank == 0:
                print(f'val/headline_acc={headline_acc: .2f}%')

            self.log('val/headline_acc', headline_acc)
            '''

        elif self.model_type == 'NrClass':
            val_acc = self.val_acc.compute()*100
            self.log('val/acc', val_acc)

            # (for debugging) print metrics
            if self.global_rank == 0:
                print(f'val/acc={val_acc: .3f}%')

            self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        if self.model_type == 'NrClass':
            y = self.model(batch).argmax(-1)

        # get truth
        t = self.get_t(batch)

        return {'y': y,
                't': t}

    def test_epoch_end(self, outputs):
        # collect yt
        y, t = self.collect_yt(outputs)

        if self.model_type == 'NrClass':
            test_acc = torchmetrics.functional.accuracy(y, t)

            if self.global_rank == 0:
                print(f'test/acc={test_acc:.3f}%')

            self.log('test/acc', test_acc, sync_dist=True)

        elif self.model_type == 'NrSpan':
            test_f1 = torchmetrics.functional.f1_score(y, t, self.ignore_index)

            if self.global_rank == 0:
                print(f'test/f1={test_f1:.3f}%')

            self.log('test/f1', test_f1, sync_dist=True)

    def configure_optimizers(self):
        # init optimizer
        optimizer = self.init_optimizer(
            self.hparams.optimizer_cfg, params=self.parameters())

        return {
            'optimizer': optimizer,
        }

    def init_optimizer(self, optimizer_cfg, params):
        optimizer_name = optimizer_cfg._target_.split('.')[-1]

        # depending on whether the optimizer is DeepSpeed, initialize differently
        if 'DeepSpeed' in optimizer_name:
            optimizer = hydra.utils.instantiate(
                optimizer_cfg, model_params=params, _convert_='partial')
        else:
            optimizer = hydra.utils.instantiate(
                optimizer_cfg, params=params, _convert_='partial')

        return optimizer

    def get_t(self, batch):
        t = batch['t']
        return t

    def get_headline_class_preds(self, batch, y_logits):
        '''Compute "headline-level", "classification", accuracy
        '''
        class_t = batch['headline_classes']

        # here we generate the "headline-level" prediction of class, given y_logits
        # 1) get the argmax of each token
        # 2) remove all "ignore_index"
        # 3) then, if all the remaining is 0, then it's "0" (i.e. "no news")
        # 4) otherwise, find the mode
        class_y = []
        y_indices = torch.argmax(y_logits, dim=1).tolist()
        for obs in y_indices:
            # remove all "ignore_index"
            obs = [i for i in obs if i != self.ignore_index]

            # if all the remaining is 0, then it's "[NONE]" (i.e. "no news")
            if obs.count(0) == len(obs):
                class_y.append('[NONE]')
            # otherwise, find the most frequent
            else:
                obs = [i for i in obs if i != 0]
                idx = Counter(obs).most_common(1)[0][0]
                headline_class = self.model.id_to_label[idx].split('-')[1]
                class_y.append(headline_class)

        # return result, "0" if wrong (class_y != class_t), "1" if correct
        return [int(y == t) for y, t in zip(class_y, class_t)]

    def collect_yt(self, outputs):
        '''Collect yt and convert them to approprite dimension
        '''
        # collect yt from one process
        y = torch.cat([x['y'] for x in outputs])
        t = torch.cat([x['t'] for x in outputs])

        # all_gather yt
        y = self.all_gather(y).detach()
        t = self.all_gather(t).detach()

        return y, t
