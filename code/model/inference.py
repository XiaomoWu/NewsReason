import datatable as dt
import numpy as np
import pyarrow as pa
import torch
import pandas as pd

from collections import Counter
from IPython.utils import io
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from src.datamodules.datamodules import NrSpanDataModule
from src.models.models import Model
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List


def load_checkpoint(zero_ckpt_dir, ckpt_resave_path):
    '''
    Args:
        zero_ckpt_path (str): path to zero checkpoint (when using deepspeed)
        pt_ckpt_path (str): path to pt checkpoint (when using ddp)
    '''

    convert_zero_checkpoint_to_fp32_state_dict(
        zero_ckpt_dir, ckpt_resave_path)

    return torch.load(ckpt_resave_path)


def init_model(device: str):
    # ------------------
    # load ckpt
    # ------------------
    zero_ckpt_dir = '/home/yu/OneDrive/NewsReason/local-dev/checkpoints/epoch=11.ckpt'
    ckpt_resave_path = '/home/yu/OneDrive/NewsReason/local-dev/checkpoints/ckpt_saved.pt'

    ckpt = load_checkpoint(zero_ckpt_dir, ckpt_resave_path)

    # ------------------
    # collect hparams
    # ------------------
    hparams = ckpt['hyper_parameters']
    datamodule_cfg = hparams['datamodule_cfg']
    model_cfg = hparams['model_cfg']

    # ------------------
    # init model
    # ------------------

    state_dict = ckpt['state_dict']

    with io.capture_output() as captured:
        model = Model(**hparams)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

    return model, datamodule_cfg, model_cfg


def init_dataloader(pretrained_model, tx_path, coarse, n_unique_labels, ignore_index):

    datamodule = NrSpanDataModule(
        inference=True,
        train_val_test_split=[1, 0, 0],
        use_biolu=False,
        tx_path=tx_path,
        ignore_index=ignore_index,
        coarse=coarse,
        n_unique_labels=n_unique_labels,
        special_tokens=['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>'],
        pretrained_model=pretrained_model)

    datamodule.setup()

    return datamodule


def inference(datamodule):
    '''
    '''
    # get dataloader
    dataloader = datamodule.test_dataloader()
    
    # get tokenizer
    tokenizer = datamodule.tokenizer

    # make predictions!
    headlines = []
    pred_reasons = []

    # get speical token_ids (i.e., the ids of speical tokens like [CLS], or </s>)
    special_tokens = tokenizer.special_tokens_map.values()
    special_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in special_tokens]

    for i, batch in enumerate(tqdm(dataloader)):
        if i >= 5:
            break

        input_ids = batch['input_ids'][0].tolist()  # List[int]
        texts = batch['texts']  # List[str]

        # collect headlines
        headlines.append(texts[0])

        # predict
        batch = {k: v.to(device) for k, v in batch.items()
                 if k not in ['texts', 'headline_classes']}

        with torch.no_grad():
            y = model.predict(batch).argmax(-1)[0].cpu()

            # get the index of each reason token
            y = y.tolist()

            # the index of reason token.
            reason_token_ixs = []

            # the id (in its tokenizer) of each reason token.
            reason_token_ids = []

            for i, (input_id, token_pred) in enumerate(zip(input_ids, y)):
                if (token_pred not in [0, ignore_index]) and (input_id not in special_token_ids):
                    reason_token_ixs.append(i)
                    reason_token_ids.append(input_id)

        # save predicted reasons
        reason = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(reason_token_ids))
        pred_reasons.append(reason)

    
    # save results to a table
    return pd.DataFrame({'headline': headlines, 'pred_reasons': pred_reasons})



if __name__ == '__main__':
    # init model
    device = 'cuda:0'

    model, datamodule_cfg, model_cfg = init_model(device)

    # init dataloader
    pretrained_model = 'roberta-large'
    tx_path = '/home/yu/OneDrive/NewsReason/local-dev/data/annotation/batch-4/2-annotated/annotated_agreed_full_batch3_4.feather'
    coarse = True
    # [BIO]: 25 (coarse), 49 (fine) | [BIOLU]: 49 (coarse), 97 (fine)
    n_unique_labels = 25
    ignore_index = -100

    datamodule = init_dataloader(
        pretrained_model, tx_path, coarse, n_unique_labels, ignore_index)

    # run inference
    inference_output = inference(datamodule)
