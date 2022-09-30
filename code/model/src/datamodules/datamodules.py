import copy
import torch
import pytorch_lightning as pl
import warnings
import transformers

from .datasets import NrSpanDataset, NrClassDataset
from tokenizers import Encoding
from torch.utils.data import DataLoader, random_split, Subset
from transformers import AutoTokenizer

# suppress warnings about number of dataloader workers
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

class BaseDataModule(pl.LightningDataModule):
    '''Base DataModule

    Implement `train_dataloader`, `val_dataloader` and `test_dataloader`

    The `collate_fn` and `setup` methods need to be overridden:
        setup: initialize self.train_dataset, self.val_dataset and self.test_dataset
    '''

    def __init__(self,
                 num_workers,
                 batch_size,
                 val_batch_size,
                 test_batch_size,
                 pin_memory):

        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        pass

    def collate_fn(self, data):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,
            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False,
            persistent_workers=True)


class NrSpanDataModule(BaseDataModule):
    def __init__(self, coarse, n_unique_labels, ignore_index,
                 pretrained_model, 
                 tx_path, 
                 special_tokens,
                 use_test_as_val=False,
                 use_train_as_val=False,
                 use_biolu=True,
                 num_workers=2,
                 train_val_test_split=None,
                 inference=False,
                 bsz=1, val_bsz=1, test_bsz=1,
                 pin_memory=True, **kwargs):
        '''
        Args:
            inference: True if doing an inference task    
            use_biolu: use BIOLU if True, otherwise BIO
        '''

        super().__init__(num_workers=num_workers,
                         batch_size=bsz,
                         val_batch_size=val_bsz,
                         test_batch_size=test_bsz,
                         pin_memory=pin_memory)

        self.inference = inference
        self.tx_path = tx_path
        self.special_tokens = special_tokens
        self.coarse = coarse
        self.n_unique_labels = n_unique_labels
        self.ignore_index = ignore_index
        self.num_workers = num_workers
        self.batch_size = bsz
        self.val_batch_size = val_bsz
        self.test_batch_size = test_bsz
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.use_test_as_val = use_test_as_val
        self.use_train_as_val = use_train_as_val
        self.use_biolu = use_biolu
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, use_fast=True)

    def setup(self, stage=None):
        # load dataset
        dataset = NrSpanDataset(tx_path=self.tx_path, coarse=self.coarse,
                                special_tokens=self.special_tokens,
                                n_unique_labels=self.n_unique_labels, ignore_index=self.ignore_index,
                                use_biolu=self.use_biolu)

        # get id-label mapping
        self.label_to_id = dataset.label_to_id
        self.id_to_label = dataset.id_to_label

        # if inference mode, use all data points
        if self.inference:
            self.train_dataset, self.val_dataset, self.test_dataset = dataset, dataset, dataset
            self.bsz, self.val_bsz, self.test_bsz = 1, 1, 1

        # otherwise, split train/val/val dataset
        else:
            assert (self.train_val_test_split is not None), \
                'Get empty `train_val_test`!'
            assert (sum(self.train_val_test_split) == 1), \
                f'Require sum(train_val_test_split)==1 but get {self.train_val_test_split=}'

            # split train/val/test
            n_train_val_test = [round(x * len(dataset))
                                for x in self.train_val_test_split]

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset, n_train_val_test,
                generator=torch.Generator().manual_seed(40))

            # override val if realy needed
            if self.use_test_as_val:
                self.val_dataset = copy.deepcopy(self.test_dataset)

            if self.use_train_as_val:
                self.val_dataset = copy.deepcopy(self.train_dataset)

            # if test_dataset is empty, override test_dataset with val_dataset
            if len(self.test_dataset) == 0:
                self.test_dataset = copy.deepcopy(self.val_dataset)


    def collate_fn(self, data):
        # Unpack a batch
        texts, annotations, headline_classes = zip(*data)

        # get tokens
        tokens = self.tokenizer(list(texts), padding=True, return_tensors='pt')

        # get input_ids, attention_mask
        # input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long)

        input_ids = tokens['input_ids'].clone().detach()
        
        # attention_mask = torch.tensor(
        #     tokens['attention_mask'], dtype=torch.long)

        attention_mask = tokens['attention_mask'].clone().detach()

        # if inference mode:
        # - don't need to align labels
        if self.inference:
            labels = tokens.attention_mask.clone().detach().zero_()

        else:
            # align labels
            aligned_labels = []

            for ix in range(len(tokens.encodings)):
                encoding = tokens.encodings[ix]
                raw_annotation = annotations[ix]

                raw_labels = self.align_tokens_and_annotation_bilou(
                    encoding, raw_annotation)

                assert set(raw_labels).issubset(self.label_to_id.keys()), \
                    f'Detect unknown labels in: {raw_labels=}'

                aligned_label = list(map(self.label_to_id.get, raw_labels))

                aligned_labels.append(aligned_label)

            labels = torch.tensor(aligned_labels, dtype=torch.long)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                't': labels,
                'texts': texts,
                'headline_classes': headline_classes}

    def align_tokens_and_annotation_bilou(self, tokenized_text: Encoding, annotation):
        '''Assign a label for each token
        '''
        tokens = tokenized_text.tokens

        # Make a list to store our labels the same length as our tokens
        # all labels are initialized to 'O'
        aligned_labels = ['O'] * len(tokens)

        # check if the special tokens is a subset of known special tokens ([CLS], [SEP], [PAD])
        special_tokens_mask = tokenized_text.special_tokens_mask

        special_tokens_in_the_headline = set(token for token, mask in zip(tokens, special_tokens_mask) if mask!=0)
        assert special_tokens_in_the_headline.issubset(set(self.special_tokens)), \
            f'Detect unknown special tokens in: {special_tokens_in_the_headline}'

        # for token in self.special_tokens, their labels are themselves
        aligned_labels = [token if token in self.special_tokens else label
                          for label, token in zip(aligned_labels, tokens)]

        for anno in annotation:
            # A set that stores the token indices of the annotation
            annotation_token_ix_set = (set())

            # if there's no span annotation in this sample, skip
            if anno == []:
                continue

            for char_ix in range(anno['start'], anno['end']):
                token_ix = tokenized_text.char_to_token(char_ix)
                if token_ix is not None:  # White spaces have no token and will return None
                    annotation_token_ix_set.add(token_ix)

            # If there is only one token
            if self.use_biolu and (len(annotation_token_ix_set) == 1):
                token_ix = annotation_token_ix_set.pop()
                # This annotation spans one token so is prefixed with U for unique
                prefix = ("U")
                aligned_labels[token_ix] = f"{prefix}-{anno['class']}"

            # spans multiple tokens
            else:
                last_token_in_anno_ix = len(annotation_token_ix_set) - 1
                for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                    if num == 0:
                        prefix = "B"
                    elif self.use_biolu and (num == last_token_in_anno_ix):
                        prefix = "L"  # Its the last token
                    else:
                        prefix = "I"  # We're inside of a multi token annotation
                    aligned_labels[token_ix] = f"{prefix}-{anno['class']}"

        return aligned_labels


class NrClassDataModule(BaseDataModule):
    def __init__(self, tx_path,
                 pretrained_model,
                 coarse,
                 n_unique_labels,
                 val_tx_path=None,
                 use_test_as_val=False,
                 use_train_as_val=False,
                 num_workers=2,
                 train_val_test_split=None,
                 inference=False,
                 bsz=1, val_bsz=1, test_bsz=1,
                 pin_memory=True, **kwargs):
        '''
        Args:
            coarse: if True, use coarse labels (i.e., `first_reason_type_coarse`)
            inference: True if doing an inference task    
        '''

        super().__init__(num_workers=num_workers,
                         batch_size=bsz,
                         val_batch_size=val_bsz,
                         test_batch_size=test_bsz,
                         pin_memory=pin_memory)

        self.inference = inference
        self.n_unique_labels = n_unique_labels
        self.coarse = coarse
        self.tx_path = tx_path
        self.val_tx_path = val_tx_path
        self.num_workers = num_workers
        self.batch_size = bsz
        self.val_batch_size = val_bsz
        self.test_batch_size = test_bsz
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.use_test_as_val = use_test_as_val
        self.use_train_as_val = use_train_as_val
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, use_fast=True)

    def setup(self, stage=None):
        # load dataset
        dataset = NrClassDataset(tx_path=self.tx_path, coarse=self.coarse)

        # check if cfg.datamodule.n_unique_labels equals to the value computed from the dataset
        assert self.n_unique_labels == len(dataset.id_to_label), \
            f'{self.n_unique_labels=} != {len(dataset.id_to_label)=}'

        # if inference mode:
        #   train/val/test all equals to the full dataset
        #   bsz/val_bsz/test_bsz all equals to 1
        if self.inference:
            self.train_dataset, self.val_dataset, self.test_dataset = dataset, dataset, dataset
            self.bsz, self.val_bsz, self.test_bsz = 1, 1, 1

        # otherwise, split train/val/val dataset
        else:

            # ---- split train/val/test ----

            # if train_val_test_split is List[float]
            if sum(self.train_val_test_split) == 1:
                n_train_val_test = [round(x * len(dataset))
                                    for x in self.train_val_test_split]

                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    dataset, n_train_val_test,
                    generator=torch.Generator().manual_seed(40))

            # if train_val_test_split is List[int]
            else:
                n_train, n_val, n_test = self.train_val_test_split
                n_total = len(dataset)
                self.train_dataset = Subset(dataset, range(0, n_train))
                self.test_dataset = Subset(dataset, range(n_total-n_test, n_total))

            # override val if realy needed
            if self.use_test_as_val:
                self.val_dataset = copy.deepcopy(self.test_dataset)

            if self.use_train_as_val:
                self.val_dataset = copy.deepcopy(self.train_dataset)

            # if test_dataset is empty, override with val
            if len(self.test_dataset) == 0:
                self.test_dataset = copy.deepcopy(self.val_dataset)

    def collate_fn(self, data):
        # Unpack a batch
        texts, labels, label_ids = zip(*data)

        # get tokens
        tokens = self.tokenizer(list(texts), padding=True)

        # get input_ids, attention_mask
        input_ids = torch.tensor(tokens['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(
            tokens['attention_mask'], dtype=torch.long)

        label_ids = torch.tensor(label_ids, dtype=torch.long)

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                't': label_ids,
                'texts': texts,
                'labels': labels}
