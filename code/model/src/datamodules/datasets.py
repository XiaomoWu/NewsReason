import itertools
import torch

from pyarrow.feather import read_feather
from typing import List
from torch.utils.data import Dataset

class NrSpanDataset(Dataset):
    def __init__(self, 
                 tx_path: str,
                 n_unique_labels, 
                 special_tokens,
                 ignore_index,
                 use_biolu, 
                 coarse):
        '''
        Args:
            tx_path: the path for targets and features
            tests: a list of texts
                Notes: only one out of tx_path and tests can be not None
        '''

        # define the unique labels
        self.special_tokens = special_tokens
        self.n_unique_labels = n_unique_labels
        self.ignore_index = ignore_index
        self.use_biolu = use_biolu

        # import training data: text, annotations, headline classes
        self.texts, self.annotations, self.headline_classes = self.read_tx(tx_path, coarse)

        # get the unique classes
        self.unique_classes = self.get_unique_classes(self.annotations)

        # get label_to_id mapping
        self.label_to_id = self.get_label_to_id()

        # get id_to_label mapping (the ignore_index (-100) is not included in "id")
        self.id_to_label = self.id_to_label()

        # since PL doesn't allow model to directly access the dataset, we have to 
        # save label_to_id to disk and load it within the model
        torch.save(self.id_to_label, '/home/yu/OneDrive/NewsReason/local-dev/code/model/src/id_to_label.pt')


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        '''
        Return:
            (texts, annotations)
                texts: str
                annotations: List[Dict]
        '''
        return (self.texts[idx], self.annotations[idx], self.headline_classes[idx])

    def read_tx(self, tx_path, coarse):
        '''Read labels and features from

        Return: 
            texts: List[str]
            annotations: List[Dict]
        '''
           
        tx = read_feather(tx_path)

        # get text (headlines)
        texts = tx['text'].tolist()

        # get headline class
        # - headline class is only used in the validation step, where
        #   we want to evaluate the classification performance even if 
        #   we're training a SPAN model
        if coarse:
            headline_classes = tx['first_reason_type_coarse'].tolist()
        else:
            headline_classes = tx['first_reason_type'].tolist()

        # get annotations
        annotations = [None] * len(texts)

        if coarse:
            reason_var_name = 'reasons_coarse'
        else:
            reason_var_name = 'reasons'

        if tx.get(reason_var_name) is not None:
            for i, reasons in enumerate(tx[reason_var_name].tolist()):
                annotation = []  # all annotations for one row (headline)
                if reasons is not None:
                    for reason in reasons:
                        annotation.append(
                            {'start': int(reason[0]), 'end': int(reason[1]), 'class': reason[2]})

                annotations[i] = annotation
                
        return texts, annotations, headline_classes

    def get_unique_classes(self, annotations):
        '''get the unique classes (i.e., unique headline types)
            
            examples of "classes": 'Ligitation', 'Fraud' 
            examples of "labels": 'B-Ligation', 'I-Ligation', 'B-Fraud', 'I-Fraud' 
        '''
        classes = [a['class'] for anno in annotations for a in anno]

        return list(set(classes))

    def get_label_to_id(self):
        '''Return a mapping from labels (str) to ids (int)

        Return:
            label_to_id: Dict[str, int]. 
        '''

        label_to_id = {}

        # all speical tokens are mapped to the same id: -100
        for t in self.special_tokens:
            label_to_id[t] = self.ignore_index

        label_to_id['O'] = 0
        num = 0  # in case there are no labels

        # Writing BILU will give us incremntal ids for the labels
        if self.use_biolu:
            for _num, (label, s) in enumerate(itertools.product(self.unique_classes, "BILU")):
                num = _num + 1  # skip 0
                l = f"{s}-{label}"
                label_to_id[l] = num
        else:
            for _num, (label, s) in enumerate(itertools.product(self.unique_classes, "BI")):
                num = _num + 1  # skip 0
                l = f"{s}-{label}"
                label_to_id[l] = num

        # check if the number of unique labels equals the value computed from the data
        n_labels_exclude_cls_sep_pad = len([k for k in label_to_id.keys() if k not in self.special_tokens])
        assert n_labels_exclude_cls_sep_pad == self.n_unique_labels, \
            f"Number of unique labels computed from the data '{len(set(label_to_id.values()))}' does not match the manually set value '{self.n_unique_labels}'"

        print(f'N unique ids={self.n_unique_labels}')
        print(f'{label_to_id=}')

        return label_to_id

    def id_to_label(self):
        '''Return a mapping from ids (int) to labels (str) (ignore_index is not included)

        Return:
            id_to_label: Dict[int, str]
        '''
        id_to_label = {v: k for k, v in self.label_to_id.items() if v != self.ignore_index}

        return id_to_label

class NrClassDataset(Dataset):
    '''Dataset for "classification" problem
    '''
    def __init__(self, 
                 tx_path: str,
                 coarse: bool):
        '''
        Args:
            tx_path: the path for targets and features
            tests: a list of texts
                Notes: only one of tx_path or tests can be not None
        '''

        # import training data: text, annotations
        self.texts, self.labels, self.label_ids, \
            self.id_to_label = self.read_tx(tx_path, coarse)

        # get labels
        print(f'\nN unique labels = {len(self.id_to_label)}\n')
        print(f'id_to_label:\n\n{self.id_to_label}\n')


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        '''
        Return:
            texts: str
            label: str
            label_id: int
        '''
        text = self.texts[idx]
        label = self.labels[idx]
        label_id = self.label_ids[idx]

        return text, label, label_id

    def read_tx(self, tx_path, coarse):
        '''Read headlines and labels

        Return: 
            texts: List[str]
            labels: List[str]
            label_ids: List[int]
            id_to_label: Dict[int, str]
        '''
           
        tx = read_feather(tx_path)

        # get text (headlines)
        texts = tx['text'].tolist()

        if coarse:
            # get labels
            labels = tx['first_reason_type_coarse'].tolist()
            label_ids = tx['first_reason_type_coarse_id'].tolist()

            # get id_to_label mappings
            unique_tx = tx[['first_reason_type_coarse', 'first_reason_type_coarse_id']].drop_duplicates()
            unique_labels = unique_tx['first_reason_type_coarse'].tolist()
            unique_label_ids = unique_tx['first_reason_type_coarse_id'].tolist()

            id_to_label = {id: label for id, label in zip(unique_label_ids, unique_labels)}

        else:
            # get labels
            labels = tx['first_reason_type'].tolist()
            label_ids = tx['first_reason_type_id'].tolist()

            # get id_to_label mappings
            unique_tx = tx[['first_reason_type', 'first_reason_type_id']].drop_duplicates()
            unique_labels = unique_tx['first_reason_type'].tolist()
            unique_label_ids = unique_tx['first_reason_type_id'].tolist()

            id_to_label = {id: label for id, label in zip(unique_label_ids, unique_labels)}

        return texts, labels, label_ids, id_to_label

