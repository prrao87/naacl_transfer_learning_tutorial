from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm_notebook as tqdm
from typing import Tuple
from pytorch_transformers import BertTokenizer

TEXT_COL, LABEL_COL = 'text', 'label'
MAX_LENGTH = 256
BATCH_SIZE = 16
LOG_DIR = "./logs/"
CACHE_DIR = "./cache/"
# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_cpu = cpu_count()


def read_sst2(data_dir, colnames=[TEXT_COL, LABEL_COL]):
    datasets = {}
    for t in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"{t}.tsv"), header=None, sep='\t', names=colnames)
        datasets[t] = df[[LABEL_COL, TEXT_COL]]
    return datasets


class TextProcessor:
    # special tokens for classification and padding
    CLS = '[CLS]'
    PAD = '[PAD]'

    def __init__(self, tokenizer, label2id: dict, max_length: int=512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.max_length = max_length

    def process(self, item: Tuple[str, str]):
        "Convert text (item[0]) to sequence of IDs and label (item[1] to integer"
        assert len(item) == 2
        label, text = item[0], item[1]
        assert isinstance(text, str)
        tokens = self.tokenizer.tokenize(text)

        # truncate if too long
        if len(tokens) >= self.max_length:
            tokens = tokens[:self.max_length - 1]
            ids = self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.CLS]]
        # pad if too short
        else:
            pad = [self.tokenizer.vocab[self.PAD]] * (self.max_length - len(tokens) - 1)
            ids = self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.CLS]] + pad

        return np.array(ids, dtype='int64'), self.label2id[label]


def process_row(processor, row):
    "Calls the process method of the text processor for passing items to executor"
    return processor.process((row[1][LABEL_COL], row[1][TEXT_COL]))


def create_dataloader(df: pd.DataFrame,
                      processor: TextProcessor,
                      batch_size: int = 16,
                      shuffle: bool = False,
                      valid_pct: float = None,
                      text_col: str = "text",
                      label_col: str = "label"):
    "Process rows in pd.DataFrame using ncpus and return a DataLoader"

    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        result = list(
            tqdm(executor.map(process_row,
                              repeat(processor),
                              df.iterrows(),
                              chunksize=len(df) // 10),
                 desc=f"Processing {len(df)} examples on {n_cpu} cores",
                 total=len(df)))

    features = [r[0] for r in result]
    labels = [r[1] for r in result]

    dataset = TensorDataset(torch.tensor(features, dtype=torch.long),
                            torch.tensor(labels, dtype=torch.long))

    if valid_pct is not None:
        valid_size = int(valid_pct * len(df))
        train_size = len(df) - valid_size
        valid_dataset, train_dataset = random_split(dataset,
                                                    [valid_size, train_size])
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        return train_loader, valid_loader

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=shuffle,
                             pin_memory=torch.cuda.is_available())
    return data_loader


def get_and_tokenize_dataset(dataset_dir: str, finetuning_config: namedtuple):
    # Dataset path
    datasets = read_sst2(dataset_dir)

    # Get list of labels from training data
    labels = list(set(datasets["train"][LABEL_COL].tolist()))
    # Map labels to dict using zero-indexing
    label2int = {label: i for i, label in enumerate(labels)}

    # Use BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    # Initialize TextProcessor
    processor = TextProcessor(tokenizer, label2int, max_length=MAX_LENGTH)

    # Create train, validation and test dataloaders
    train_dl = create_dataloader(datasets["dev"], processor,
                                 batch_size=finetuning_config.batch_size,
                                 valid_pct=None)

    # val_dl = create_dataloader(datasets["dev"], processor,
    #                            batch_size=finetuning_config.batch_size,
    #                            valid_pct=None)
    val_dl = None

    test_dl = create_dataloader(datasets["test"], processor, 
                                batch_size=finetuning_config.batch_size,
                                valid_pct=None)

    return train_dl, val_dl, test_dl
