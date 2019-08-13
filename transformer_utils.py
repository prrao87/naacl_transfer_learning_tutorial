from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import multiprocessing
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from tqdm import tqdm
from typing import Tuple

from pytorch_transformers import BertTokenizer

TEXT_COL, LABEL_COL = 'text', 'label'
MAX_LENGTH = 256
BATCH_SIZE = 16
LOG_DIR = "./logs/"
CACHE_DIR = "./cache/"
DATASET_DIR = "./data/sst2"
# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_cpu = multiprocessing.cpu_count()


def read_sst2(data_dir, colnames=[TEXT_COL, LABEL_COL]):
    datasets = {}
    for t in ["train", "dev", "test"]:
        df = pd.read_csv(os.path.join(data_dir, f"{t}.tsv"), header=None, sep='\t', names=colnames)
        datasets[t] = df[[LABEL_COL, TEXT_COL]]
    return datasets


class TextProcessor:
    def __init__(self, tokenizer, label2id: dict, clf_token, pad_token, max_length):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.clf_token = clf_token
        self.pad_token = pad_token

    def encode(self, input):
        return list(self.tokenizer.convert_tokens_to_ids(o) for o in input)

    def process(self, item: Tuple[str, str]):
        "Convert text (item[0]) to sequence of IDs and label (item[1]) to integer"
        assert len(item) == 2   # Need a row of text AND labels
        label, text = item[0], item[1]
        assert isinstance(text, str)   # Need position 1 of input to be of type(str)
        inputs = self.tokenizer.tokenize(text)
        # Trim or pad dataset
        if len(inputs) >= self.max_length:
            inputs = inputs[:self.max_length - 1]
            ids = self.encode(inputs) + [self.clf_token]
        else:
            pad = [self.pad_token] * (self.max_length - len(inputs) - 1)
            ids = self.encode(inputs) + [self.clf_token] + pad

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
    "Process rows in pd.DataFrame using n_cpus and return a DataLoader"

    tqdm.pandas()
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        result = list(
            tqdm(executor.map(process_row, repeat(processor), df.iterrows(), chunksize=8192),
                 desc=f"Processing {len(df)} examples on {n_cpu} cores",
                 total=len(df)))

    features = [r[0] for r in result]
    labels = [r[1] for r in result]

    dataset = TensorDataset(torch.tensor(features, dtype=torch.long),
                            torch.tensor(labels, dtype=torch.long))

    if valid_pct is not None:
        valid_size = int(valid_pct * len(df))
        train_size = len(df) - valid_size
        valid_dataset, train_dataset = random_split(dataset, [valid_size, train_size])
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, valid_loader

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=shuffle,
                             pin_memory=torch.cuda.is_available())
    return data_loader


if __name__ == "__main__":
    # Config settings
    FineTuningConfig = namedtuple('FineTuningConfig', field_names="num_classes, dropout, \
                                    init_range, batch_size, lr, max_norm, n_epochs, n_warmup, \
                                    valid_pct, gradient_acc_steps, device, log_dir")

    finetuning_config = FineTuningConfig(2, 0.05, 0.02, BATCH_SIZE, 6.5e-5, 1.0, 2,
                                         10, 0.1, 2, device, LOG_DIR)

    datasets = read_sst2(DATASET_DIR)
    labels = list(set(datasets["train"][LABEL_COL].tolist()))
    label2int = {label: i for i, label in enumerate(labels)}
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    clf_token = tokenizer.vocab['[CLS]']  # classifier token
    pad_token = tokenizer.vocab['[PAD]']  # pad token
    processor = TextProcessor(tokenizer, label2int, clf_token, pad_token, max_length=MAX_LENGTH)

    train_dl, valid_dl = create_dataloader(datasets["dev"], processor,
                                           batch_size=finetuning_config.batch_size,
                                           valid_pct=finetuning_config.valid_pct)
    test_dl = create_dataloader(datasets["test"], processor, 
                                batch_size=finetuning_config.batch_size,
                                valid_pct=None)

    print(len(train_dl), len(test_dl))
