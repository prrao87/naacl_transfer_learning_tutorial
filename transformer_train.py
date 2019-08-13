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
from pytorch_transformers import BertTokenizer, cached_path
from pytorch_transformers.optimization import AdamW
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy 
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import CosineAnnealingScheduler, PiecewiseLinear, create_lr_scheduler_with_warmup, ProgressBar
import torch.nn.functional as F

from transformer_utils import (TextProcessor, read_sst2, process_row, create_dataloader, 
                               get_and_tokenize_dataset)
from transformer_model import Transformer, TransformerWithClfHead, get_num_params

TEXT_COL, LABEL_COL = 'text', 'label'
MAX_LENGTH = 256
BATCH_SIZE = 16
LOG_DIR = "./logs/"
CACHE_DIR = "./cache/"
DATASET_DIR = "./data/sst2"
# Define device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_cpu = cpu_count()

# Config settings
FineTuningConfig = namedtuple('FineTuningConfig', field_names="num_classes, dropout, \
                                init_range, batch_size, lr, max_norm, n_epochs, n_warmup, \
                                valid_pct, gradient_acc_steps, device, log_dir")

finetuning_config = FineTuningConfig(2, 0.05, 0.02, BATCH_SIZE, 6.5e-5, 1.0, 2,
                                     10, 0.1, 2, device, LOG_DIR)


def load_pretrained_model():
    "download pre-trained model and config"
    state_dict = torch.load(cached_path("https://s3.amazonaws.com/models.huggingface.co/"
                                        "naacl-2019-tutorial/model_checkpoint.pth"), map_location='cpu')
    config = torch.load(cached_path("https://s3.amazonaws.com/models.huggingface.co/"
                                    "naacl-2019-tutorial/model_training_args.bin"))
    # Initialize model: Transformer base + classifier head
    model = TransformerWithClfHead(config=config, fine_tuning_config=finetuning_config).to(finetuning_config.device)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Parameters discarded from the pretrained model: {incompatible_keys.unexpected_keys}")
    print(f"Parameters added in the model: {incompatible_keys.missing_keys}")

    return model, state_dict, config


def train():
    "Trainer"
    datasets = read_sst2(DATASET_DIR)

    # Get list of labels from training data
    labels = list(set(datasets["train"][LABEL_COL].tolist()))
    # Map labels to dict using zero-indexing
    label2int = {label: i for i, label in enumerate(labels)}

    # Use BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    # Initialize TextProcessor
    processor = TextProcessor(tokenizer, label2int, max_length=MAX_LENGTH)

    # Create train, validation and test dataloaders
    train_dl, valid_dl = create_dataloader(datasets["dev"], processor,
                                           batch_size=finetuning_config.batch_size,
                                           valid_pct=finetuning_config.valid_pct)
    # val_dl = create_dataloader(datasets["dev"], processor,
    #                            batch_size=finetuning_config.batch_size,
    #                            valid_pct=None)
    test_dl = create_dataloader(datasets["test"], processor, 
                                batch_size=finetuning_config.batch_size,
                                valid_pct=None)

    # Define pretrained model and optimizer
    model, state_dict, config = load_pretrained_model()
    optimizer = AdamW(model.parameters(), lr=finetuning_config.lr, correct_bias=False)

    def update(engine, batch):
        "update function for training"
        model.train()
        inputs, labels = (t.to(finetuning_config.device) for t in batch)
        inputs = inputs.transpose(0, 1).contiguous()  # [S, B]
        _, loss = model(inputs,
                        clf_tokens_mask=(inputs == tokenizer.vocab[processor.CLS]), 
                        clf_labels=labels)
        loss = loss / finetuning_config.gradient_acc_steps
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), finetuning_config.max_norm)
        if engine.state.iteration % finetuning_config.gradient_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def inference(engine, batch):
        "update function for evaluation"
        model.eval()
        with torch.no_grad():
            batch, labels = (t.to(finetuning_config.device) for t in batch)
            inputs = batch.transpose(0, 1).contiguous()
            logits = model(inputs,
                           clf_tokens_mask=(inputs == tokenizer.vocab[processor.CLS]),
                           padding_mask=(batch == tokenizer.vocab[processor.PAD]))
        return logits, labels

    def predict(model, tokenizer, int2label, input="test"):
        "predict `input` with `model`"
        tok = tokenizer.tokenize(input)
        ids = tokenizer.convert_tokens_to_ids(tok) + [tokenizer.vocab['[CLS]']]
        tensor = torch.tensor(ids, dtype=torch.long)
        tensor = tensor.to(device)
        tensor = tensor.reshape(1, -1)
        tensor_in = tensor.transpose(0, 1).contiguous() # [S, 1]
        logits = model(tensor_in,
                       clf_tokens_mask=(tensor_in == tokenizer.vocab['[CLS]']),
                       padding_mask=(tensor == tokenizer.vocab['[PAD]']))
        val, _ = torch.max(logits, 0)
        val = F.softmax(val, dim=0).detach().cpu().numpy()    
        return {int2label[val.argmax()]: val.max(),
                int2label[val.argmin()]: val.min()}

    trainer = Engine(update)
    evaluator = Engine(inference)

    # add metric to evaluator
    Accuracy().attach(evaluator, "accuracy")

    # add evaluator to trainer: eval on valid set after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(valid_dl)
        print(f"validation epoch: {engine.state.epoch} acc: {100*evaluator.state.metrics['accuracy']}")
     
    # lr schedule: linearly warm-up to lr and then to zero
    scheduler = PiecewiseLinear(optimizer, 'lr', [(0, 0.0), (finetuning_config.n_warmup, finetuning_config.lr),
                                (len(train_dl) * finetuning_config.n_epochs, 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # add progressbar with loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    # save checkpoints and finetuning config
    checkpoint_handler = ModelCheckpoint(finetuning_config.log_dir, 'finetuning_checkpoint', 
                                         save_interval=1, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'imdb_model': model})

    int2label = {i: label for label, i in label2int.items()}

    # save metadata
    torch.save({
        "config": config,
        "config_ft": finetuning_config,
        "int2label": int2label
    }, os.path.join(finetuning_config.log_dir, "metadata.bin"))

    # Run trainer
    trainer.run(train_dl, max_epochs=3)
    # Evaluate
    evaluator.run(test_dl)
    print(f"test results - acc: {100*evaluator.state.metrics['accuracy']:.3f}")
    # save model weights
    torch.save(model.state_dict(), os.path.join(finetuning_config.log_dir, "model_weights.pth"))


if __name__ == "__main__":
    train()
