import argparse
import multiprocessing
import os
import torch
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, cached_path
from pytorch_transformers.optimization import AdamW
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import PiecewiseLinear, ProgressBar

from transformer_utils import TextProcessor, read_sst2, create_dataloader
from transformer_model import TransformerWithClfHead

PRETRAINED_MODEL_URL = "https://s3.amazonaws.com/models.huggingface.co/naacl-2019-tutorial/"
TEXT_COL, LABEL_COL = 'text', 'label'  # Column names in pd.DataFrame for sst dataset
n_cpu = multiprocessing.cpu_count()


def load_pretrained_model(args):
    "download pre-trained model and config"
    state_dict = torch.load(cached_path(os.path.join(args.model_checkpoint, "model_checkpoint.pth")),
                            map_location='cpu')
    config = torch.load(cached_path(os.path.join(args.model_checkpoint, "model_training_args.bin")))
    # Initialize model: Transformer base + classifier head
    model = TransformerWithClfHead(config=config, fine_tuning_config=args).to(args.device)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Parameters discarded from the pretrained model: {incompatible_keys.unexpected_keys}")
    print(f"Parameters added in the model: {incompatible_keys.missing_keys}")

    return model, state_dict, config


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default=PRETRAINED_MODEL_URL, help="Path to the pretrained model checkpoint")
    parser.add_argument("--dataset_path", type=str, default='./data/sst2', help="Directory to dataset.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path to dataset cache")
    parser.add_argument("--logdir", type=str, default='./logs', help="Path to logs")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for the target classification task")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout for transformer module")
    parser.add_argument("--clf_loss_coef", type=float, default=1, help="If >0 add a classification loss")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--valid_pct", type=float, default=0.1, help="Percentage of test data to use for validation")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--n_warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="Accumulate gradient")
    parser.add_argument("--init_range", type=float, default=0.02, help="Normal initialization standard deviation")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Define pretrained model and optimizer
    model, state_dict, config = load_pretrained_model(args)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=False)
    # Define datasets
    datasets = read_sst2(args.dataset_path)
    labels = list(set(datasets["train"][LABEL_COL].tolist()))
    label2int = {label: i for i, label in enumerate(labels)}
    # Get BertTokenizer for this pretrained model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    clf_token = tokenizer.vocab['[CLS]']  # classifier token
    pad_token = tokenizer.vocab['[PAD]']  # pad token
    processor = TextProcessor(tokenizer, label2int, clf_token, pad_token, max_length=config.num_max_positions)

    train_dl, valid_dl = create_dataloader(datasets["dev"], processor,
                                           batch_size=args.train_batch_size,
                                           valid_pct=args.valid_pct)

    test_dl = create_dataloader(datasets["test"], processor,
                                batch_size=args.valid_batch_size,
                                valid_pct=None)

    def update(engine, batch):
        "update function for training"
        model.train()
        inputs, labels = (t.to(args.device) for t in batch)
        inputs = inputs.transpose(0, 1).contiguous()  # to shape [seq length, batch]
        _, loss = model(inputs,
                        clf_tokens_mask=(inputs == clf_token),
                        clf_labels=labels)
        loss = loss / args.gradient_acc_steps
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_acc_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    def inference(engine, batch):
        "update function for evaluation"
        model.eval()
        with torch.no_grad():
            batch, labels = (t.to(args.device) for t in batch)
            inputs = batch.transpose(0, 1).contiguous()  # to shape [seq length, batch]
            logits = model(inputs,
                           clf_tokens_mask=(inputs == clf_token),
                           padding_mask=(batch == pad_token))
        return logits, labels

    def predict(model, tokenizer, int2label, input="test"):
        "predict `input` with `model`"
        tok = tokenizer.tokenize(input)
        ids = tokenizer.convert_tokens_to_ids(tok) + [tokenizer.vocab['[CLS]']]
        tensor = torch.tensor(ids, dtype=torch.long)
        tensor = tensor.to(args.device)
        tensor = tensor.reshape(1, -1)
        tensor_in = tensor.transpose(0, 1).contiguous()  # to shape [seq length, batch]
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
    scheduler = PiecewiseLinear(optimizer, 'lr', [(0, 0.0), (args.n_warmup, args.lr),
                                (len(train_dl) * args.n_epochs, 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # add progressbar with loss
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    ProgressBar(persist=True).attach(trainer, metric_names=['loss'])

    # save checkpoints and finetuning config
    checkpoint_handler = ModelCheckpoint(args.logdir, 'finetuning_checkpoint',
                                         save_interval=1, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'sst2_model': model})

    int2label = {i: label for label, i in label2int.items()}

    # save metadata
    torch.save({
        "config": config,
        "config_ft": args,
        "int2label": int2label
    }, os.path.join(args.logdir, "metadata.bin"))

    # Run trainer
    trainer.run(train_dl, max_epochs=3)
    # Evaluate
    evaluator.run(test_dl)
    print(f"test results - acc: {100*evaluator.state.metrics['accuracy']:.3f}")
    # save model weights
    torch.save(model.state_dict(), os.path.join(args.logdir, "model_weights.pth"))


if __name__ == "__main__":
    train()
