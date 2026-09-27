import logging
import os
import sys

import mlflow
import torch

from config import load_config
from model import LanguageModel
from tokenizer import CharacterTokenizer  # , WordTokenizer

torch.manual_seed(1337)

logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_batch(split, train_data, valid_data, config):
    # generate a small batch of data of inputs x and targets y

    data = train_data if split == "train" else valid_data
    ix = torch.randint(len(data) - config.model.block_size, (config.train.batch_size,))
    x = torch.stack([data[i : i + config.model.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.model.block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(train_data, valid_data, config):
    out = {}
    model.eval()
    for split in ["train", "valid"]:
        losses = torch.zeros(config.eval.eval_iters)
        for k in range(config.eval.eval_iters):
            X, Y = get_batch(split, train_data, valid_data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out


if __name__ == "__main__":
    # Read training configuration
    config_path = os.path.join("configs", f"{sys.argv[1]}.yaml")
    cfg = load_config(config_path)
    # Read dataset

    tokenizer = CharacterTokenizer()
    # tokenizer = WordTokenizer()
    # Train - validation split

    data = torch.tensor(tokenizer.encode(tokenizer.text), dtype=torch.long)
    n = int(0.9 * len(data))

    train_data = data[:n]
    valid_data = data[n:]

    # build model

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embed=cfg.model.n_embed,
        block_size=cfg.model.block_size,
        n_blocks=cfg.model.n_blocks,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(num_params / 1e6, "M parameters")

    # Initialize optimizer and start training loop

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    mlflow.set_tracking_uri("sqlite:///runs/mlflow.db")

    if mlflow.get_experiment_by_name(cfg.experiment) is None:
        mlflow.create_experiment(cfg.experiment, artifact_location="runs/artifacts")
    mlflow.set_experiment(cfg.experiment)

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(10)

    with mlflow.start_run(run_name=cfg.run_name, description=cfg.description) as run:
        mlflow.log_params(
            params=cfg.model.model_dump()
            | cfg.train.model_dump()
            | cfg.eval.model_dump()
            | {"total_params": num_params}
        )
        for iter in range(cfg.train.training_steps):
            if iter % cfg.eval.eval_interval == 0:
                losses = estimate_loss(train_data, valid_data, cfg)
                print(
                    f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}"
                )

                mlflow.log_metrics(
                    {"train_loss": losses["train"], "val_loss": losses["valid"]},
                    step=iter,
                )

            # sample a batch of data
            xb, yb = get_batch("train", train_data, valid_data, cfg)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

        final_losses = estimate_loss(train_data, valid_data, cfg)
        mlflow.log_metrics(
            {"train_loss": losses["train"], "val_loss": losses["valid"]},
            step=cfg.train.training_steps,
        )
    # Create predictions

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    predicted_tokens = model.generate(idx, max_new_tokens=600)[0].tolist()

    print(tokenizer.decode(predicted_tokens))
