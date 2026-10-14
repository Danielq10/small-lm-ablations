import argparse
import logging
import os

import mlflow
import numpy as np
import torch

from config import load_config
from display import TrainingDisplay
from model import LanguageModel
from tokenizer import BPETokenizer  # , WordTokenizer
from utils import (
    calculate_budget_and_model_size,
    calculate_training_steps,
    count_model_params,
)

torch.manual_seed(1337)

logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_batch(split, train_data, valid_data, config):
    # generate a small batch of data of inputs x and targets y

    data = train_data if split == "train" else valid_data
    ix = torch.randint(len(data) - config.model.block_size, (config.train.batch_size,))
    x = torch.stack(
        [
            torch.from_numpy((data[i : i + config.model.block_size]).astype(np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1 : i + 1 + config.model.block_size]).astype(np.int64)
            )
            for i in ix
        ]
    )
    x = x.to(device)
    y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(train_data, valid_data, config, display):
    out = {}
    model.eval()

    for split in ["train", "valid"]:
        losses = torch.zeros(config.eval.eval_iters)
        display.eval_start(split)
        for k in range(config.eval.eval_iters):
            X, Y = get_batch(split, train_data, valid_data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            display.eval_tick()

        out[split] = losses.mean()

    display.eval_end()
    model.train()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--config", type=str, help="Config name without .yaml")
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint  .pt to resume training from",
        default=None,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test training on a subset of data",
        default=False,
    )
    args = parser.parse_args()

    # Read training configuration
    config_path = os.path.join("configs", f"{args.config}.yaml")
    cfg = load_config(config_path)
    # Read dataset

    tokenizer = BPETokenizer("data/polish_bpe_8k.json")

    if not args.test:
        # Load memory-mapped binary datasets
        train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
        valid_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")

    else:
        # Jsut load validation data (11M tokens) and split it

        data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
        split_idx = int(len(data) * 0.95)

        train_data = data[:split_idx]
        valid_data = data[split_idx:]

    # build model

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embed=cfg.model.n_embed,
        block_size=cfg.model.block_size,
        n_blocks=cfg.model.n_blocks,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        pos_encoding=cfg.model.pos_encoding,
    )
    model = model.to(device)

    # Initialize optimizer and start training loop

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    # Check if we are resuming from checkpoint
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume} ...")

        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        # 1. Restore weights
        model.load_state_dict(ckpt["model_state"])

        # 2. Restore optimizer momentum buffers
        optimizer.load_state_dict(ckpt["optimizer_state"])

        # Retrieve iteration count
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    model_params, embedding_params = count_model_params(
        seq_len=cfg.model.block_size,
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.model.n_embed,
        num_layers=cfg.model.n_blocks,
    )

    calculated_training_steps = calculate_training_steps(
        num_tokens=len(train_data),
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
        epochs=1,
    )

    calculated_eval_interval = max(1, calculated_training_steps // 20)

    hours, optimal_params, expected_loss = calculate_budget_and_model_size(
        len(train_data)
    )

    display = TrainingDisplay(
        cfg=cfg,
        vocab_size=tokenizer.vocab_size,
        embedding_params=embedding_params,
        model_params=model_params,
        training_tokens=len(train_data),
        validation_tokens=len(valid_data),
        training_steps=calculated_training_steps,
        expected_training_time=hours,
        optimal_model_params=optimal_params,
        expected_loss=expected_loss,
    )

    display.print_header()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///runs/mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    if mlflow.get_experiment_by_name(cfg.experiment) is None:
        if tracking_uri.startswith("sqlite"):
            mlflow.create_experiment(cfg.experiment, artifact_location="runs/artifacts")
        else:
            mlflow.create_experiment(cfg.experiment)
    mlflow.set_experiment(cfg.experiment)

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(10)

    with mlflow.start_run(run_name=cfg.run_name, description=cfg.description) as run:
        mlflow.log_params(
            params=cfg.model.model_dump()
            | cfg.train.model_dump()
            | cfg.eval.model_dump()
            | {"total_params": (embedding_params + model_params)}
            | {
                "embedding_params": embedding_params,
                "model_params": model_params,
                "calulated_training_steps": calculated_training_steps,
                "calculated_eval_interval": calculated_eval_interval,
                "expected_loss": expected_loss,
                "expected_training_hours": hours,
            }
        )

        display.start_training()

        for iter in range(calculated_training_steps):
            if iter % calculated_eval_interval == 0:
                losses = estimate_loss(train_data, valid_data, cfg, display)

                display.log_eval(
                    step=iter, train_loss=losses["train"], val_loss=losses["valid"]
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

            display.log_step(loss=loss.item())

        final_losses = estimate_loss(train_data, valid_data, cfg, display)
        mlflow.log_metrics(
            {"train_loss": final_losses["train"], "val_loss": final_losses["valid"]},
            step=calculated_training_steps,
        )

        display.finish_training()

        # 1. local checkpoint
        os.makedirs("runs/checkpoints", exist_ok=True)
        ckpt_path = f"runs/checkpoints/{cfg.run_name}.pt"

        display.print_summary(
            final_train_loss=final_losses["train"],
            final_validation_loss=final_losses["valid"],
            ckpt_path=ckpt_path,
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.model_dump(),
                "step": cfg.train.training_steps,
                "vocab_size": tokenizer.vocab_size,
            },
            ckpt_path,
        )

        # 2. also track in MLflow - cloudpickle avoids pt2 tracing
        # (forward returns a tuple, generate() samples - bad fit for tracing)
        model.eval()
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        mlflow.log_artifact(ckpt_path)
        # Create predictions

    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    predicted_tokens = model.generate(idx, max_new_tokens=600)[0].tolist()

    print(tokenizer.decode(predicted_tokens))
