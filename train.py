import argparse
import logging
import os
import warnings

import mlflow
import mlflow.pytorch
import numpy as np
import torch

from config import load_config
from dataloaderlite import DataLoaderLite
from display import TrainingDisplay
from model import LanguageModel
from tokenizer import BPETokenizer  # , WordTokenizer
from utils import (
    calculate_training_steps,
    chinchilla_plan,
    count_model_params,
    detect_gpu,
)

torch.manual_seed(1337)

# logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")


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


# @torch.no_grad()
# def estimate_loss(train_data, valid_data, config, display):
#     out = {}
#     model.eval()

#     for split in ["train", "valid"]:
#         losses = torch.zeros(config.eval.eval_iters)
#         display.eval_start(split)
#         for k in range(config.eval.eval_iters):
#             X, Y = get_batch(split, train_data, valid_data, config)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#             display.eval_tick()

#         out[split] = losses.mean()

#     display.eval_end()
#     model.train()
#     return out


@torch.no_grad()
def estimate_loss(model, train_data, valid_data, split, config, display):
    model.eval()

    losses = torch.zeros(config.eval.eval_iters)

    display.eval_start(f"{split} loss")
    for k in range(config.eval.eval_iters):
        X, Y = get_batch(split, train_data, valid_data, config)
        _, loss = model(X, Y)
        losses[k] = loss.item()
        display.eval_tick()

    display.eval_end()
    model.train()
    return losses.mean()


@torch.no_grad()
def evaluate_full_validation(model, val_loader, device):
    model.eval()
    val_loader.current_position = 0
    total_batches = len(val_loader.data) // val_loader.tokens_per_batch

    losses = []
    for _ in range(total_batches):
        X, Y = val_loader.next_batch(device=device)
        _, loss = model(X, Y)
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else 0.0


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

    train_loader = DataLoaderLite(
        data=train_data,
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
    )
    valid_loader = DataLoaderLite(
        data=valid_data,
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
    )
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

    params = count_model_params(model)
    gpu = detect_gpu()
    plan = chinchilla_plan(params["total"], len(train_data), gpu["peak_flops"])
    hours = plan["hours"]
    expected_loss = plan["loss"]

    calculated_training_steps = calculate_training_steps(
        num_tokens=len(train_data),
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
        epochs=1,
    )

    calculated_eval_interval = max(1, calculated_training_steps // 20)

    display = TrainingDisplay(
        cfg=cfg,
        vocab_size=tokenizer.vocab_size,
        params=params,
        plan=plan,
        gpu=gpu,
        training_tokens=len(train_data),
        validation_tokens=len(valid_data),
        training_steps=calculated_training_steps,
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
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        mlflow.log_params(
            params=cfg.model.model_dump()
            | cfg.train.model_dump()
            | cfg.eval.model_dump()
            | {"total_params": params["total"]}
            | {
                "embedding_params": params["embedding"],
                "non_embedding_params": params["non_embedding"],
                "head_params": params["head"],
                "calulated_training_steps": calculated_training_steps,
                "calculated_eval_interval": calculated_eval_interval,
                "expected_loss": expected_loss,
                "expected_training_hours": hours if hours is not None else -1,
                "gpu_name": gpu["name"],
                "gpu_dtype": gpu["dtype"],
            }
        )

        display.start_training()

        for iter in range(calculated_training_steps):
            if iter % calculated_eval_interval == 0:
                # losses = estimate_loss(train_data, valid_data, cfg, display)

                # display.log_eval(
                #     step=iter, train_loss=losses["train"], val_loss=losses["valid"]
                # )
                # mlflow.log_metrics(
                #     {"train_loss": losses["train"], "val_loss": losses["valid"]},
                #     step=iter,
                # )

                train_loss = estimate_loss(
                    model, train_data, valid_data, "train", cfg, display
                )
                val_loss = estimate_loss(
                    model, train_data, valid_data, "validation", cfg, display
                )

                display.log_eval(step=iter, train_loss=train_loss, val_loss=val_loss)
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=iter,
                )

            # sample a batch of data
            # xb, yb = get_batch("train", train_data, valid_data, cfg)
            xb, yb = train_loader.next_batch(device=device)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            optimizer.step()

            display.log_step(loss=loss.item())

        # final_losses = estimate_loss(train_data, valid_data, cfg, display)
        train_loss = estimate_loss(model, train_data, valid_data, "train", cfg, display)
        full_valid_loss = evaluate_full_validation(
            model=model, val_loader=valid_loader, device=device
        )
        mlflow.log_metrics(
            {"train_loss": train_loss, "val_loss": full_valid_loss},
            step=calculated_training_steps,
        )

        display.finish_training()

        # 1. local checkpoint
        os.makedirs("runs/checkpoints", exist_ok=True)
        ckpt_path = f"runs/checkpoints/{cfg.run_name}.pt"

        display.print_summary(
            final_train_loss=train_loss,
            final_validation_loss=full_valid_loss,
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

        # Create predictions
        model.eval()
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        predicted_tokens = model.generate(idx, max_new_tokens=600)[0].tolist()
        display.print_generation(tokenizer.decode(predicted_tokens))

        # 2. also track in MLflow - cloudpickle avoids pt2 tracing
        # (forward returns a tuple, generate() samples - bad fit for tracing)
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")
        mlflow.log_artifact(ckpt_path)
