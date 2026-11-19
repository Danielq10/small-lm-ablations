import argparse
import logging
import math
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
from tracing import generate_model_traces
from utils import (
    calculate_training_steps,
    chinchilla_plan,
    count_model_params,
    detect_gpu,
)

torch.set_float32_matmul_precision("high")

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


def get_lr(step, max_lr, total_steps):
    warmup_steps = max(1, int(0.05 * total_steps))
    min_lr = 0.1 * max_lr
    # 1) Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) Cosine decay down to min_lr
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, valid_data, split, iters, config, display):
    model.eval()

    losses = torch.zeros(iters)

    display.eval_start(f"{split} loss")
    for k in range(iters):
        X, Y = get_batch(split, train_data, valid_data, config)
        _, loss = model(X, Y)
        losses[k] = loss.item()
        display.eval_tick()

    display.eval_end()
    model.train()
    return losses.mean().item()


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

    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def setup_mlflow_experiment(tracking_uri: str, experiment_name: str) -> str:
    """Configure MLflow tracking URI and guard against local/proxied artifact mismatches."""
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        if tracking_uri.startswith("sqlite"):
            exp_id = mlflow.create_experiment(
                experiment_name, artifact_location="runs/artifacts"
            )
        else:
            exp_id = mlflow.create_experiment(experiment_name)
        exp = mlflow.get_experiment(exp_id)

    if tracking_uri.startswith(
        ("http://", "https://")
    ) and not exp.artifact_location.startswith("mlflow-artifacts:"):
        raise RuntimeError(
            f"Experiment '{experiment_name}' has local artifact_location "
            f"'{exp.artifact_location}', so remote runs via '{tracking_uri}' "
            "would NOT upload artifacts to your MLflow server. Use an experiment "
            "created through `mlflow server` (with `mlflow-artifacts:/`)."
        )
    if tracking_uri.startswith("sqlite") and exp.artifact_location.startswith(
        "mlflow-artifacts:"
    ):
        raise RuntimeError(
            f"Experiment '{experiment_name}' uses proxied '{exp.artifact_location}', "
            "which requires an HTTP tracking URI. Start `mlflow server` and set "
            "MLFLOW_TRACKING_URI=http://127.0.0.1:5000."
        )

    mlflow.set_experiment(experiment_name)
    return exp.experiment_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument(
        "--config", type=str, required=True, help="Config name without .yaml"
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

    torch.manual_seed(cfg.seed)
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
        n_groups=cfg.model.n_groups,
        ffn=cfg.model.ffn,
        dropout=cfg.model.dropout,
        pos_encoding=cfg.model.pos_encoding,
    )
    model = model.to(device)

    # Initialize optimizer and start training loop

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

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

    total_val_batches = calculate_training_steps(
        num_tokens=len(valid_data),
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
        epochs=1,
    )
    calculated_eval_iters = max(1, min(100, total_val_batches // 10))

    display = TrainingDisplay(
        cfg=cfg,
        vocab_size=tokenizer.vocab_size,
        params=params,
        plan=plan,
        gpu=gpu,
        training_tokens=len(train_data),
        validation_tokens=len(valid_data),
        training_steps=calculated_training_steps,
        eval_iters=calculated_eval_iters,
    )

    display.print_header()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///runs/mlflow.db")
    setup_mlflow_experiment(tracking_uri, cfg.experiment)

    # Enable system metrics logging
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(10)

    with mlflow.start_run(run_name=cfg.run_name, description=cfg.description) as run:
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        mlflow.log_params(
            params=cfg.model.model_dump()
            | cfg.train.model_dump()
            | {"total_params": params["total"]}
            | {
                "embedding_params": params["embedding"],
                "non_embedding_params": params["non_embedding"],
                "head_params": params["head"],
                "calculated_training_steps": calculated_training_steps,
                "calculated_eval_interval": calculated_eval_interval,
                "calculated_eval_iterations": calculated_eval_iters,
                "expected_loss": expected_loss,
                "expected_training_hours": hours if hours is not None else -1,
                "gpu_name": gpu["name"],
                "gpu_dtype": gpu["dtype"],
                "seed": cfg.seed,
            }
        )

        display.start_training()

        lr = get_lr(0, cfg.train.learning_rate, calculated_training_steps)

        for iter in range(calculated_training_steps):
            lr = get_lr(iter, cfg.train.learning_rate, calculated_training_steps)

            if iter % calculated_eval_interval == 0:
                train_loss = estimate_loss(
                    model,
                    train_data,
                    valid_data,
                    "train",
                    calculated_eval_iters,
                    cfg,
                    display,
                )
                val_loss = estimate_loss(
                    model,
                    train_data,
                    valid_data,
                    "validation",
                    calculated_eval_iters,
                    cfg,
                    display,
                )

                display.log_eval(step=iter, train_loss=train_loss, val_loss=val_loss)
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, "lr": lr},
                    step=iter,
                )

            # sample a batch of data
            xb, yb = train_loader.next_batch(device=device)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            # Rescale gradients so their global L2 norm is at most 1.0 (guards
            # against loss spikes from occasional huge gradients)
            # clip_grad_norm_ returns the norm *before* clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

            display.log_step(loss=loss.item())

        train_loss = estimate_loss(
            model, train_data, valid_data, "train", calculated_eval_iters, cfg, display
        )
        full_valid_loss = evaluate_full_validation(
            model=model, val_loader=valid_loader, device=device
        )
        mlflow.log_metrics(
            {"train_loss": train_loss, "final_val_loss": full_valid_loss, "lr": lr},
            step=calculated_training_steps,
        )

        display.finish_training()

        # 1. local checkpoint (weights + config only; optimizer state omitted)
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
                "config": cfg.model_dump(),
                "step": calculated_training_steps,
                "vocab_size": tokenizer.vocab_size,
            },
            ckpt_path,
        )

        # 2. also track in MLflow - cloudpickle avoids pt2 tracing
        # (forward returns a tuple, generate() samples - bad fit for tracing)
        model.eval()
        model_info = mlflow.pytorch.log_model(
            model, name="model", serialization_format="pickle"
        )
        mlflow.log_artifact(ckpt_path)

        # 3. Generate MLflow traces linked to both this run and the logged model
        samples = generate_model_traces(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=cfg.run_name,
            model_id=model_info.model_id,
        )
        if samples:
            display.print_generation(samples[0]["completion"])
