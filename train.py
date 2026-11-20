import argparse
import logging
import math
import os
import time
import warnings

import mlflow
import mlflow.pytorch
import numpy as np
import torch

from config import TrainingConfig, load_config
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
    flops_per_token,
)

torch.set_float32_matmul_precision("high")

# logging.getLogger("mlflow.system_metrics.metrics.gpu_monitor").setLevel(logging.ERROR)
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="mlflow")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_batch(split, train_data, valid_data, config, generator=None):
    # generate a small batch of data of inputs x and targets y

    data = train_data if split == "train" else valid_data
    ix = torch.randint(
        len(data) - config.model.block_size,
        (config.train.batch_size,),
        generator=generator,
    )
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


def get_lr(
    step: int, max_lr: float, total_steps: int, warmup_frac: float, min_lr_ratio: float
) -> float:
    warmup_steps = max(1, int(warmup_frac * total_steps))
    min_lr = min_lr_ratio * max_lr
    # 1) Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) Cosine decay down to min_lr
    if step >= total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizer(
    model: LanguageModel, train_cfg: TrainingConfig
) -> tuple[torch.optim.Optimizer, int, int]:
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)

    optimizer = torch.optim.AdamW(
        params=[
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )
    return optimizer, num_decay_params, num_nodecay_params


@torch.no_grad()
def estimate_loss(model, train_data, valid_data, split, iters, config, display):
    model.eval()

    # Use a fixed local RNG so every eval call and every ablation variant
    # evaluates on the exact same batches regardless of model init RNG draws.
    eval_seed = config.seed if split == "train" else config.seed + 1
    eval_gen = torch.Generator().manual_seed(eval_seed)

    losses = torch.zeros(iters)

    display.eval_start(f"{split} loss")
    for k in range(iters):
        X, Y = get_batch(split, train_data, valid_data, config, generator=eval_gen)
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
    total_batches = max(0, (len(val_loader.data) - 1) // val_loader.tokens_per_batch)

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
    raw_model = model.to(device)

    # compile the model to be faster
    model = torch.compile(model)

    # Initialize optimizer and start training loop

    optimizer, num_decay_params, num_nodecay_params = configure_optimizer(
        model=model, train_cfg=cfg.train
    )

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
                "train_tokens": len(train_data),
                "val_tokens": len(valid_data),
                "calculated_training_steps": calculated_training_steps,
                "calculated_eval_interval": calculated_eval_interval,
                "calculated_eval_iterations": calculated_eval_iters,
                "expected_loss": expected_loss,
                "expected_training_hours": hours if hours is not None else -1,
                "gpu_name": gpu["name"],
                "gpu_dtype": gpu["dtype"],
                "seed": cfg.seed,
                "config_name": args.config,
                "is_test_run": args.test,
                "num_decay_params": num_decay_params,
                "num_nodecay_params": num_nodecay_params,
            }
        )

        display.start_training()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train_start = time.perf_counter()
        best_val_loss = float("inf")

        training_window_loss_sum = 0
        train_time = 0
        compile_time = 0

        for iter in range(calculated_training_steps):
            lr = get_lr(
                step=iter,
                max_lr=cfg.train.learning_rate,
                total_steps=calculated_training_steps,
                warmup_frac=cfg.train.warmup_frac,
                min_lr_ratio=cfg.train.min_lr_ratio,
            )

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
                best_val_loss = min(best_val_loss, val_loss)

                display.log_eval(step=iter, train_loss=train_loss, val_loss=val_loss)
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss},
                    step=iter,
                )

            t0 = time.perf_counter()

            # sample a batch of data
            xb, yb = train_loader.next_batch(device=device)

            # evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            # Rescale gradients so their global L2 norm is at most 1.0 (guards
            # against loss spikes from occasional huge gradients)
            # clip_grad_norm_ returns the norm *before* clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=cfg.train.grad_clip
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

            loss_val = loss.item()

            if iter == 0:
                compile_time = time.perf_counter() - t0
            else:
                train_time += time.perf_counter() - t0

            display.log_step(loss=loss_val)

            training_window_loss_sum += loss_val

            if (iter + 1) % cfg.train.log_interval == 0:
                mlflow.log_metrics(
                    {
                        "train_loss_step": training_window_loss_sum
                        / cfg.train.log_interval,
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                    },
                    step=iter,
                    synchronous=False,
                )
                training_window_loss_sum = 0

        train_loss = estimate_loss(
            model, train_data, valid_data, "train", calculated_eval_iters, cfg, display
        )
        full_valid_loss = evaluate_full_validation(
            model=model, val_loader=valid_loader, device=device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_elapsed = time.perf_counter() - t_train_start
        best_val_loss = min(best_val_loss, full_valid_loss)
        tokens_seen = (
            cfg.train.batch_size * cfg.model.block_size * calculated_training_steps
        )

        tokens_per_step = cfg.train.batch_size * cfg.model.block_size
        train_tokens_per_sec = (
            tokens_per_step * (calculated_training_steps - 1) / train_time
        )

        final_metrics = {
            "train_loss": train_loss,
            "final_val_loss": full_valid_loss,
            "final_val_perplexity": math.exp(full_valid_loss),
            "best_val_loss": best_val_loss,
            "lr": lr,
            "total_time_sec": train_elapsed,
            "training_time_sec": train_time,
            "compile_time_sec": compile_time,
            "avg_tokens_per_sec": train_tokens_per_sec,
        }

        if gpu["peak_flops"]:
            mfu = (
                flops_per_token(
                    params=params,
                    n_embed=cfg.model.n_embed,
                    block_size=cfg.model.block_size,
                    n_blocks=cfg.model.n_blocks,
                )
                * train_tokens_per_sec
                / gpu["peak_flops"]
            )
            final_metrics["measured_mfu"] = mfu
        else:
            mfu = None

        mlflow.log_metrics(final_metrics, step=calculated_training_steps)

        display.finish_training()

        # 1. local checkpoint (weights + config only; optimizer state omitted)
        os.makedirs("runs/checkpoints", exist_ok=True)
        ckpt_path = f"runs/checkpoints/{cfg.run_name}.pt"

        display.print_summary(
            train_elapsed=train_elapsed,
            mfu=mfu,
            final_train_loss=train_loss,
            final_validation_loss=full_valid_loss,
            ckpt_path=ckpt_path,
            train_tokens_per_sec=train_tokens_per_sec,
        )

        torch.save(
            {
                "model_state": raw_model.state_dict(),
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
            raw_model, name="model", serialization_format="pickle"
        )
        mlflow.log_artifact(ckpt_path)

        # 3. Generate MLflow traces linked to both this run and the logged model
        samples = generate_model_traces(
            model=raw_model,
            tokenizer=tokenizer,
            device=device,
            model_name=cfg.run_name,
            model_id=model_info.model_id,
        )
        if samples:
            display.print_generation(samples[0]["completion"])
