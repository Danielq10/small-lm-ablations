import os

import numpy as np
import torch

from config import load_config
from model import LanguageModel
from tokenizer import BPETokenizer


def count_model_params(model):
    """Count parameters directly from the model (source of truth).

    Returns a dict with:
      total          - every parameter (Chinchilla's N)
      embedding      - token (+ learned position) embedding tables
      head           - lm_head weights and bias
      non_embedding  - transformer blocks + final layer norm (Kaplan's N)
    """

    def n(m):
        return sum(p.numel() for p in m.parameters())

    embedding = n(model.token_embedding_table)
    if hasattr(model, "position_embedding_table"):
        embedding += n(model.position_embedding_table)
    head = n(model.lm_head)
    total = n(model)
    return {
        "total": total,
        "embedding": embedding,
        "head": head,
        "non_embedding": total - embedding - head,
    }


def calculate_training_steps(num_tokens, batch_size, block_size, epochs=1):
    """Given dataset size and training config, calculate number of training steps"""
    tokens_per_step = batch_size * block_size
    total_steps = (num_tokens * epochs) // tokens_per_step
    return total_steps


def approx_loss(N, D):
    """
    Approximates loss given N parameters and D dataset size (in tokens),
    per Chinchilla paper.
    """
    E = 1.69  # entropy of natural language, limit of infinite model on infinite data
    A = 406.4
    B = 410.7
    alpha = 0.34
    beta = 0.28
    return A / (N**alpha) + B / (D**beta) + E


MFU = 0.25  # assumption: measure this (the training summary prints the measured value)

# Dense (no sparsity) peak TFLOPs per GPU, matched by substring of the device name.
# "fp32" = CUDA cores, "tf32"/"half" = tensor cores (half = fp16/bf16).
# Taken from NVIDIA datasheets; sparsity figures were halved to get dense numbers.
GPU_PEAK_TFLOPS = [
    ("1050 Ti", {"fp32": 2.14}),  # Pascal consumer card: no tensor cores
    ("T4", {"fp32": 8.1, "half": 65.0}),
    ("L4", {"fp32": 30.3, "tf32": 60.0, "half": 121.0}),
    ("A100", {"fp32": 19.5, "tf32": 156.0, "half": 312.0}),
    # RTX PRO 6000 Blackwell (Colab G4): the 1 PFLOP bf16 datasheet number is the
    # sparse one, so dense is taken as half. tf32 omitted (datasheet unclear).
    ("RTX PRO 6000", {"fp32": 120.0, "half": 500.0}),
]


def detect_gpu():
    """Detect the GPU and the matmul dtype in use, and look up its peak FLOPs.

    Training currently runs in plain fp32 (no autocast); matmuls only use tensor
    cores in tf32 if torch's float32 matmul precision has been lowered.
    Returns {"name", "dtype", "peak_flops"}; peak_flops is None if unknown or CPU.
    """
    if not torch.cuda.is_available():
        return {"name": "cpu", "dtype": "fp32", "peak_flops": None}

    name = torch.cuda.get_device_name()
    dtype = "fp32" if torch.get_float32_matmul_precision() == "highest" else "tf32"
    peak = None
    for key, table in GPU_PEAK_TFLOPS:
        if key.lower() in name.lower():
            tflops = table.get(dtype)
            peak = tflops * 1e12 if tflops is not None else None
            break
    return {"name": name, "dtype": dtype, "peak_flops": peak}


def estimate_hours(flops, peak_flops):
    """Estimated wall-clock hours to spend `flops` at peak_flops * MFU (None if unknown)."""
    if peak_flops is None:
        return None
    return flops / (peak_flops * MFU * 3600)


def chinchilla_plan(n_params, n_tokens, peak_flops=None):
    """Compare the configured model with the Chinchilla-optimal one.

    n_params must be TOTAL parameters (Chinchilla's definition, embeddings included).
      C = 6*N*D;  optimal at fixed compute: D = 20*N  ->  C = 120*N^2
    """
    compute = 6 * n_params * n_tokens
    opt_params = (compute / 120) ** 0.5
    opt_tokens = 20 * opt_params
    return {
        "compute": compute,
        "hours": estimate_hours(compute, peak_flops),
        "loss": approx_loss(n_params, n_tokens),
        "tokens_per_param": n_tokens / n_params,
        "data_limited": n_tokens < 20 * n_params,
        # optimal model for the same compute budget
        "optimal_params": opt_params,
        "optimal_tokens": opt_tokens,
        "optimal_loss": approx_loss(opt_params, opt_tokens),
        # optimal model size for the data we actually have
        "optimal_params_for_data": n_tokens / 20,
    }


def scaling_report(config_path):
    config_path = os.path.join("configs", f"{config_path}.yaml")
    cfg = load_config(config_path)

    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    valid_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    tokenizer = BPETokenizer("data/polish_bpe_8k.json")

    model = LanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embed=cfg.model.n_embed,
        block_size=cfg.model.block_size,
        n_blocks=cfg.model.n_blocks,
        n_heads=cfg.model.n_heads,
        dropout=cfg.model.dropout,
        pos_encoding=cfg.model.pos_encoding,
    )
    params = count_model_params(model)
    gpu = detect_gpu()
    plan = chinchilla_plan(params["total"], len(train_data), gpu["peak_flops"])

    print("CONFIGURED MODEL PARAMS")
    print("-----------------")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"n_embed: {cfg.model.n_embed}")
    print(f"block_size: {cfg.model.block_size}")
    print(f"n_blocks: {cfg.model.n_blocks}")
    print(f"n_heads: {cfg.model.n_heads}")
    print(f"dropout: {cfg.model.dropout}")
    print(f"pos_encoding: {cfg.model.pos_encoding}")
    print("-----------------\n")

    print("CONFIGURED MODEL SUMMARY")
    print("-----------------")
    print(f"Embedding parameters: {params['embedding'] / 1e6:.2f} M")
    print(f"Non-embedding parameters: {params['non_embedding'] / 1e6:.2f} M")
    print(f"Total parameters: {params['total'] / 1e6:.2f} M")
    print(f"Token to parameters ratio: {plan['tokens_per_param']:.2f}")
    print("-----------------\n")

    calculated_training_steps = calculate_training_steps(
        num_tokens=len(train_data),
        batch_size=cfg.train.batch_size,
        block_size=cfg.model.block_size,
        epochs=1,
    )

    calculated_eval_interval = max(1, calculated_training_steps // 20)

    print("CONFIGURED TRAINING CONFIGURATION")
    print("-----------------")
    print(f"Train tokens: {len(train_data) / 1e6:.2f} M")
    print(f"Validation tokens: {len(valid_data) / 1e6:.2f} M")
    print(f"Batch size: {cfg.train.batch_size}")
    print(f"Training steps: {calculated_training_steps}")
    print(f"Eval interval: {calculated_eval_interval}")
    print(f"Expected loss: {plan['loss']:.2f}")
    hours = "n/a" if plan["hours"] is None else f"{plan['hours']:.2f} hours"
    print(f"GPU: {gpu['name']} ({gpu['dtype']})")
    print(f"Expected Training time: {hours}")
    print("-----------------\n")

    print("OPTIMAL MODEL (same compute budget)")
    print("-----------------")
    print(f"Optimal parameters: {plan['optimal_params'] / 1e6:.2f} M")
    print(f"Optimal tokens: {plan['optimal_tokens'] / 1e6:.2f} M")
    print(f"Expected loss: {plan['optimal_loss']:.2f}")
    print(
        f"Optimal parameters for available data: "
        f"{plan['optimal_params_for_data'] / 1e6:.2f} M"
    )
    print("-----------------\n")


if __name__ == "__main__":
    scaling_report("base")
