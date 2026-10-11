import os

import numpy as np

from config import load_config
from tokenizer import BPETokenizer


def count_model_params(seq_len, vocab_size, d_model, num_layers):
    """Given LM config calculate total number of parameters"""
    ffw_size = 4 * d_model  # in the number of intermediate features is always 4*d_model
    # token and position embeddings
    embeddings = d_model * vocab_size + d_model * seq_len
    # transformer blocks
    attention = 3 * d_model**2 + 3 * d_model  # weights and biases
    attproj = d_model**2 + d_model
    ffw = d_model * (ffw_size) + ffw_size
    ffwproj = ffw_size * d_model + d_model
    layernorms = 2 * 2 * d_model
    # dense
    ln_f = 2 * d_model
    dense = d_model * vocab_size  # note: no bias here
    # note: embeddings are not included in the param count!
    total_params = (
        num_layers * (attention + attproj + ffw + ffwproj + layernorms) + ln_f + dense
    )
    return total_params, embeddings


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


def calculate_optimal_parameters(budget_hours):
    peak = 65e12  # T4 fp16 tensor peak

    mfu = 0.25  # assumption: measure this
    C = peak * mfu * budget_hours * 3600

    # Chinchilla-optimal: C = 6*N*D with D = 20*N  ->  C = 120*N^2
    N = (C / 120) ** 0.5
    D = 20 * N

    loss = approx_loss(N, D)

    return N, D, loss


def calculate_budget_and_model_size(train_tokens):
    # C = 6*N*D
    # D = 20*N
    # C -> compute budget in FLOPs
    # N -> number of model parameters
    # D -> number of training tokens

    optimal_params = train_tokens / 20

    C = 6 * optimal_params * train_tokens

    budget_hours = C / (65e12 * 0.25 * 3600)

    loss = approx_loss(optimal_params, train_tokens)

    return budget_hours, optimal_params, loss


def scaling_report(config_path):
    config_path = os.path.join("configs", f"{config_path}.yaml")
    cfg = load_config(config_path)

    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    valid_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    tokenizer = BPETokenizer("data/polish_bpe_8k.json")

    model_params, embedding_params = count_model_params(
        seq_len=cfg.model.block_size,
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.model.n_embed,
        num_layers=cfg.model.n_blocks,
    )

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
    print(f"Embedding parameters: {embedding_params / 1e6:.2f} M")
    print(f"Model parameters: {model_params / 1e6:.2f} M")
    print(f"Total parameters: {(embedding_params + model_params) / 1e6:.2f} M")
    print(f"Token to parameters ratio: {len(train_data) / model_params:.2f}")
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
    print(f"Expected loss: {approx_loss(model_params, len(train_data)):.2f}")
    print("-----------------\n")

    print("OPTIMAL TRAINING CONFIGURATION (for given training tokens)")
    print("-----------------")
    print(f"Train tokens: {len(train_data) / 1e6:.2f} M")
    print(f"Validation tokens: {len(valid_data) / 1e6:.2f} M")

    hours, optimal_params, expected_loss = calculate_budget_and_model_size(
        len(train_data)
    )
    print(f"Optimal parameters: {optimal_params / 1e6:.2f} M")
    print(f"Expected Training time: {hours:.2f} hours")
    print(f"Expected loss: {expected_loss:.2f}")
    print("-----------------\n")


if __name__ == "__main__":
    scaling_report("base")
