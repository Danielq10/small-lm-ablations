from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """Transformer model architecture hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    n_embed: int = 384
    n_blocks: int = 6
    n_heads: int = 6
    n_groups: int = 6
    ffn: Literal["mlp", "swiglu"] = "mlp"
    dropout: float = Field(0.2, ge=0, lt=1, description="Dropout rate")
    block_size: int = 256
    pos_encoding: Literal["learned", "rope"] = "learned"

    @model_validator(mode="after")
    def validate_attention_dims(self) -> "ModelConfig":
        """Default n_groups to n_heads (MHA) when omitted and enforce valid head/group splits."""
        if "n_groups" not in self.model_fields_set:
            self.n_groups = self.n_heads
        if self.n_embed % self.n_heads != 0:
            raise ValueError(
                f"n_embed ({self.n_embed}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_groups != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_groups ({self.n_groups})"
            )
        head_size = self.n_embed // self.n_heads
        if self.pos_encoding == "rope" and head_size % 2 != 0:
            raise ValueError(f"head_size ({head_size}) must be even for RoPE")
        return self


class TrainingConfig(BaseModel):
    """Optimizer and batching hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = 64
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_frac: float = 0.05  # % of training steps for warmup
    min_lr_ratio: float = 0.1  # learning_rate/10 per Chinchilla
    grad_clip: float = 1.0  # clip gradients at this value
    log_interval: int = 50  # training loss logging interval


class Config(BaseModel):
    """Top-level experiment configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    experiment: str = "arch-ablation"
    run_name: str = "char-tokenizer"
    description: str = "Training a small transformer LM with character tokenizer"
    seed: int = 1337


def load_config(path: str) -> Config:
    """Load and validate a YAML configuration file."""
    with open(path, "r") as f:
        return Config.model_validate(yaml.safe_load(f))
