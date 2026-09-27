import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    n_embed: int = 384
    n_blocks: int = 6
    n_heads: int = 6
    dropout: float = Field(0.2, ge=0, lt=1, description="Dropout rate")
    block_size: int = 256


class TrainingConfig(BaseModel):
    batch_size: int = 64
    learning_rate: float = 3e-4
    training_steps: int = 5000


class EvaluationConfig(BaseModel):
    eval_iters: int = 200
    eval_interval: int = 500


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiment: str = "arch-ablation"
    run_name: str = "char-tokenizer"
    description: str = "Training a small transformer LM with character tokenizer"
    seed: int = 1337


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        return Config.model_validate(yaml.safe_load(f))
