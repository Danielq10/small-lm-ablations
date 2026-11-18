from contextlib import nullcontext
from typing import Any

import mlflow
import torch
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from model import LanguageModel
from tokenizer import BPETokenizer

# Use "</s>" (token 3) for unconditional generation: prepare_data_hf.py appends
# it as the document separator, whereas <pad>/<unk>/<s> never appear in the corpus.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "</s>",
    "Dawno temu w pewnej",
    "Polska to kraj",
    "W roku 1989",
)
DEFAULT_MAX_NEW_TOKENS = 200


@mlflow.trace(name="generate_text", span_type=SpanType.LLM)
def generate_traced_sample(
    model: LanguageModel,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Generate a text continuation for a single prompt and record an MLflow LLM trace."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError(f"Prompt {prompt!r} encoded to zero tokens")

    span = mlflow.get_current_active_span()
    if span is not None:
        # Override decorator auto-capture so unserializable model/tokenizer/device args are excluded.
        span.set_inputs({"prompt": prompt, "max_new_tokens": max_new_tokens})
        if model_name:
            span.set_attribute(SpanAttributeKey.MODEL, model_name)
        span.set_attribute(SpanAttributeKey.MODEL_PROVIDER, "custom-pytorch")

    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    completion_ids = output_ids[len(prompt_ids) :]

    completion = tokenizer.decode(completion_ids)
    input_tokens = len(prompt_ids)
    output_tokens = len(completion_ids)
    total_tokens = len(output_ids)

    if span is not None:
        span.set_attribute(
            SpanAttributeKey.CHAT_USAGE,
            {
                TokenUsageKey.INPUT_TOKENS: input_tokens,
                TokenUsageKey.OUTPUT_TOKENS: output_tokens,
                TokenUsageKey.TOTAL_TOKENS: total_tokens,
            },
        )
        span.set_outputs({"completion": completion})

    mlflow.update_current_trace(
        request_preview=prompt,
        response_preview=completion,
        tags={
            "prompt_kind": "unconditional" if prompt == "</s>" else "polish_prompt",
        },
    )

    return {
        "prompt": prompt,
        "completion": completion,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def generate_model_traces(
    model: LanguageModel,
    tokenizer: BPETokenizer,
    device: torch.device,
    prompts: tuple[str, ...] | list[str] = DEFAULT_PROMPTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    model_name: str | None = None,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Generate and flush one MLflow trace per prompt for a trained LanguageModel."""
    model.eval()
    active_model_ctx = (
        mlflow.set_active_model(model_id=model_id) if model_id else nullcontext()
    )
    samples: list[dict[str, Any]] = []
    with active_model_ctx:
        for prompt in prompts:
            samples.append(
                generate_traced_sample(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    model_name=model_name,
                )
            )
    mlflow.flush_trace_async_logging()
    return samples
