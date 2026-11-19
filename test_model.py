import argparse

import torch
from torch.nn import functional as F

from model import LanguageModel
from tokenizer import BPETokenizer


@torch.no_grad()
def generate_text(
    model: LanguageModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> torch.Tensor:
    """Autoregressively sample tokens with optional temperature scaling and top-k filtering."""
    if temperature == 1.0 and not top_k:
        return model.generate(idx, max_new_tokens=max_new_tokens)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text from a trained LanguageModel checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/checkpoints/baseline.pt",
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/polish_bpe_8k.json",
        help="Path to BPE tokenizer JSON",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="</s>",
        help='Prompt text (default: "</s>" document delimiter for unconditional generation)',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 for greedy decoding)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff (0 disables top-k filtering)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda:0 if available else cpu)",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    tokenizer = BPETokenizer(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_cfg = dict(ckpt["config"]["model"])
    model_cfg.setdefault("n_groups", model_cfg["n_heads"])
    vocab_size = ckpt.get("vocab_size", tokenizer.vocab_size)

    model = LanguageModel(vocab_size=vocab_size, **model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError(f"Prompt {args.prompt!r} encoded to zero tokens")

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out_ids = generate_text(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )[0].tolist()

    if args.prompt == "</s>":
        print(tokenizer.decode(out_ids[len(prompt_ids) :]))
    else:
        print(tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
