import torch

from model import LanguageModel

ckpt = torch.load(
    "runs/checkpoints/baseline.pt", map_location="cpu", weights_only=False
)

model = LanguageModel(
    vocab_size=len(ckpt["vocab"]),  # 94, don't use tokenizer.vocab_size directly
    **ckpt["config"]["model"],  # n_embed, block_size, n_blocks, n_heads, dropout
)
model.load_state_dict(ckpt["model_state"])
model.eval()  # important: disables dropout in model.py

# generate
idx = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(idx, max_new_tokens=200)[0].tolist()
itos = {i: ch for i, ch in enumerate(ckpt["vocab"])}
print("".join(itos[i] for i in out))
