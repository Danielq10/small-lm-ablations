import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)  # (B, T, 16)
        q = self.query(x)  # (B, T, 16)

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embed, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    head_size=head_size,
                    n_embed=n_embed,
                    block_size=block_size,
                    dropout=dropout,
                )
                for head in range(num_heads)
            ]
        )
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projection(out)
        return out


class FeedForwardNetwork(nn.Module):
    """Simple MLP feedforwad network"""

    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class DecoderBlock(nn.Module):
    """Transformers decoder block"""

    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.heads = MultiHeadAttention(
            num_heads=n_heads,
            head_size=head_size,
            n_embed=n_embed,
            block_size=block_size,
            dropout=dropout,
        )
        self.mlp = FeedForwardNetwork(n_embed=n_embed, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self, vocab_size, n_embed=64, block_size=8, n_blocks=6, n_heads=8, dropout=0.2
    ):
        super().__init__()

        self.block_size = block_size

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    n_embed=n_embed,
                    n_heads=n_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for n in range(n_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indixes in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]

            # get the prediction logits
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append smaple index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
