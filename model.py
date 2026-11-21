import torch
import torch.nn as nn
from torch.nn import functional as F


def precompute_angles(
    sequence_length,
    head_size,
):
    freqs = torch.arange(
        0, sequence_length
    ).float()  # Initialize freqs table with i (0, 1, 2, ..., T-1)

    channel_index = torch.arange(start=0, end=head_size, step=2).float()

    # For each i,, theta_i = 10000 ** (-2i/d_model)
    thetas = 1.0 / (10000 ** (channel_index / head_size))

    # Now, create a matrix of rotation angles for every token posisition [0, T-1] and every channel pair [0, C/2 - 1]
    angles = torch.outer(freqs, thetas)  # (T, C/2)
    return angles


def rotate_vector(vector, cos_theta, sin_theta):
    T = vector.shape[-2]

    x0 = vector[..., 0::2]
    x1 = vector[..., 1::2]

    # 2. Swap and negate: (x0, x1) -> (-x1, x0)
    vector_rotated = torch.stack((-x1, x0), dim=-1).flatten(-2)

    result = (vector * cos_theta[:T]) + (vector_rotated * sin_theta[:T])

    return result


def apply_rotary_pos_emb(vector):
    # Rotates the vector q or k

    # vector is of shape (B, T, C)
    # B = batch size, T = sequence length, C = embedding dimension

    # If we treat the pair $(x_0, x_1)$ as a single complex number $z = x_0 + i x_1$, Euler's formula states:

    # $$e^{i\theta} = \cos\theta + i\sin\theta$$

    sequence_length = vector.shape[-2]  # Gives back T
    device = vector.device

    freqs = torch.arange(
        0, sequence_length, device=device
    ).float()  # Initialize freqs table with i (0, 1, 2, ..., T-1)

    channel_index = torch.arange(
        start=0, end=vector.shape[-1], step=2, device=device
    ).float()

    # For each i,, theta_i = 10000 ** (-2i/d_model)
    thetas = 1.0 / (10000 ** (channel_index / vector.shape[-1]))

    # Now, create a matrix of rotation angles for every token posisition [0, T-1] and every channel pair [0, C/2 - 1]
    angles = torch.outer(freqs, thetas)  # (T, C/2)

    # Why the shapes don't match yet (C/2 vs C)

    #   • vector has C channels: (x₀, x₁, x₂, x₃, …, x_{C-1}).
    #   • angles only has C/2 values: (θ₀, θ₁, …, θ_{C/2-1}).

    #   Why? Because every 2 channels form a 2D plane that shares 1 angle:

    #   • (x₀, x₁) both rotate by θ₀
    #   • (x₂, x₃) both rotate by θ₁

    # Duplicate each angle so it covers both channels in the pair:
    cos_theta = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)  # (T, C)
    sin_theta = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)  # (T, C)

    # 1. Grab even channels (x0, x2, ...) and odd channels (x1, x3, ...)
    x0 = vector[..., 0::2]
    x1 = vector[..., 1::2]

    # 2. Swap and negate: (x0, x1) -> (-x1, x0)
    vector_rotated = torch.stack((-x1, x0), dim=-1).flatten(-2)

    result = (vector * cos_theta) + (vector_rotated * sin_theta)

    return result


class AttentionHead(nn.Module):
    def __init__(
        self,
        head_size,
        n_embed,
        block_size,
        pos_encoding,
        dropout=0.2,
    ):
        super().__init__()
        self.pos_encoding = pos_encoding
        self.query = nn.Linear(n_embed, head_size, bias=False)
        # TODO: tril can also be registered once in the GQA
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k, v, cos_theta, sin_theta):
        B, T, C = x.shape

        q = self.query(x)  # (B, T, 16)

        if self.pos_encoding == "rope":
            q = rotate_vector(q, cos_theta, sin_theta)

        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted aggregation of the values
        out = wei @ v
        return out


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_size,
        n_embed,
        block_size,
        num_groups,
        dropout=0.2,
        pos_encoding=None,
    ):
        super().__init__()
        assert num_heads % num_groups == 0, (
            "num_heads must be divisible by num_kv_groups"
        )

        assert pos_encoding in ["learned", "rope"], (
            f"pos_encoding {pos_encoding} not supported, please choose one of ['learned', 'rope']"
        )

        self.keys = nn.ModuleList(
            [nn.Linear(n_embed, head_size, bias=False) for _ in range(num_groups)]
        )
        self.values = nn.ModuleList(
            [nn.Linear(n_embed, head_size, bias=False) for _ in range(num_groups)]
        )

        self.heads_per_group = num_heads // num_groups
        self.num_groups = num_groups
        self.pos_encoding = pos_encoding

        if pos_encoding == "rope":
            assert head_size % 2 == 0, "head_size must be even for RoPE"
            angles = precompute_angles(block_size, head_size)
            self.register_buffer(
                "cos_theta", torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
            )

            self.register_buffer(
                "sin_theta", torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
            )

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size=head_size,
                    n_embed=n_embed,
                    block_size=block_size,
                    dropout=dropout,
                    pos_encoding=pos_encoding,
                )
                for head in range(num_heads)
            ]
        )
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        precomputed_keys, precomputed_values = (
            [
                rotate_vector(k(x), self.cos_theta, self.sin_theta)
                if self.pos_encoding == "rope"
                else k(x)
                for k in self.keys
            ],
            [v(x) for v in self.values],
        )

        head_results = []
        for i, head in enumerate(self.heads):
            group_index = i // self.heads_per_group
            head_results.append(
                head(
                    x,
                    precomputed_keys[group_index],
                    precomputed_values[group_index],
                    self.cos_theta if self.pos_encoding == "rope" else None,
                    self.sin_theta if self.pos_encoding == "rope" else None,
                )
            )

        out = torch.cat(head_results, dim=-1)
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForwardNetwork(nn.Module):
    """Simple MLP feedforwad network"""

    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed Forward Network"""

    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        # 8/3 * n_embed to preserve parameter parity with a 4*n_embed standard MLP,
        # rounded to the nearest multiple of 64 for GPU tensor core efficiency.
        raw_hidden_dim = 2 * (4 * n_embed) / 3
        hidden_dim = max(64, 64 * round(raw_hidden_dim / 64))

        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)  # Gate
        self.w3 = nn.Linear(n_embed, hidden_dim, bias=False)  # Up
        self.w2 = nn.Linear(hidden_dim, n_embed, bias=False)  # Down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: (SiLU(W1 x) * W3 x) W2
        silu = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(silu))


class DecoderBlock(nn.Module):
    """Transformers decoder block"""

    def __init__(
        self, n_embed, n_heads, block_size, dropout, pos_encoding, n_groups, ffn
    ):
        super().__init__()
        head_size = n_embed // n_heads

        self.heads = GroupedQueryAttention(
            num_heads=n_heads,
            head_size=head_size,
            n_embed=n_embed,
            block_size=block_size,
            num_groups=n_groups,
            dropout=dropout,
            pos_encoding=pos_encoding,
        )

        if ffn == "mlp":
            self.ffn = FeedForwardNetwork(n_embed=n_embed, dropout=dropout)
        elif ffn == "swiglu":
            self.ffn = SwiGLUFFN(n_embed=n_embed, dropout=dropout)
        else:
            raise ValueError(
                f"Wrong ffn type: {ffn}, supported values are ['mlp', 'swiglu']"
            )
        self.ln1 = nn.RMSNorm(n_embed)
        self.ln2 = nn.RMSNorm(n_embed)

    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embed=64,
        block_size=8,
        n_blocks=6,
        n_heads=8,
        n_groups=8,
        ffn="mlp",
        dropout=0.2,
        pos_encoding="learned",
    ):
        super().__init__()

        self.block_size = block_size
        self.pos_encoding = pos_encoding

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        if pos_encoding == "learned":
            self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    n_embed=n_embed,
                    n_heads=n_heads,
                    block_size=block_size,
                    dropout=dropout,
                    pos_encoding=pos_encoding,
                    n_groups=n_groups,
                    ffn=ffn,
                )
                for n in range(n_blocks)
            ]
        )
        self.ln_f = nn.RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        if self.pos_encoding == "learned":
            pos_emb = self.position_embedding_table(
                torch.arange(T, device=idx.device)
            )  # (T, C)
            x = tok_emb + pos_emb  # (B, T, C)
        else:
            x = tok_emb

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

    @torch.no_grad()
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
