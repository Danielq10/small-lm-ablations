from pathlib import Path

from tokenizers import Tokenizer


class BPETokenizer:
    def __init__(self, tokenizer_path: str = "data/tokenizer.json"):
        path = Path(tokenizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer artifact not found at {path}")
        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)
