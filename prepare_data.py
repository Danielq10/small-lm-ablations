from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from dataloader import DataLoader

OUTPUT_DIR = Path("data/wolne_lektury")


def main():
    print("Downloading books from Wolne Lektury ...")
    loader = DataLoader()
    loader.create_corpus()

    ds = Dataset.from_dict({"text": loader.corpus})
    # 1. Deterministic 95/5 split at the document level
    split = ds.train_test_split(test_size=0.05, seed=42, shuffle=True)
    train_ds = split["train"]

    # 4. Save clean Arrow tables to disk
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved text-only Arrow dataset to {OUTPUT_DIR.resolve()}")

    # 5. Export train text to flat file for training the tokenizer
    raw_txt_path = Path("data/train_for_tokenizer.txt")
    print(f"Exporting raw text for BPE training to {raw_txt_path}...")
    with open(raw_txt_path, "w", encoding="utf-8") as f:
        for doc in train_ds:
            f.write(doc["text"] + "\n\n")
    print("Done.")

    # 6. Initialize Byte-Level BPE (handles arbitrary UTF-8 / Polish diacritics cleanly)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    # 7. Keep vocab compact so embedding parameters don't dominate the model
    trainer = BpeTrainer(
        vocab_size=8192,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=ByteLevel.alphabet(),
    )

    # 8. Train tokenizer directly on the Polish text file
    tokenizer.train(files=["data/train_for_tokenizer.txt"], trainer=trainer)
    tokenizer.save("data/polish_bpe_8k.json")

    dataset = load_from_disk(OUTPUT_DIR)

    eos_id = tokenizer.token_to_id("</s>")

    # pretokenize data
    for split, filename in [
        ("train", "data/train.bin"),
        ("test", "data/val.bin"),
    ]:
        print(f"Tokenizing {split}...")
        texts = dataset[split]["text"]
        # Fast parallel tokenization via tokenizers Rust backend
        encodings = tokenizer.encode_batch(texts)

        # Flatten token IDs and delimit documents with </s>
        tokens = []
        for enc in encodings:
            tokens.extend(enc.ids)
            tokens.append(eos_id)

        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(filename)
        print(f"Saved {filename} ({len(arr):,d} tokens, {arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
