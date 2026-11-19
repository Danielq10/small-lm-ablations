from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

OUTPUT_DIR = Path("data/polish_dynaword_clean")
SHUFFLE_SEED = 42


def main():
    print("Loading Polish Dynaword dataset from Hugging Face...")
    ds = load_dataset("SlayerLab/polish-dynaword-mix-extended-500M")

    # 1. Deterministically shuffle all splits at the document level so long books
    #    and short fragments are mixed uniformly before tokenizing into .bin files.
    print(f"Shuffling dataset splits with seed={SHUFFLE_SEED}...")
    ds = ds.shuffle(seed=SHUFFLE_SEED)

    # 2. Save shuffled Arrow tables to disk (save_to_disk flattens the shuffle
    #    indices so subsequent reads are fast, contiguous sequential scans).
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved shuffled Arrow dataset to {OUTPUT_DIR.resolve()}")

    dataset = load_from_disk(str(OUTPUT_DIR))
    batch_size = 20000

    # 3. Export shuffled train text to flat file for training the tokenizer
    raw_txt_path = Path("data/train_for_tokenizer.txt")
    train_ds = dataset["train"]
    total_train_docs = len(train_ds)
    print(
        f"Exporting raw text for BPE training ({total_train_docs:,d} docs) to {raw_txt_path}..."
    )
    with open(raw_txt_path, "w", encoding="utf-8") as f, tqdm(
        total=total_train_docs, desc="Exporting train text", unit="docs"
    ) as pbar:
        for i in range(0, total_train_docs, batch_size):
            batch_texts = train_ds[i : i + batch_size]["text"]
            if batch_texts:
                f.write("\n\n".join(batch_texts) + "\n\n")
            pbar.update(len(batch_texts))
    print("Done.")

    # 4. Initialize Byte-Level BPE (handles arbitrary UTF-8 / Polish diacritics cleanly)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    # 5. Keep vocab compact so embedding parameters don't dominate the model
    trainer = BpeTrainer(
        vocab_size=8192,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=ByteLevel.alphabet(),
    )

    # 6. Train tokenizer directly on the Polish text file
    tokenizer.train(files=[str(raw_txt_path)], trainer=trainer)
    tokenizer.save("data/polish_bpe_8k.json")

    eos_id = tokenizer.token_to_id("</s>")

    # 7. Pretokenize shuffled data in batches directly to disk to prevent OOM
    for split, filename in [
        ("train", "data/train.bin"),
        ("validation", "data/val.bin"),
    ]:
        dest_path = Path(filename)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        split_ds = dataset[split]
        total_docs = len(split_ds)
        print(f"Tokenizing {split} ({total_docs:,d} docs) in batches of {batch_size}...")

        total_tokens = 0
        with open(dest_path, "wb") as f, tqdm(
            total=total_docs, desc=f"Tokenizing {split}", unit="docs"
        ) as pbar:
            for i in range(0, total_docs, batch_size):
                batch_texts = split_ds[i : i + batch_size]["text"]
                encodings = tokenizer.encode_batch(batch_texts)

                tokens = []
                for enc in encodings:
                    tokens.extend(enc.ids)
                    tokens.append(eos_id)

                arr = np.array(tokens, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(arr)
                pbar.update(len(batch_texts))

        file_size_mb = dest_path.stat().st_size / 1e6
        print(f"Saved {filename} ({total_tokens:,d} tokens, {file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
