import json
import os
import urllib.request

from tokenizer import CharacterTokenizer


class DataLoader:
    def __init__(self):
        self.books = list(
            json.load(open("data/book_urls.json", "r", encoding="utf-8"))
        )[:100]
        self.data_path = "data/"

    def get_size(self):
        size = 0
        for filename in os.listdir(self.data_path):
            size += os.path.getsize(os.path.join(self.data_path, filename))
        if size < 1024:
            return f"{size} bytes"
        elif size < pow(1024, 2):
            return f"{round(size / 1024, 2)} KB"
        elif size < pow(1024, 3):
            return f"{round(size / (pow(1024, 2)), 2)} MB"
        elif size < pow(1024, 4):
            return f"{round(size / (pow(1024, 3)), 2)} GB"

    def download_books(self):
        for i, book in enumerate(self.books):
            try:
                urllib.request.urlretrieve(
                    book, os.path.join(self.data_path, f"book_{i}.txt")
                )
            except Exception as e:
                print(f"Error downloading {book}: {e}")
                continue

    def load_data(self):
        data = []
        for filename in os.listdir(self.data_path):
            with open(
                os.path.join(self.data_path, filename), "r", encoding="utf-8"
            ) as f:
                text = f.read()
                data.append(text)
        return data


if __name__ == "__main__":
    loader = DataLoader()
    loader.download_books()

    print("All books downloaded successfully.")
    print("Total size of all books: ", loader.get_size())

    data = loader.load_data()
    tokenizer = CharacterTokenizer()

    tokens = 0
    for text in data:
        tokens += len(tokenizer.encode(text))

    print("-----------------------")
    print("Token stats:\n")
    print(f"Total training tokens: {tokens}")
    print(f"Vocab size: {tokenizer.vocab_size}")
