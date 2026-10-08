import json
import os
import re
import unicodedata
import urllib.request

import tqdm


class DataLoader:
    def __init__(self, num_books=None):
        self.books = (
            list(json.load(open("data/book_urls.json", "r", encoding="utf-8")))[
                :num_books
            ]
            if num_books
            else list(json.load(open("data/book_urls.json", "r", encoding="utf-8")))
        )
        self.data_path = "data/"
        self.cleaner = BookTextCleaner(ascii_quotes=False)
        self.corpus = []

    def get_size(self, file_path):
        size = 0
        if file_path:
            size = os.path.getsize(file_path)
        else:
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

    def get_book_text(self, book_url):
        try:
            response = urllib.request.urlopen(book_url)
            book = response.read().decode("utf-8")
            book = self.cleaner.clean(book)
            return book
        except Exception:
            pass

    def create_corpus(self):
        print(f"Creating corpus from {len(self.books)} books...")
        for i, book in tqdm.tqdm(enumerate(self.books), desc="Creating corpus"):
            book_text = self.get_book_text(book)
            if book_text:
                self.corpus.append(book_text)

            else:
                print(f"Skipping book {i} due to download/cleaning error.")

    def save_corpus(self, output_file="data/corpus.txt"):
        with open(output_file, "w", encoding="utf-8") as f:
            for text in self.corpus:
                f.write(text + "\n\n")

        print(f"Corpus saved to {output_file}")

    def load_data(self):
        data = []
        for filename in os.listdir(self.data_path):
            with open(
                os.path.join(self.data_path, filename), "r", encoding="utf-8"
            ) as f:
                text = f.read()
                data.append(text)
        return data


class BookTextCleaner:
    def __init__(self, ascii_quotes: bool = False):
        """
        :param ascii_quotes: If True, normalizes all quotation marks to ASCII '"' and "'".
                             If False, retains typographic quotes but cleans inconsistencies.
        """
        self.ascii_quotes = ascii_quotes
        self.polish_caps = "A-ZĄĆĘŁŃÓŚŹŻ"

        # 1. Structure & Whitespace
        self.crlf_re = re.compile(r"\r\n|\r")
        self.dehyphen_re = re.compile(r"(\b\w+)-\n(\w+\b)")
        self.spaces_re = re.compile(
            r"[ \t\u00A0\u200B]+"
        )  # Includes non-breaking & zero-width spaces
        self.newlines_re = re.compile(r"\n{3,}")

        # 2. ISBN & Boilerplate lines
        self.isbn_re = re.compile(
            r"(?im)^[^\n]*?\bISBN(?:-1[03])?:?\s*(?:(?:97[89][-\s]?)?[0-9]{1,5}[-\s]?[0-9]+[-\s]?[0-9]+[-\s]?[0-9X])[^\n]*?\n?"
        )

        # 3. Dashes & Ellipses
        self.em_dash_normalize_re = re.compile(r"\s*--+\s*")
        self.dots_to_ellipsis_re = re.compile(r"\.{3,}")

        # 4. Glued punctuation & dialogue lineation
        self.dialogue_break_re = re.compile(r"([.:!?])\s*(—\s*)")
        self.dialogue_start_re = re.compile(rf"([^\n])(—\s*[{self.polish_caps}])")
        self.glued_sentence_re = re.compile(rf"([.!?])(?=[{self.polish_caps}])")

        # 5. Quote normalizations
        self.double_quotes_re = re.compile(r"[«»“„”″‟]")
        self.single_quotes_re = re.compile(r"[’‘`´ʻʼ]")

    def normalize_quotes_and_symbols(self, text: str) -> str:
        # Standardize ellipses
        text = self.dots_to_ellipsis_re.sub("…", text)

        # Normalize ASCII double-hyphens to standard em-dash
        text = self.em_dash_normalize_re.sub(" — ", text)

        if self.ascii_quotes:
            text = self.double_quotes_re.sub('"', text)
            text = self.single_quotes_re.sub("'", text)
        else:
            # Normalize single quotes/apostrophes
            text = self.single_quotes_re.sub("’", text)

            # Opening quotes: replace quote preceded by start-of-line or whitespace and followed by word char
            text = re.sub(r'(^|\s)["“«](?=\w)', r"\1„", text, flags=re.MULTILINE)

            # Closing quotes: replace quote preceded by word char
            text = re.sub(r'(?<=\w)["»”]', "”", text)

        return text

    def clean(self, raw_text: str) -> str:
        # Unicode normalization (NFC ensures single codepoints for Polish accented letters)
        text = unicodedata.normalize("NFC", raw_text)

        # Convert line breaks to \n
        text = self.crlf_re.sub("\n", text)

        # Strip entire lines containing ISBNs
        text = self.isbn_re.sub("", text)

        # Normalize quotes, apostrophes, dashes, and ellipses
        text = self.normalize_quotes_and_symbols(text)

        # Re-join words hyphenated at line wraps (e.g. "podmor-\nskiej" -> "podmorskiej")
        text = self.dehyphen_re.sub(r"\1\2", text)

        # Separate dialogues glued to prior sentences (e.g., ":— Boże" -> ":\n\n— Boże")
        text = self.dialogue_break_re.sub(r"\1\n\n\2", text)
        text = self.dialogue_start_re.sub(r"\1\n\n\2", text)

        # Fix glued sentences missing spaces (e.g. "serca.Przybyłem" -> "serca. Przybyłem")
        text = self.glued_sentence_re.sub(r"\1 ", text)

        # Clean spaces and tabs within individual lines
        text = "\n".join(
            self.spaces_re.sub(" ", line).strip() for line in text.split("\n")
        )

        # Collapse excess empty lines to double newlines (standard paragraph break)
        text = self.newlines_re.sub("\n\n", text)

        return text.strip()


if __name__ == "__main__":
    # loader = DataLoader(num_books=100)
    # loader.download_books()

    # print("All books downloaded successfully.")
    # print("Total size of all books: ", loader.get_size())

    # data = loader.load_data()
    # tokenizer = CharacterTokenizer()

    # tokens = 0
    # for text in data:
    #     tokens += len(tokenizer.encode(text))

    # print("-----------------------")
    # print("Token stats:\n")
    # print(f"Total training tokens: {tokens}")
    # print(f"Vocab size: {tokenizer.vocab_size}")

    loader = DataLoader(num_books=100)
    loader.create_corpus()
    loader.save_corpus(output_file="data/corpus.txt")

    print("Corpus created and saved.")
    print("Total size of corpus: ", loader.get_size(file_path="data/corpus.txt"))
