import os


class CharacterTokenizer:
    def __init__(self):
        DATA_PATH = "data/"
        # Read dataset
        self.text = ""
        for filename in os.listdir(DATA_PATH):
            with open(os.path.join(DATA_PATH, filename), "r", encoding="utf-8") as f:
                text = f.read()
                self.text += text

        self.tokens = sorted(list(set(self.text)))
        self.vocab_size = len(self.tokens)

        # Build encoder and decoder (tokenization)
        self.character_to_integer_map = {char: i for i, char in enumerate(self.tokens)}
        self.integer_to_character_map = {i: char for i, char in enumerate(self.tokens)}

    def encode(self, s):
        return [self.character_to_integer_map[char] for char in s]

    def decode(self, ls):
        return "".join([self.integer_to_character_map[i] for i in ls])


class WordTokenizer:
    def __init__(self):
        DATA_PATH = "data/input.txt"
        # Read dataset
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        words = [i.strip() for i in text.split()]
        self.tokens = sorted(list(set(words)))
        self.vocab_size = len(words)
        self.text = text

        # Build encoder and decoder (tokenization)
        self.character_to_integer_map = {char: i for i, char in enumerate(self.tokens)}
        self.integer_to_character_map = {i: char for i, char in enumerate(self.tokens)}

    def encode(self, s):
        return [self.character_to_integer_map[char] for char in s.split()]

    def decode(self, ls):
        return "".join([self.integer_to_character_map[i] for i in ls.split()])
