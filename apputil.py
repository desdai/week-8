from collections import defaultdict
import re
import numpy as np

class MarkovText(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.term_dict = None

    def get_term_dict(self):
        """Build a term dictionary of tokens → list of followers (duplicates kept)."""
        tokens = re.findall(r"\b\w+\b", self.corpus)
        term_dict = defaultdict(list)

        for i in range(len(tokens) - 1):
            term_dict[tokens[i]].append(tokens[i + 1])

        if tokens:
            term_dict[tokens[-1]] = term_dict.get(tokens[-1], [])

        self.term_dict = dict(term_dict)
        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        """
        Generate text using the Markov chain property.

        Args:
            seed_term (str, optional): Starting token. If not provided, chosen randomly.
            term_count (int): Number of terms to generate (including the seed).

        Returns:
            str: Generated sequence of tokens.
        """
        # Make sure the dictionary is built
        if self.term_dict is None:
            self.get_term_dict()

        # If user provides a seed, validate it
        if seed_term is not None:
            if seed_term not in self.term_dict:
                raise ValueError(f"'{seed_term}' not found in the corpus.")
            current = seed_term
        else:
            current = np.random.choice(list(self.term_dict.keys()))

        words = [current]

        for _ in range(term_count - 1):
            followers = self.term_dict.get(current, [])
            if not followers:
                break  # stop if no next word
            next_word = np.random.choice(followers)
            words.append(next_word)
            current = next_word

        return " ".join(words)

class MarkovTextK:
    def __init__(self, corpus, k=2):
        """
        corpus : str
            The input text used to train the model.
        k : int
            State window size (number of consecutive words in each state).
        """
        self.corpus = corpus
        self.k = k
        self.term_dict = None

    def get_term_dict(self):
        """
        Build a Markov dictionary with tuple keys of length k.

        Example (k=2):
        {("Healing", "comes"): ["from", ...], ("comes", "from"): ["taking", ...]}
        """
        tokens = re.findall(r"\b\w+\b", self.corpus)
        term_dict = defaultdict(list)
        k = self.k

        for i in range(len(tokens) - k):
            key = tuple(tokens[i:i + k])
            next_word = tokens[i + k]
            term_dict[key].append(next_word)

        # ensure the last tuple appears
        if len(tokens) >= k:
            last_key = tuple(tokens[-k:])
            term_dict[last_key] = term_dict.get(last_key, [])

        self.term_dict = dict(term_dict)
        return self.term_dict

    def generate(self, seed_term=None, term_count=20):
        """
        Generate a text sequence using a Markov chain of order k.
        """
        if self.term_dict is None:
            self.get_term_dict()

        keys = list(self.term_dict.keys())

        # process seed
        if seed_term is not None:
            if isinstance(seed_term, str):
                seed_term = tuple(seed_term.split())
            if seed_term not in self.term_dict:
                raise ValueError(f"Seed term {seed_term} not found in corpus.")
            current_state = seed_term
        else:
            current_state = keys[np.random.randint(len(keys))]

        generated = list(current_state)

        for _ in range(term_count - self.k):
            followers = self.term_dict.get(current_state, [])
            if not followers:
                break
            next_word = np.random.choice(followers)
            generated.append(next_word)
            current_state = tuple(generated[-self.k:])

        return " ".join(generated)
