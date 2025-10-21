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
