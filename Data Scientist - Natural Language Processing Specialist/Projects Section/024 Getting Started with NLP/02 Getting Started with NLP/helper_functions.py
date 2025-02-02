""" This module contains helper functions for the Section: 24 Getting Started with NLP
"""

import re
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def get_part_of_speech(word: str, verbose: bool=False) -> str:
    """This function returns the most likely part of speech (pos) for a given `word` using [wordnet](https://wordnet.princeton.edu/) 
    from `nltk.corpus`. 
    
    In this case we are only interested in the following parts of speech:
        - `n` - NOUN
        - `v` - VERB
        - `a` - ADJECTIVE
        - `r` - ADVERB
    
    Args:
        word (str): The `word` for which the part of speech is to be determined.
        verbose (bool, optional): More informations about founded `synsets`. Defaults to `False`.
    
    Returns:
        str: The most likely part of speech for the given `word`.
    """
    
    probable_part_of_speech = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in probable_part_of_speech if item.pos()=="n"])
    pos_counts["v"] = len([item for item in probable_part_of_speech if item.pos()=="v"])
    pos_counts["a"] = len([item for item in probable_part_of_speech if item.pos()=="a"])
    pos_counts["r"] = len([item for item in probable_part_of_speech if item.pos()=="r"])
    most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
    
    if verbose:
        print(f"Found {len(probable_part_of_speech)} Synsets for the word: {word}")
        print(f"Counts: {pos_counts}", end="\n\n")
        print("List of Synsets:", end="\n")
        for synset in probable_part_of_speech:
            print(f"Name: {synset.name()} | POS: {synset.pos()} | Definition: {synset.definition()}")
    
    return most_likely_part_of_speech



def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words: list) -> str:
    """ Preprocesses the given `text` by cleaning (removing non-characters), tokenizing, removing 
    `stop_words`, and lemmatizing the words in the `text`.

    Args:
        text (str): The `text` to be preprocessed.
        lemmatizer (WordNetLemmatizer): `WordNetLemmatizer` object from `nltk.stem`.
        stop_words (list): A `list` of `stop_words` to be removed from the `text`.

    Returns:
        str: The preprocessed `text`.
    """
    cleaned = re.sub(r'\W+', ' ', text).lower()
    tokenized = word_tokenize(cleaned)
    filtered = list(filter(lambda x: x not in stop_words, tokenized))
    normalized = list(map(lambda x: lemmatizer.lemmatize(x, get_part_of_speech(x)), filtered))
    return " ".join(normalized)




if __name__ == "__main__":
    s = "The quick brown fox jumps over the lazy dog."
    lemmatizer = WordNetLemmatizer()
    stop_words = ["a", "an", "the"]
    print(preprocess_text(s, lemmatizer, stop_words))