import re
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import math


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        eng_regex = r'\b(a|an|the)\b'
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punct(text):
        return re.sub(r"[^\wàéíòúüçÀÉÍÒÚÜÇ\s]", '', text)

    return white_space_fix(remove_articles(remove_punct(s.lower())))

def f1_score(prediction, ground_truth):
    """
    Compute the token-based F1 score between prediction and ground truth.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Taken from llm-evaluation-harness
def mean(arr):
    return sum(arr) / len(arr)


def calculate_bleu_score(prediction, ground_truth):
    """
    Calculate BLEU score for a prediction against a ground truth.

    Args:
    prediction (str): The predicted text.
    ground_truth (str): The reference text (ground truth).

    Returns:
    float: The BLEU score.
    """
    # Tokenizing the texts into words
    prediction_tokens = word_tokenize(prediction)
    ground_truth_tokens = [word_tokenize(ground_truth)]  # List of lists for multiple references support

    # Calculating BLEU score
    bleu_score = sentence_bleu(ground_truth_tokens, prediction_tokens)
    return bleu_score


def perplexity(items):
    return math.exp(-mean(items))