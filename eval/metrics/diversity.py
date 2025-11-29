
from collections import Counter
import numpy as np

def calculate_distinct_n(generated_samples, n=2):
    """
    Calculates distinct-n score.

    Args:
        generated_samples: A list of generated text samples.
        n: The "n" in distinct-n.

    Returns:
        The distinct-n score.
    """
    if not generated_samples:
        return 0.0
        
    total_ngrams = 0
    distinct_ngrams = set()
    
    for sample in generated_samples:
        tokens = sample.split()
        if len(tokens) >= n:
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                distinct_ngrams.add(ngram)
                total_ngrams += 1
    
    if total_ngrams == 0:
        return 0.0
        
    return len(distinct_ngrams) / total_ngrams

def calculate_self_bleu(generated_samples):
    """
    Calculates self-BLEU score. This is a simplified version.
    A full implementation would require a standard BLEU scoring library.

    Args:
        generated_samples: A list of generated text samples.

    Returns:
        The self-BLEU score.
    """
    print("Calculating self-BLEU (simplified)...")
    # This is a placeholder. A real implementation should use a library like sacrebleu or nltk.
    # The logic involves comparing each sample against all others as references.
    # For simplicity, we return a dummy value.
    
    if len(generated_samples) < 2:
        return 0.0

    # A very rough approximation of diversity
    from random import sample as random_sample
    subset_for_demo = random_sample(generated_samples, min(10, len(generated_samples)))
    
    # Very basic: check for pairwise exact matches in a small subset
    matches = 0
    for i in range(len(subset_for_demo)):
        for j in range(i + 1, len(subset_for_demo)):
            if subset_for_demo[i] == subset_for_demo[j]:
                matches +=1
                
    # Lower score is more diverse
    return matches / len(subset_for_demo)


def calculate_entropy(generated_samples):
    """
    Calculates the entropy of the token distribution.

    Args:
        generated_samples: A list of generated text samples.

    Returns:
        The entropy score.
    """
    all_tokens = []
    for sample in generated_samples:
        all_tokens.extend(sample.split())
        
    if not all_tokens:
        return 0.0

    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    probs = [count / total_tokens for count in token_counts.values()]
    
    entropy = -np.sum(p * np.log2(p) for p in probs if p > 0)
    return entropy

if __name__ == '__main__':
    # Example usage
    samples = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat sat on the mat", # Repetition
        "a bird is in the sky"
    ]

    dist2 = calculate_distinct_n(samples, n=2)
    print(f"Distinct-2 score: {dist2:.3f}")
    
    s_bleu = calculate_self_bleu(samples)
    print(f"Self-BLEU (simplified): {s_bleu:.3f}")

    entropy_val = calculate_entropy(samples)
    print(f"Token Entropy: {entropy_val:.3f}")
