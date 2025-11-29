
import numpy as np

def calculate_accuracy(predictions, references):
    """
    Calculates accuracy.

    Args:
        predictions: A list of predictions.
        references: A list of reference answers.

    Returns:
        The accuracy score.
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)

def get_confidence_interval(data, confidence=0.95):
    """
    Calculates the confidence interval for a given dataset.

    Args:
        data: A list or numpy array of numerical data.
        confidence: The confidence level.

    Returns:
        A tuple of (mean, margin_of_error).
    """
    n = len(data)
    mean = np.mean(data)
    if n < 2:
        return mean, 0
    
    se = np.std(data, ddof=1) / np.sqrt(n)
    
    # Using a t-distribution for small sample sizes
    from scipy.stats import t
    margin_of_error = se * t.ppf((1 + confidence) / 2., n-1)
    
    return mean, margin_of_error

if __name__ == '__main__':
    # Example usage
    preds = ["a", "b", "c", "d"]
    refs = ["a", "b", "c", "c"]
    acc = calculate_accuracy(preds, refs)
    print(f"Accuracy: {acc}")

    # Example for confidence interval
    accuracies = [0.8, 0.85, 0.9, 0.75, 0.88]
    mean, ci = get_confidence_interval(accuracies)
    print(f"Mean Accuracy: {mean:.2f} +/- {ci:.2f}")
