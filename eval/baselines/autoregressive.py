
def run_autoregressive_baseline(prompt, top_k=40, top_p=0.9, temperature=1.0):
    """
    Runs a strong autoregressive (AR) baseline for comparison.

    Args:
        prompt: The input prompt.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        temperature: Sampling temperature.

    Returns:
        The generated text from the AR baseline model.
    """
    print(f"Running AR baseline with top_k={top_k}, top_p={top_p}, temp={temperature}...")
    
    # This is a placeholder for the actual implementation.
    # You would typically use a library like `transformers` to load a
    # pre-trained AR model (e.g., GPT-2, Llama) and generate text.
    
    # Simulate different outputs based on parameters
    if temperature > 1.2:
        return "The baseline AR model produced a very creative but possibly incoherent response."
    elif top_p < 0.8:
        return "The baseline AR model produced a safe and predictable response."
    else:
        return f"A standard response from the baseline AR model for prompt: {prompt}"

def sweep_ar_hyperparameters(prompt, param_grid):
    """
    Sweeps through a grid of hyperparameters for the AR baseline.

    Args:
        prompt: The input prompt.
        param_grid: A dictionary with lists of hyperparameter values to test.
                    Example: {'top_k': [40, 50], 'top_p': [0.9, 0.95]}

    Returns:
        A list of dictionaries, each containing the params and generated output.
    """
    print("\nSweeping AR hyperparameters...")
    results = []
    
    # A simple grid search
    from itertools import product
    keys, values = zip(*param_grid.items())
    
    for v in product(*values):
        params = dict(zip(keys, v))
        output = run_autoregressive_baseline(prompt, **params)
        result_item = {"params": params, "output": output}
        results.append(result_item)
        print(f"  - Params: {params}, Output: '{output[:30]}...'")
        
    return results

if __name__ == '__main__':
    # Example usage
    test_prompt = "The best way to learn programming is"
    
    # Single run
    run_autoregressive_baseline(test_prompt)
    
    # Hyperparameter sweep
    grid = {
        'top_k': [40, 100],
        'top_p': [0.9, 0.95],
        'temperature': [0.8, 1.0, 1.2]
    }
    sweep_ar_hyperparameters(test_prompt, grid)
