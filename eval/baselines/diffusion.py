
def run_diffusion_baseline(prompt, sampler='DDIM', steps=50):
    """
    Runs a standard diffusion model baseline (e.g., DDIM, DDPM).

    Args:
        prompt: The input prompt.
        sampler: The type of sampler to use ('DDIM', 'DDPM', etc.).
        steps: The number of denoising steps.

    Returns:
        The generated text from the diffusion baseline.
    """
    print(f"Running Diffusion baseline with sampler={sampler}, steps={steps}...")

    # This is a placeholder. The actual implementation would depend on the
    # specific diffusion model library you are using. It should invoke the
    # vanilla sampler for your model.

    if steps < 20:
        return f"A low-quality response due to few steps ({steps}) from the {sampler} sampler."
    else:
        return f"A standard-quality response from the {sampler} sampler with {steps} steps."

def sweep_diffusion_hyperparameters(prompt, param_grid):
    """
    Sweeps through a grid of hyperparameters for the diffusion baseline.

    Args:
        prompt: The input prompt.
        param_grid: A dictionary with lists of hyperparameter values to test.
                    Example: {'sampler': ['DDIM', 'DDPM'], 'steps': [20, 50]}

    Returns:
        A list of dictionaries, each containing the params and generated output.
    """
    print("\nSweeping Diffusion hyperparameters...")
    results = []

    from itertools import product
    keys, values = zip(*param_grid.items())

    for v in product(*values):
        params = dict(zip(keys, v))
        output = run_diffusion_baseline(prompt, **params)
        result_item = {"params": params, "output": output}
        results.append(result_item)
        print(f"  - Params: {params}, Output: '{output[:40]}...'")

    return results

if __name__ == '__main__':
    # Example usage
    test_prompt = "A recipe for a delicious chocolate cake"

    # Single run
    run_diffusion_baseline(test_prompt)

    # Hyperparameter sweep
    grid = {
        'sampler': ['DDIM', 'DDPM'],
        'steps': [10, 25, 50, 100]
    }
    sweep_diffusion_hyperparameters(test_prompt, grid)
