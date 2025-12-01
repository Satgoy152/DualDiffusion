# metrics/speed.py
import torch

def measure_speed_and_efficiency(total_tokens, total_elapsed_time, total_gpu_bytes, batch_sizes=[1]):
    """
    Computes throughput and wall-clock metrics based on precomputed totals.

    Args:
        total_tokens: Total number of tokens generated during evaluation.
        total_elapsed_time: Total wall-clock time spent generating.
        total_gpu_bytes: Peak GPU memory used during generation.
        batch_sizes: List of batch sizes for reporting (for compatibility).

    Returns:
        A dictionary mapping batch_size -> metrics dictionary.
    """
    results = {}
    if total_elapsed_time == 0:
        total_elapsed_time = 1e-6  # avoid division by zero

    for batch_size in batch_sizes:
        sequences_per_sec = None  # we don't know total sequences without prompt list
        tokens_per_sec = total_tokens / total_elapsed_time

        results[batch_size] = {
            "wall_clock_time_total": total_elapsed_time,
            "peak_gpu_memory_bytes": total_gpu_bytes,
            "throughput_tokens_per_sec": tokens_per_sec,
            "total_tokens_generated": total_tokens
        }

        print(f"--- Speed Metrics (batch size {batch_size}) ---")
        print(f"Total elapsed time: {total_elapsed_time:.4f}s")
        print(f"Peak GPU memory: {total_gpu_bytes / 1e6:.2f} MB")
        print(f"Tokens per second: {tokens_per_sec:.2f}")
        print(f"Total tokens generated: {total_tokens}")

    return results


def estimate_flops(model_details):
    """
    Estimates FLOPs for sampling. This is highly model-specific.

    Args:
        model_details: A dictionary with model-specific information.

    Returns:
        Estimated FLOPs.
    """
    print("Estimating FLOPs...")
    params = model_details.get("parameters", 1e9)  # e.g., 1 billion parameters
    nfe = model_details.get("nfe", 50)            # Number of function evaluations
    
    # Very rough approximation: 2 * params * nfe per token
    flops_per_token = 2 * params * nfe
    print(f"  - Estimated FLOPs per token: {flops_per_token / 1e9:.2f} GFLOPs")
    return flops_per_token


if __name__ == "__main__":
    # Example usage with precomputed totals
    total_tokens = 5000
    total_elapsed_time = 12.5
    total_gpu_bytes = 3 * 1024**3  # 3GB

    batch_sizes_to_test = [1, 8]

    results = measure_speed_and_efficiency(
        total_tokens=total_tokens,
        total_elapsed_time=total_elapsed_time,
        total_gpu_bytes=total_gpu_bytes,
        batch_sizes=batch_sizes_to_test
    )
    print("\nResults:")
    print(results)

    model_info = {"parameters": 1.5e9, "nfe": 100}
    estimate_flops(model_info)
