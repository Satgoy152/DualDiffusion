
import time
import psutil
import os

def measure_speed_and_efficiency(generate_func, prompts, batch_sizes):
    """
    Measures wall-clock time, memory peak, and throughput.

    Args:
        generate_func: The function to generate text.
        prompts: A list of prompts to use for generation.
        batch_sizes: A list of batch sizes to test.

    Returns:
        A dictionary with the measured metrics.
    """
    results = {}
    for batch_size in batch_sizes:
        print(f"Measuring speed and efficiency for batch size: {batch_size}")
        
        start_time = time.time()
        
        process = psutil.Process(os.getpid())
        initial_mem = process.memory_info().rss
        
        generated_tokens = 0
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # This part needs to be adapted to how your generate_func handles batches
            for prompt in batch_prompts:
                generated_text = generate_func(prompt)
                generated_tokens += len(generated_text.split()) # A simple token count

        end_time = time.time()
        
        final_mem = process.memory_info().rss
        peak_mem = final_mem - initial_mem

        total_time = end_time - start_time
        sequences_per_sec = len(prompts) / total_time
        tokens_per_sec = generated_tokens / total_time
        
        results[batch_size] = {
            "wall_clock_time_per_sequence": total_time / len(prompts),
            "peak_memory_usage_bytes": peak_mem,
            "throughput_sequences_per_sec": sequences_per_sec,
            "throughput_tokens_per_sec": tokens_per_sec,
        }
        
        print(f"  - Wall-clock time per sequence: {results[batch_size]['wall_clock_time_per_sequence']:.4f}s")
        print(f"  - Peak memory usage: {peak_mem / 1e6:.2f} MB")
        print(f"  - Throughput: {tokens_per_sec:.2f} tokens/sec")

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
    # This is a highly simplified and placeholder estimation.
    # A real implementation requires deep knowledge of the model architecture.
    params = model_details.get("parameters", 1e9) # e.g., 1 billion parameters
    nfe = model_details.get("nfe", 50) # Number of function evaluations
    
    # A very rough approximation: 2 * params * nfe per token
    flops_per_token = 2 * params * nfe
    
    print(f"  - Estimated FLOPs per token: {flops_per_token / 1e9:.2f} GFLOPs")
    return flops_per_token


if __name__ == '__main__':
    # Example usage
    def dummy_generate(prompt):
        time.sleep(0.1) # Simulate work
        return "this is a generated sentence"

    test_prompts = ["prompt " + str(i) for i in range(20)]
    batch_sizes_to_test = [1, 8]
    
    measure_speed_and_efficiency(dummy_generate, test_prompts, batch_sizes_to_test)

    model_info = {"parameters": 1.5e9, "nfe": 100}
    estimate_flops(model_info)
