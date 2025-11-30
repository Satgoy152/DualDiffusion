
import argparse
import importlib
from functools import partial
import itertools
from utils import helpers, reporting
from generate import generate

# Import metric functions
from metrics.speed import measure_speed_and_efficiency
from metrics.diversity import calculate_distinct_n, calculate_entropy

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Suite for Diffusion LLMs")
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = helpers.load_config(args.config)
    config['config_file'] = args.config

    # Get system info for reporting
    system_info = reporting.get_system_info()

    # --- Load prompts for metrics from the specified datasets ---
    metric_prompts = []
    if 'datasets' in config:
        print("--- Loading prompts for metrics from specified datasets... ---")
        for dataset_name in config['datasets']:
            try:
                dataset_module = importlib.import_module(f"dataset_suite.{dataset_name}")
                load_func = getattr(dataset_module, f"load_{dataset_name}")
                metric_prompts.extend([item['question'] for item in load_func()])
                print(f"Loaded prompts from {dataset_name}")
            except (ImportError, AttributeError) as e:
                print(f"Could not load prompts from {dataset_name}: {e}")

    if not metric_prompts:
        print("Warning: No prompts loaded for metrics. Skipping speed and diversity.")

    # --- Set up parameter grid ---
    param_runs = []
    if 'baselines' in config:
        for baseline_name, baseline_config in config['baselines'].items():
            param_grid = baseline_config.get('param_grid', {})
            if not param_grid:
                # If no grid, create a single run with default params
                param_runs.append({'run_parameters': {'baseline': baseline_name}})
                continue

            keys, values = zip(*param_grid.items())
            for v_combination in itertools.product(*values):
                run_params = dict(zip(keys, v_combination))
                run_params['baseline'] = baseline_name
                param_runs.append({'run_parameters': run_params})
    
    if not param_runs:
        # If no baselines are defined, do a single run with default model_params
        param_runs.append({'run_parameters': config.get('model_params', {})})

    # --- Run evaluations for each parameter combination ---
    all_results = []
    for i, run_config in enumerate(param_runs):
        run_params = run_config['run_parameters']
        print(f"\n===== Running evaluation for parameter set {i+1}/{len(param_runs)} =====")
        print(f"Parameters: {run_params}")

        run_results = {"run_parameters": run_params}
        
        # Use a single seed for each parameter run for now.
        # A more complex setup could loop over seeds here as well.
        seed = config.get('seeds', [42])[0]
        helpers.set_seed(seed)

        # Create a generate function with parameters for this run "frozen"
        configured_generate = partial(generate, params=run_params)

        # --- Run Accuracy Evaluations ---
        if 'datasets' in config:
            print(f"\n--- Running Accuracy Evaluations ---")
            for dataset_name, run_func_path in config['datasets'].items():
                _, func_name = run_func_path.rsplit('.', 1)
                try:
                    dataset_module = importlib.import_module(f"dataset_suite.{dataset_name}")
                    run_func = getattr(dataset_module, func_name)
                    accuracy = run_func(configured_generate)
                    run_results[f"{dataset_name}_accuracy"] = accuracy
                except (ImportError, AttributeError) as e:
                    print(f"Could not run {dataset_name}: {e}")
        
        # --- Run Other Metric Evaluations ---
        if 'metrics' in config and metric_prompts:
            print(f"\n--- Running Other Metric Evaluations ---")
            
            # Speed & Efficiency
            if 'speed_efficiency' in config['metrics']:
                speed_config = config['metrics']['speed_efficiency']
                speed_results = measure_speed_and_efficiency(configured_generate, metric_prompts, speed_config.get('batch_sizes', [1]))
                run_results['speed_efficiency'] = speed_results

            # Diversity
            if 'diversity' in config['metrics']:
                diversity_config = config['metrics']['diversity']
                num_samples = diversity_config.get("num_samples_to_generate", 20)
                prompts_for_diversity = metric_prompts[:num_samples]
                generated_samples = [configured_generate(p) for p in prompts_for_diversity]
                
                diversity_results = {}
                for n in diversity_config.get('n_values', [2, 3]):
                    diversity_results[f'distinct_{n}'] = calculate_distinct_n(generated_samples, n=n)
                diversity_results['entropy'] = calculate_entropy(generated_samples)
                run_results['diversity'] = diversity_results
        
        all_results.append(run_results)

    # --- Generate reports ---
    reporting.generate_reproducibility_report(config, system_info, {"all_runs": all_results})
    reporting.print_summary_for_readme(config, all_results)
    
    # Save all raw results
    helpers.save_results({"all_runs": all_results}, "full_evaluation_results.json")

    print("\nEvaluation run finished.")

if __name__ == '__main__':
    main()
