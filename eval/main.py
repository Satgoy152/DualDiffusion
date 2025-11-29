
import argparse
import importlib
from functools import partial
from utils import helpers, reporting
from generate import generate

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

    # Initialize results
    all_results = {}

    # Run evaluations for each seed
    for seed in config.get('seeds', [42]):
        helpers.set_seed(seed)
        seed_results = {}

        # --- Run Accuracy Evaluations ---
        if 'datasets' in config:
            print(f"--- Running Accuracy Evaluations for seed {seed} ---")
            for dataset_name, run_func_path in config['datasets'].items():
                module_name, func_name = run_func_path.rsplit('.', 1)
                try:
                    dataset_module = importlib.import_module(f"dataset_suite.{dataset_name}")
                    run_func = getattr(dataset_module, func_name)
                    
                    # Create a generate function with parameters from the config "frozen"
                    model_params = config.get("model_params", {})
                    configured_generate = partial(generate, params=model_params)
                    
                    accuracy = run_func(configured_generate)
                    seed_results[f"{dataset_name}_accuracy"] = accuracy
                except (ImportError, AttributeError) as e:
                    print(f"Could not run {dataset_name}: {e}")
        
        # --- Other evaluations would go here ---
        # For brevity, this main script only runs the accuracy tasks.
        # A complete script would import and run functions from metrics, baselines etc.
        # based on the config file.

        all_results[f"seed_{seed}"] = seed_results

    # --- Aggregate results and generate reports ---
    # This is a simplified aggregation. A real one would be more complex.
    final_summary = {}
    if config.get('datasets'):
        for dataset_name in config['datasets']:
            accuracies = [all_results[f"seed_{s}"][f"{dataset_name}_accuracy"] for s in config['seeds']]
            mean_acc = sum(accuracies) / len(accuracies)
            final_summary[f"mean_{dataset_name}_accuracy"] = f"{mean_acc:.3f}"

    # Generate reports
    reporting.generate_reproducibility_report(config, system_info, final_summary)
    reporting.print_summary_for_readme(config, final_summary)
    
    # Save all raw results
    helpers.save_results(all_results, "full_evaluation_results.json")

    print("\nEvaluation run finished.")

if __name__ == '__main__':
    main()
