import argparse
import importlib
from functools import partial
import itertools
from utils import helpers, reporting
from metrics.diversity import calculate_distinct_n, calculate_entropy
from metrics.speed import measure_speed_and_efficiency


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Suite for Diffusion LLMs")
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    config = helpers.load_config(args.config)
    print(f"Loaded configuration from {args.config}")
    config['config_file'] = args.config

    # Get system info for reporting
    system_info = reporting.get_system_info()

    # ---------------------------------------------------------
    # Load Model Dynamically (fastDLLM, slowDLLM, baseline, etc.)
    # ---------------------------------------------------------
    model_name = config.get('model', 'fastDLLM')

    try:
        model_module = importlib.import_module(f"models.{model_name}")
        generate = getattr(model_module, "generate")
        get_model_instance = getattr(model_module, "get_model_instance")
        model_instance = get_model_instance()
        print(f"--- Loaded model: {model_name} ---\n")

    except (ImportError, AttributeError) as e:
        print(f"Could not load model {model_name}: {e}")
        return

    # ---------------------------------------------------------
    # Load prompts from datasets
    # ---------------------------------------------------------
    metric_prompts = []
    if 'datasets' in config:
        print("--- Loading prompts from specified datasets ---")
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

    # ---------------------------------------------------------
    # Parameter grid setup
    # ---------------------------------------------------------
    param_runs = []
    if 'baselines' in config:
        for baseline_name, baseline_config in config['baselines'].items():
            param_grid = baseline_config.get('param_grid', {})
            if not param_grid:
                param_runs.append({'run_parameters': {'baseline': baseline_name}})
                continue

            keys, values = zip(*param_grid.items())
            for combination in itertools.product(*values):
                run_params = dict(zip(keys, combination))
                run_params['baseline'] = baseline_name
                param_runs.append({'run_parameters': run_params})

    if not param_runs:
        param_runs.append({'run_parameters': config.get('model_params', {})})

    # ---------------------------------------------------------
    # Evaluation Loop
    # ---------------------------------------------------------
    all_results = []

    for i, run_config in enumerate(param_runs):
        run_params = run_config['run_parameters']
        print(f"\n===== Running evaluation for parameter set {i+1}/{len(param_runs)} =====")
        print(f"Parameters: {run_params}")

        run_results = {"run_parameters": run_params}

        # Seed
        seed = config.get('seeds', [42])[0]
        helpers.set_seed(seed)

        configured_generate = partial(generate, params=run_params)

        global_total_tokens = 0

        # -----------------------------------------------------
        # Accuracy Evaluation
        # -----------------------------------------------------
        if 'datasets' in config:
            print(f"\n--- Running Accuracy Evaluations ---")
            for dataset_name, run_func_path in config['datasets'].items():
                _, func_name = run_func_path.rsplit('.', 1)
                try:
                    dataset_module = importlib.import_module(f"dataset_suite.{dataset_name}")
                    run_func = getattr(dataset_module, func_name)

                    accuracy, total_tokens = run_func(configured_generate)
                    global_total_tokens += total_tokens

                    run_results[f"{dataset_name}_accuracy"] = accuracy
                    run_results[f"{dataset_name}_total_tokens"] = total_tokens

                except (ImportError, AttributeError) as e:
                    print(f"Could not run {dataset_name}: {e}")

        # -----------------------------------------------------
        # Speed & Efficiency Metrics
        # -----------------------------------------------------
        if 'metrics' in config and 'speed_efficiency' in config['metrics']:
            speed_cfg = config['metrics']['speed_efficiency']

            speed_results = measure_speed_and_efficiency(
                total_tokens=global_total_tokens,
                total_elapsed_time=model_instance.total_elapsed_time,
                total_gpu_bytes=model_instance.absolute_gpu_peak_memory,
                batch_sizes=speed_cfg.get('batch_sizes', [1])
            )

            run_results['speed_efficiency'] = speed_results

        # -----------------------------------------------------
        # Diversity Metrics
        # -----------------------------------------------------
        if 'metrics' in config and 'diversity' in config['metrics'] and metric_prompts:
            diversity_cfg = config['metrics']['diversity']

            n_samples = diversity_cfg.get("num_samples_to_generate", 20)
            prompts_for_diversity = metric_prompts[:n_samples]

            generated_samples = [configured_generate(p)[0] for p in prompts_for_diversity]

            diversity_results = {}
            for n in diversity_cfg.get('n_values', [2, 3]):
                diversity_results[f'distinct_{n}'] = calculate_distinct_n(generated_samples, n=n)

            diversity_results['entropy'] = calculate_entropy(generated_samples)

            run_results['diversity'] = diversity_results

        all_results.append(run_results)

    # ---------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------
    reporting.generate_reproducibility_report(config, system_info, {"all_runs": all_results})
    reporting.print_summary_for_readme(config, all_results)
    helpers.save_results({"all_runs": all_results}, "full_evaluation_results.json")

    print("\nEvaluation run finished.")


if __name__ == '__main__':
    main()
