
import sys
import platform
import os

def get_system_info():
    """
    Gathers and returns key system and environment information.
    """
    info = {
        "python_version": sys.version.split('\n')[0],
        "platform": platform.platform(),
        "hardware": platform.machine(),
        # In a real scenario, you'd add library versions e.g. 
        # "torch_version": torch.__version__,
        # "cuda_version": torch.version.cuda
    }
    return info

def generate_reproducibility_report(config, hardware_info, results_summary):
    """
    Generates a text report for reproducibility.

    Args:
        config: The configuration dictionary for the run.
        hardware_info: A dictionary from get_system_info().
        results_summary: A high-level summary of the results, possibly containing a list of runs.
    """
    report = []
    report.append("="*40)
    report.append("REPRODUCIBILITY REPORT")
    report.append("="*40)
    
    report.append("\n[Configuration]")
    for key, value in config.items():
        report.append(f"  - {key}: {value}")
        
    report.append("\n[System & Environment]")
    for key, value in hardware_info.items():
        report.append(f"  - {key}: {value}")
        
    report.append("\n[Results Summary]")
    if "all_runs" in results_summary and isinstance(results_summary["all_runs"], list):
        for i, run_results in enumerate(results_summary["all_runs"]):
            report.append(f"\n--- Run {i+1} ---")
            for key, value in run_results.items():
                report.append(f"  - {key}: {value}")
    else:
        for key, value in results_summary.items():
            report.append(f"  - {key}: {value}")
        
    report.append("\n[Notes]")
    report.append("  - This report contains the key details to reproduce the experiment.")
    report.append("  - For full details, refer to the saved artifacts (configs, results.json).")

    full_report = "\n".join(report)
    
    # Save to file
    with open("reproducibility_report.txt", "w") as f:
        f.write(full_report)
        
    print("\nGenerated reproducibility report to reproducibility_report.txt")
    
    return full_report

def print_summary_for_readme(config, main_results_list):
    """
    Prints a markdown-formatted summary suitable for a README, handling
    a list of results from multiple parameter runs.
    """
    
    summary = []
    summary.append(f"## Experiment: {config.get('experiment_name', 'N/A')}")
    summary.append(f"**Model:** `{config.get('model_path', 'N/A')}`")
    
    summary.append("\n### Key Results")

    if not isinstance(main_results_list, list):
        main_results_list = [main_results_list]

    for i, main_results in enumerate(main_results_list):
        summary.append(f"\n#### Run {i+1}")
        
        run_params = main_results.get("run_parameters")
        if run_params:
            params_str = ", ".join([f"`{k}={v}`" for k, v in run_params.items()])
            summary.append(f"**Parameters:** {params_str}")

        for key, val in main_results.items():
            if key == "run_parameters":
                continue
            if isinstance(val, dict):
                summary.append(f"- **{key}**:")
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, dict):
                        summary.append(f"  - `{sub_key}`:")
                        for ssub_key, ssub_val in sub_val.items():
                            formatted_val = f"{ssub_val:.4f}" if isinstance(ssub_val, float) else ssub_val
                            summary.append(f"    - `{ssub_key}`: {formatted_val}")
                    else:
                        formatted_val = f"{sub_val:.4f}" if isinstance(sub_val, float) else sub_val
                        summary.append(f"  - `{sub_key}`: {formatted_val}")
            else:
                formatted_val = f"{val:.4f}" if isinstance(val, float) else val
                summary.append(f"- **{key}**: {formatted_val}")
        
    summary.append("\n### How to Run")
    summary.append("```bash")
    summary.append(f"python eval/main.py --config {config.get('config_file', 'path/to/your.json')}")
    summary.append("```")

    print("\n--- README Summary ---")
    print("\n".join(summary))
    print("--- End README Summary ---")


if __name__ == '__main__':
    # Example usage
    system_details = get_system_info()
    print("System Info:", system_details)

    run_config = {
        'experiment_name': 'Spec-Decode-Alpha-Run',
        'model': 'diffusion_lm_v1',
        'seeds': [42, 100, 200],
        'batch_size': 8,
        'datasets': ['gsm8k', 'mmlu']
    }
    
    final_results = {
        'mean_gsm8k_accuracy': '0.78 ± 0.03',
        'mean_mmlu_accuracy': '0.71 ± 0.04',
        'avg_tokens_per_sec': 120.5
    }

    generate_reproducibility_report(run_config, system_details, final_results)
    
    run_config['config_file'] = 'configs/experiment1.json'
    print_summary_for_readme(run_config, final_results)
