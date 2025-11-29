
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
        results_summary: A high-level summary of the results.
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

def print_summary_for_readme(config, main_results):
    """
    Prints a markdown-formatted summary suitable for a README.
    """
    
    summary = []
    summary.append(f"# Experiment: {config.get('experiment_name', 'N/A')}")
    summary.append(f"**Date:** {platform.uname().version.split(' ')[-2]}")
    summary.append(f"**Model:** {config.get('model_path', 'N/A')}")
    
    summary.append("\n## Key Results")
    # This is a simple example; you'd format your main_results dict nicely
    for key, val in main_results.items():
        summary.append(f"- **{key}**: {val}")
        
    summary.append("\n## How to Run")
    summary.append("```bash")
    summary.append(f"python main.py --config {config.get('config_file', 'path/to/your.json')}")
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
