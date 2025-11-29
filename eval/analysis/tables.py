
import pandas as pd

def create_main_results_table(results_data):
    """
    Creates and prints a main results table using pandas.

    Args:
        results_data: A list of dictionaries, where each dict is a row
                      with keys like 'Method', 'Task', 'Accuracy', 'Gen-PPL', etc.

    Returns:
        A pandas DataFrame of the main results.
    """
    print("\nGenerating Main Results Table...")
    
    if not results_data:
        print("No data provided for the results table.")
        return None

    df = pd.DataFrame(results_data)
    
    # Example of pivoting to get a nice format: Method vs. Task
    try:
        pivot_df = df.pivot(index='Method', columns='Task', values='Accuracy')
        print("--- Accuracy ---")
        print(pivot_df.to_markdown())
        
        # You could create other pivots for other metrics (Gen-PPL, NFEs, etc.)
        
    except Exception as e:
        print(f"Could not pivot the table. Displaying raw data instead. Error: {e}")
        print(df.to_markdown())

    # You can also save this to a file
    df.to_csv("main_results.csv", index=False)
    print("\nTable saved to main_results.csv")
    
    return df

def create_hyperparameter_table(hyperparam_data):
    """
    Creates a table of hyperparameters used for different experiments.

    Args:
        hyperparam_data: List of dicts, each with experiment details.

    Returns:
        A pandas DataFrame of the hyperparameters.
    """
    print("\nGenerating Hyperparameter Table...")
    if not hyperparam_data:
        print("No data for hyperparameter table.")
        return None
        
    df = pd.DataFrame(hyperparam_data)
    print(df.to_markdown(index=False))
    
    df.to_csv("hyperparameters.csv", index=False)
    print("\nTable saved to hyperparameters.csv")
    
    return df

if __name__ == '__main__':
    # Example data for Main Results Table
    main_results = [
        {'Method': 'Our Speculative', 'Task': 'GSM8K', 'Accuracy': '0.75 ± 0.02', 'Gen-PPL': 15.2, 'NFEs': 35},
        {'Method': 'Our Speculative', 'Task': 'MMLU', 'Accuracy': '0.68 ± 0.03', 'Gen-PPL': 18.1, 'NFEs': 35},
        {'Method': 'Diffusion Baseline', 'Task': 'GSM8K', 'Accuracy': '0.72 ± 0.02', 'Gen-PPL': 17.5, 'NFEs': 100},
        {'Method': 'Diffusion Baseline', 'Task': 'MMLU', 'Accuracy': '0.65 ± 0.03', 'Gen-PPL': 20.3, 'NFEs': 100},
        {'Method': 'AR Baseline', 'Task': 'GSM8K', 'Accuracy': '0.80 ± 0.01', 'Gen-PPL': 12.0, 'NFEs': 'N/A'},
        {'Method': 'AR Baseline', 'Task': 'MMLU', 'Accuracy': '0.75 ± 0.02', 'Gen-PPL': 14.0, 'NFEs': 'N/A'},
    ]
    create_main_results_table(main_results)

    # Example data for Hyperparameter Table
    hyperparams = [
        {'Experiment': 'Our Speculative - Fast', 'Steps': 20, 'Spec. Length': 5, 'Temp.': 1.0, 'Top-p': 'N/A'},
        {'Experiment': 'Our Speculative - High Quality', 'Steps': 50, 'Spec. Length': 10, 'Temp.': 1.0, 'Top-p': 'N/A'},
        {'Experiment': 'AR Baseline - Balanced', 'Steps': 'N/A', 'Spec. Length': 'N/A', 'Temp.': 1.0, 'Top-p': 0.9},
    ]
    create_hyperparameter_table(hyperparams)
