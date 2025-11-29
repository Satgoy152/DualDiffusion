
import random
import numpy as np
import json

def set_seed(seed_value):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    # If using torch, also set torch.manual_seed(seed_value)
    print(f"Random seed set to {seed_value}")

def save_results(results, filename="results.json"):
    """
    Saves a dictionary of results to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

def load_config(config_path):
    """
    Loads a configuration file (e.g., YAML or JSON).
    For this example, we'll stick to JSON for simplicity without new dependencies.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from {config_path}")
    return config

if __name__ == '__main__':
    # Example usage
    set_seed(42)
    
    # Demonstrate that the seed works for random
    print(f"First random number: {random.random()}")
    print(f"Second random number: {random.random()}")
    
    # Reset seed to get the same sequence
    set_seed(42)
    print(f"First random number after reset: {random.random()}")

    # Example of saving results
    my_results = {
        "model": "TestModel",
        "accuracy": 0.95,
        "metrics": {
            "perplexity": 20.5,
            "speed_tps": 50.1
        }
    }
    save_results(my_results, "test_results.json")
    
    # Example of creating and loading a config
    dummy_config = {
        "experiment_name": "My First Eval",
        "model_path": "/path/to/my/model",
        "datasets": ["gsm8k", "mmlu"],
        "seeds": [42, 123, 1024]
    }
    with open("dummy_config.json", "w") as f:
        json.dump(dummy_config, f, indent=4)
        
    loaded_conf = load_config("dummy_config.json")
    print("Loaded config name:", loaded_conf["experiment_name"])
