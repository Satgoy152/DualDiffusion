
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_frontier(results, title="Pareto Frontier: Quality vs. Cost"):
    """
    Plots a Pareto frontier of quality (e.g., PPL) vs. cost (e.g., NFEs).

    Args:
        results: A list of dictionaries, where each dict contains:
                 {'method': str, 'cost': float, 'quality': float, 'label': str}
        title: The title of the plot.
    """
    print(f"\nGenerating plot: {title}")
    plt.figure(figsize=(10, 7))
    
    methods = sorted(list(set(r['method'] for r in results)))
    
    for method in methods:
        method_results = [r for r in results if r['method'] == method]
        costs = [r['cost'] for r in method_results]
        qualities = [r['quality'] for r in method_results]
        
        # Sort by cost to draw lines
        sorted_points = sorted(zip(costs, qualities))
        sorted_costs, sorted_qualities = zip(*sorted_points)
        
        plt.plot(sorted_costs, sorted_qualities, marker='o', linestyle='--', label=method)
        # Optionally, add labels to points
        # for r in method_results:
        #     plt.text(r['cost'], r['quality'], r['label'])

    plt.xlabel("Inference Cost (e.g., NFEs or Wall-clock Time)")
    plt.ylabel("Quality (e.g., Generative PPL - lower is better)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    filename = f"{title.lower().replace(' ', '_').replace(':', '')}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

def plot_bar_chart(data, title="Metric Comparison", xlabel="", ylabel=""):
    """
    Generates a simple bar chart.

    Args:
        data: A dictionary of {label: value}.
        title: The plot title.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
    """
    print(f"\nGenerating bar chart: {title}")
    labels = list(data.keys())
    values = list(data.values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Bar chart saved to {filename}")
    plt.close()


if __name__ == '__main__':
    # Example for Pareto Frontier plot
    pareto_data = [
        # Our method
        {'method': 'Our Speculative', 'cost': 10, 'quality': 25, 'label': 'S=5'},
        {'method': 'Our Speculative', 'cost': 20, 'quality': 18, 'label': 'S=10'},
        {'method': 'Our Speculative', 'cost': 35, 'quality': 15, 'label': 'S=20'},
        # Baseline
        {'method': 'Diffusion Baseline', 'cost': 25, 'quality': 22, 'label': 'DDIM-25'},
        {'method': 'Diffusion Baseline', 'cost': 50, 'quality': 17, 'label': 'DDIM-50'},
        {'method': 'Diffusion Baseline', 'cost': 100, 'quality': 14, 'label': 'DDIM-100'},
        # AR Baseline
        {'method': 'AR Baseline', 'cost': 80, 'quality': 12, 'label': 'Top-p=0.9'},
    ]
    plot_pareto_frontier(pareto_data)

    # Example for Bar Chart
    accuracy_data = {
        'GSM8K': 0.75,
        'MMLU': 0.68,
        'ARC-Challenge': 0.82
    }
    plot_bar_chart(accuracy_data, title="Accuracy on Benchmarks", ylabel="Accuracy")
