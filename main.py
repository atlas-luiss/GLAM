import argparse
import random

import numpy as np
import torch
import yaml

from methods import get_method
from utils.data import create_benchmark, infer_num_classes
from utils.results import log_results

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_config(config):
    """Print the full experiment configuration."""
    def print_section(section, indent=0):
        for key, value in section.items():
            prefix = " " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_section(value, indent + 2)
            else:
                print(f"{prefix}{key}: {value}")

    print("==== Configuration ====")
    print_section(config)
    print("======================")

def main(args):
    """Main function to run the experiment."""
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Update config with CLI overrides, but ignore unset optional args (None)
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update(cli_overrides)
    config.setdefault("seed", 0)
    
    # Set seed
    set_seed(config['seed'])

    # Print configuration
    print_config(config)
    
    # Create benchmark
    print("Creating benchmark...")
    scenario = create_benchmark(config)
    config["num_classes"] = infer_num_classes(scenario)

    # Create method
    print("Creating method...")
    method = get_method(config)
    
    # Run experiment
    try:
        print("Starting experiment...")
        method.run_experiment(scenario)
        
        print("Logging results...")
        performance_stats = None
        if hasattr(method, "performance_monitor"):
            performance_stats = method.performance_monitor.get_all_stats()

        result_dir = log_results(
            config=config,
            accuracy_matrix=method.accuracy_matrix,
            performance_stats=performance_stats,
            merged_accuracies=getattr(method, "merged_accuracies", None),
            groups=getattr(method, "final_groups", None),
            forgetting=getattr(method, "forgetting", None),
        )
        
        print(f"Experiment completed successfully! Results saved to {result_dir}")
    except KeyboardInterrupt:
        print("Experiment interrupted. Cleaning up...")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GLAM continual-learning experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (overrides config)")
    parser.add_argument("--results_path", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()
    main(args)
