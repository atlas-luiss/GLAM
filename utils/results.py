import json
import os
from datetime import datetime

import numpy as np


def log_results(
    config,
    accuracy_matrix,
    performance_stats=None,
    merged_accuracies=None,
    groups=None,
    forgetting=None,
):
    """Save the experiment configuration, accuracy results, and efficiency metrics."""
    result_dir = create_result_dir(config)
    accuracy_matrix = np.asarray(accuracy_matrix, dtype=float)

    with open(os.path.join(result_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    np.save(os.path.join(result_dir, "accuracy_matrix.npy"), accuracy_matrix)
    with open(os.path.join(result_dir, "accuracy_matrix.json"), "w") as f:
        json.dump(accuracy_matrix.tolist(), f, indent=4)

    if performance_stats is not None:
        with open(os.path.join(result_dir, "performance.json"), "w") as f:
            json.dump(performance_stats, f, indent=4)

    if merged_accuracies is not None:
        merged_accuracies = np.asarray(merged_accuracies, dtype=float)
        np.save(os.path.join(result_dir, "merged_accuracies.npy"), merged_accuracies)
        with open(os.path.join(result_dir, "merged_accuracies.json"), "w") as f:
            json.dump(merged_accuracies.tolist(), f, indent=4)

    if groups is not None:
        with open(os.path.join(result_dir, "groups.json"), "w") as f:
            json.dump(groups, f, indent=4)

    if forgetting is not None:
        with open(os.path.join(result_dir, "forgetting.json"), "w") as f:
            json.dump(forgetting, f, indent=4)

    return result_dir


def create_result_dir(config):
    if config.get("results_path"):
        result_dir = config["results_path"]
    else:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method_name = config["active_method"]["name"]
        result_dir = os.path.join("results", f"{method_name}_{timestamp}")

    os.makedirs(result_dir, exist_ok=True)
    return result_dir
