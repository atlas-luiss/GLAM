import torch


def resolve_device(device=None):
    """Resolve config/CLI device values into a torch.device."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(device)


def get_num_tasks(config):
    """Read num_tasks from dataset config or legacy locations."""
    if "dataset" in config and "num_tasks" in config["dataset"]:
        return config["dataset"]["num_tasks"]

    if "num_tasks" in config:
        return config["num_tasks"]

    return config["training"]["num_tasks"]


def get_dataset_name(config):
    """Read dataset name from dataset config or legacy top-level key."""
    if "dataset" in config and "name" in config["dataset"]:
        return config["dataset"]["name"]

    return config["dataset_name"]


def get_data_path(config):
    """Read data path from dataset config or legacy top-level key."""
    if "dataset" in config and "data_path" in config["dataset"]:
        return config["dataset"]["data_path"]

    return config["data_path"]

