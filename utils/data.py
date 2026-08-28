import os

import torch
import torchvision
from avalanche.benchmarks import nc_benchmark
from sklearn.model_selection import train_test_split
from torchvision import datasets as datasets
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from utils.config import get_data_path, get_dataset_name, get_num_tasks

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, targets, transform=None):
        self.samples = list(zip(paths, targets))
        self.targets = list(targets)
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)

def create_transforms(dataset_name):
    """Create the dataset-specific transformations used by HAM-CL."""
    supported_datasets = {"cifar100", "cub200", "imagenet-r", "imagenet-a"}
    if dataset_name not in supported_datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name == "cifar100":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif dataset_name == "cub200":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                224,
                scale=(0.05, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    return train_transform, test_transform


def split_image_folder(data_path, seed):
    full_dataset = torchvision.datasets.ImageFolder(data_path)

    paths, targets = zip(*full_dataset.samples)
    train_paths, test_paths, train_targets, test_targets = train_test_split(
        paths,
        targets,
        test_size=0.3,
        stratify=targets,
        random_state=seed,
    )

    train_dataset = ImageDataset(train_paths, train_targets)
    test_dataset = ImageDataset(test_paths, test_targets)
    return train_dataset, test_dataset


def load_cub200_datasets(data_path):
    def read_mapping(file_name, value_fn):
        mapping = {}
        with open(os.path.join(data_path, file_name)) as f:
            for line in f:
                image_id, value = line.strip().split(maxsplit=1)
                mapping[int(image_id)] = value_fn(value)
        return mapping

    image_paths = read_mapping("images.txt", str)
    labels = read_mapping("image_class_labels.txt", lambda value: int(value) - 1)
    split_flags = read_mapping("train_test_split.txt", int)

    train_paths = []
    train_targets = []
    test_paths = []
    test_targets = []

    for image_id, relative_path in image_paths.items():
        image_path = os.path.join(data_path, "images", relative_path)
        target = labels[image_id]

        if split_flags[image_id] == 1:
            train_paths.append(image_path)
            train_targets.append(target)
        else:
            test_paths.append(image_path)
            test_targets.append(target)

    train_dataset = ImageDataset(train_paths, train_targets)
    test_dataset = ImageDataset(test_paths, test_targets)
    return train_dataset, test_dataset

def create_benchmark(config):
    """
    Create an Avalanche continual learning benchmark from the configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Avalanche scenario
    """
    dataset_name = get_dataset_name(config)
    num_tasks = get_num_tasks(config)
    data_path = get_data_path(config)
    seed = config['seed']
    train_transform, test_transform = create_transforms(dataset_name)
    train_dataset = None
    test_dataset = None
    
    if dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=data_path, 
            train=True, 
            download=True,
        )
        test_dataset = datasets.CIFAR100(
            root=data_path, 
            train=False, 
            download=True,
        )
    
    elif dataset_name == "cub200":
        train_dataset, test_dataset = load_cub200_datasets(
            data_path,
        )

    elif dataset_name == "imagenet-r":
        train_dataset, test_dataset = split_image_folder(
            data_path,
            seed,
        )
    
    elif dataset_name == "imagenet-a":
        train_dataset, test_dataset = split_image_folder(
            data_path,
            seed,
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=num_tasks,
        shuffle=True,
        seed=seed,
        task_labels=False,
        train_transform=train_transform,
        eval_transform=test_transform,
    )


def infer_num_classes(scenario):
    """Infer classifier output size from the classes in an Avalanche scenario."""
    class_ids = set()

    for stream_name in ("train_stream", "test_stream"):
        stream = getattr(scenario, stream_name, [])
        for experience in stream:
            classes = getattr(experience, "classes_in_this_experience", None)
            if classes is not None:
                class_ids.update(int(class_id) for class_id in classes)

    if not class_ids:
        raise ValueError("Could not infer num_classes from benchmark scenario.")

    return max(class_ids) + 1
