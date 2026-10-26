import os
import json
import logging
from collections import Counter

import torch
import yaml
from sklearn.model_selection import train_test_split

from utils import seed_everything

logging.basicConfig(level=logging.INFO)
logging.info("Loading configuration from configs/resnet18.yaml")

with open('configs/resnet18.yaml', 'r') as file:
    config = yaml.safe_load(file)


def data_setup():
    seed = config['.global_settings']['seed']
    test_size = config['data']['test_size']
    restratify = config['data']['restratify']

    train_path = config['data']['train_path']
    test_path = config['data']['test_path']
    val_path = config['data']['val_path']

    train_original = torch.datasets.ImageFolder(train_path, transform=config['data']['transforms']['train'])
    val_original = torch.datasets.ImageFolder(val_path, transform=config['data']['transforms']['val'])
    test_dataset = torch.datasets.ImageFolder(test_path, transform=config['data']['transforms']['test'])

    if restratify:
        # Stratify Train into Train and Validation sets
        # Original Val set has very few samples, so we will get some from train
        train_indices, val_indices = train_test_split(list(range(len(train_original))),
                                                    test_size=test_size,
                                                    random_state=seed,
                                                    stratify=train_original.targets)
        # Dump indices into json for reproducibility
        with open('train_val_indices.json', 'w') as f:
            json.dump({'train_indices': train_indices, 'val_indices': val_indices}, f)
        logging.info("Train and validation indices saved to train_val_indices.json")
        logging.info(f'Check for overlapping indices: {set(train_indices) & set(val_indices)}')

    else:
        # Load train and val indices from JSON
        with open(config['data']['train_val_indices_path'], 'r') as f:
            loaded_indices = json.load(f)
            train_indices = loaded_indices['train_indices']
            val_indices = loaded_indices['val_indices']
        logging.info(f'Loaded train indices: {len(loaded_indices["train_indices"])}, Loaded val indices: {len(loaded_indices["val_indices"])}')
        logging.info(f'Loaded train class distribution: {Counter([train_original.targets[i] for i in loaded_indices["train_indices"]])}')
        logging.info(f'Loaded val class distribution: {Counter([train_original.targets[i] for i in loaded_indices["val_indices"]])}')


    train_dataset = torch.utils.data.Subset(train_original, train_indices)

    val_subset = torch.utils.data.Subset(train_original, val_indices)
    val_dataset = torch.utils.data.ConcatDataset([val_original, val_subset])

    logging.info(f'Train dataset info:\nLength: {len(train_dataset)}\nclasses: {train_dataset.dataset.classes}\nclass sizes: {Counter([train_dataset.dataset.targets[i] for i in train_dataset.indices])}')
    logging.info(f'Val dataset info:\nLength: {len(val_dataset)}\nclasses: {val_original.classes}\nclass sizes: {Counter(val_original.targets) + Counter([val_dataset.datasets[1].dataset.targets[i] for i in val_dataset.datasets[1].indices])}')

    logging.info(f'Train dataset class distribution: {Counter([train_dataset.dataset.targets[i] for i in train_dataset.indices])}')
    logging.info(f'Val dataset class distribution: {Counter(val_original.targets) + Counter([val_dataset.datasets[1].dataset.targets[i] for i in val_dataset.datasets[1].indices])}')

    logging.info(f'Test dataset info:\nLength: {len(test_dataset)}\nclasses: {test_dataset.classes}\nclass sizes: {Counter(test_dataset.targets)}')
    logging.info(f'Test dataset class distribution: {Counter(test_dataset.targets)}')

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size, train_dataset, val_dataset, test_dataset):
    seed_everything(seed=42)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
