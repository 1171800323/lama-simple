import os

import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: config
    """
    ext = os.path.splitext(file_path)[1]
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return edict(config)


def move_to_device(obj, device):
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')
