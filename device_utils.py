# Valentin Macé
# valentin.mace@kedgebs.com
# Developed for fun
# Feel free to use this code as you wish as long as you quote me as author

"""
device_utils.py
~~~~~~~~~~

Device detection and management utilities for CUDA/MPS/CPU acceleration
"""

import torch
import warnings

def get_device(preferred_device=None):
    """
    Automatically detect the best available device or use preferred device
    
    :param preferred_device: str, 'cuda', 'mps', or 'cpu' to force specific device
    :return: torch.device object
    """
    if preferred_device:
        if preferred_device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif preferred_device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif preferred_device == 'cpu':
            return torch.device('cpu')
        else:
            warnings.warn(f"Preferred device '{preferred_device}' not available, falling back to auto-detection")
    
    # Auto-detection
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_device_info(device):
    """
    Get detailed information about the device
    
    :param device: torch.device object
    :return: dict with device information
    """
    info = {
        'type': device.type,
        'name': str(device),
        'available_memory': None,
        'device_count': 1
    }
    
    if device.type == 'cuda':
        info['name'] = torch.cuda.get_device_name(device)
        info['available_memory'] = f"{torch.cuda.get_device_properties(device).total_memory // 1024**3} GB"
        info['device_count'] = torch.cuda.device_count()
    elif device.type == 'mps':
        info['name'] = "Apple Metal Performance Shaders"
        info['available_memory'] = "Shared with system memory"
    else:
        info['name'] = "CPU"
        
    return info

def print_device_info(device):
    """
    Print formatted device information
    
    :param device: torch.device object
    """
    info = get_device_info(device)
    print(f"Using device: {info['name']} ({info['type'].upper()})")
    if info['available_memory']:
        print(f"Available memory: {info['available_memory']}")
    if info['device_count'] > 1:
        print(f"Available devices: {info['device_count']}")

def tensor_to_device(tensor, device):
    """
    Move tensor to specified device if it's a torch tensor
    
    :param tensor: torch.Tensor or other object
    :param device: torch.device object
    :return: tensor moved to device or original object
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor