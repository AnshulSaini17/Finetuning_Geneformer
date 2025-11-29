"""
Utility functions
"""

import yaml
import os
from typing import Dict


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded configuration from {config_path}")
    return config


def setup_environment(config: Dict):
    """
    Setup environment variables and settings
    
    Args:
        config: Configuration dictionary
    """
    env_config = config.get("environment", {})
    
    # Set random seed
    seed = env_config.get("seed", 42)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Disable wandb if not enabled
    logging_config = config.get("logging", {})
    if not logging_config.get("wandb_enabled", False):
        os.environ['WANDB_DISABLED'] = 'true'
    
    print("✓ Environment configured")


def print_config(config: Dict):
    """
    Pretty print configuration
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    
    for section, values in config.items():
        if isinstance(values, dict):
            print(f"\n{section.upper()}:")
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"{section}: {values}")
    
    print("="*60 + "\n")

