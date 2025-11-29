"""
Geneformer classifier wrapper and utilities
"""

import os
from typing import Dict, List, Optional
from geneformer import Classifier
from .model_patch import apply_model_patch


def create_classifier(
    cell_types: List[str],
    training_config: Dict,
    max_cells: int = 50000,
    freeze_layers: int = 0,
    model_version: str = "V1",
    state_key: str = "cell_type"
) -> Classifier:
    """
    Initialize Geneformer classifier
    
    Args:
        cell_types: List of cell type labels to classify
        training_config: Training configuration dictionary
        max_cells: Maximum number of cells to use
        freeze_layers: Number of layers to freeze (0 = train all)
        model_version: Model version ("V1", "V2")
        state_key: Column name for cell state labels
        
    Returns:
        Initialized Classifier object
    """
    # Apply model loading patch (required for V1 models)
    if model_version == "V1":
        apply_model_patch()
    
    # Disable wandb if not explicitly enabled
    if not training_config.get("wandb_enabled", False):
        os.environ['WANDB_DISABLED'] = 'true'
    
    cell_state_dict = {
        "state_key": state_key,
        "states": cell_types
    }
    
    classifier = Classifier(
        classifier="cell",
        cell_state_dict=cell_state_dict,
        training_args=training_config,
        max_ncells=max_cells,
        freeze_layers=freeze_layers,
        model_version=model_version,
        nproc=1,
    )
    
    print("✓ Classifier initialized")
    print(f"  - Model version: {model_version}")
    print(f"  - Cell types: {len(cell_types)}")
    print(f"  - Max cells: {max_cells}")
    print(f"  - Frozen layers: {freeze_layers}")
    
    return classifier


def get_training_args(config: Dict, output_dir: str) -> Dict:
    """
    Convert config to Transformers TrainingArguments format
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for checkpoints
        
    Returns:
        Training arguments dictionary
    """
    training_config = config.get("training", {})
    eval_config = config.get("evaluation", {})
    logging_config = config.get("logging", {})
    dataloader_config = config.get("dataloader", {})
    
    return {
        # Training
        "num_train_epochs": training_config.get("num_epochs", 3),
        "learning_rate": training_config.get("learning_rate", 5e-5),
        "per_device_train_batch_size": training_config.get("batch_size", 16),
        "gradient_accumulation_steps": training_config.get("gradient_accumulation_steps", 1),
        "warmup_steps": training_config.get("warmup_steps", 500),
        "weight_decay": training_config.get("weight_decay", 0.001),
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
        
        # Precision
        "fp16": training_config.get("fp16", False),
        "bf16": training_config.get("bf16", True),
        "tf32": training_config.get("tf32", True),
        
        # Evaluation
        "evaluation_strategy": eval_config.get("strategy", "steps"),
        "eval_steps": eval_config.get("eval_steps", 1000),
        "save_strategy": eval_config.get("save_strategy", "steps"),
        "save_steps": eval_config.get("save_steps", 1000),
        "save_total_limit": eval_config.get("save_total_limit", 3),
        "load_best_model_at_end": eval_config.get("load_best_model_at_end", True),
        "metric_for_best_model": eval_config.get("metric_for_best_model", "eval_loss"),
        
        # Logging
        "logging_steps": logging_config.get("steps", 500),
        "output_dir": output_dir,
        "report_to": "wandb" if logging_config.get("wandb_enabled", False) else "none",
        
        # Dataloader
        "dataloader_num_workers": dataloader_config.get("num_workers", 4),
        "dataloader_pin_memory": dataloader_config.get("pin_memory", True),
        "dataloader_prefetch_factor": dataloader_config.get("prefetch_factor", 2),
        "group_by_length": dataloader_config.get("group_by_length", False),
    }

