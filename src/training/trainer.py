"""
Training utilities for Geneformer fine-tuning
"""

import os
import pickle
from typing import Dict, Optional
from geneformer import Classifier


def prepare_training_data(
    classifier: Classifier,
    dataset_dir: str,
    output_dir: str,
    output_prefix: str
) -> str:
    """
    Prepare dataset for training
    
    Args:
        classifier: Initialized Classifier object
        dataset_dir: Directory containing the dataset
        output_dir: Output directory for prepared data
        output_prefix: Prefix for output files
        
    Returns:
        Path to ID-class dictionary pickle file
    """
    print("\n" + "="*60)
    print("Preparing Training Data")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    classifier.prepare_data(
        input_data_file=dataset_dir,
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
    
    # Verify prepared data
    id_class_dict_file = f"{output_dir}/{output_prefix}_id_class_dict.pkl"
    
    with open(id_class_dict_file, "rb") as f:
        id_class_dict = pickle.load(f)
    
    n_samples = len(id_class_dict)
    n_classes = len(set(id_class_dict.values()))
    class_labels = sorted(set(id_class_dict.values()))
    
    print(f"\n✓ Data preparation complete")
    print(f"  - Total samples: {n_samples}")
    print(f"  - Number of classes: {n_classes}")
    print(f"  - Class labels: {class_labels}")
    print("="*60 + "\n")
    
    return id_class_dict_file


def train_model(
    classifier: Classifier,
    model_directory: str,
    prepared_data_dir: str,
    output_prefix: str,
    output_dir: str
) -> Dict:
    """
    Train the Geneformer classifier
    
    Args:
        classifier: Initialized Classifier object
        model_directory: Path to pretrained model
        prepared_data_dir: Directory with prepared training data
        output_prefix: Prefix for data files
        output_dir: Output directory for checkpoints
        
    Returns:
        Training metrics dictionary
    """
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    id_class_dict_file = f"{output_dir}/{output_prefix}_id_class_dict.pkl"
    train_data_file = f"{output_dir}/{output_prefix}_labeled_train.dataset"
    
    print(f"Model: {model_directory}")
    print(f"Training data: {train_data_file}")
    print(f"Starting training...\n")
    
    metrics = classifier.validate(
        model_directory=model_directory,
        prepared_input_data_file=train_data_file,
        id_class_dict_file=id_class_dict_file,
        output_directory=output_dir,
        output_prefix=output_prefix,
    )
    
    print("\n✓ Training complete")
    print("="*60 + "\n")
    
    return metrics


def get_output_directory(base_dir: str = "outputs") -> str:
    """
    Create timestamped output directory
    
    Args:
        base_dir: Base directory for outputs
        
    Returns:
        Path to created output directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

