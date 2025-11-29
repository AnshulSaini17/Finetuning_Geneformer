"""
Dataset loading and preparation utilities
"""

import os
from typing import Dict, Optional
from datasets import Dataset
import pickle


def load_dataset(data_file: str) -> Dataset:
    """
    Load dataset from arrow file
    
    Args:
        data_file: Path to .arrow file
        
    Returns:
        Loaded dataset
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset not found at: {data_file}")
    
    print(f"Loading dataset from {data_file}")
    dataset = Dataset.from_file(data_file)
    print(f"Loaded {len(dataset)} samples")
    print(f"Features: {list(dataset.features.keys())}")
    
    return dataset


def save_dataset_to_disk(dataset: Dataset, output_dir: str) -> str:
    """
    Args:
        dataset: Dataset to save
        output_dir: Directory to save to
        
    Returns:
        Path to saved dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Saved dataset to {output_dir}")
    
    return output_dir


def analyze_dataset(dataset: Dataset, label_column: str = "cell_type"):

    print("\n" + "="*60)
    print("Dataset Analysis")
    print("="*60)
    
    df = dataset.to_pandas()
    
    print(f"Total samples: {len(df)}")
    print(f"\nFeatures: {list(df.columns)}")
    
    if label_column in df.columns:
        print(f"\nUnique {label_column}s: {df[label_column].nunique()}")
        print(f"\n{label_column.title()} distribution:")
        print(df[label_column].value_counts())
    
    print("="*60 + "\n")


def load_id_class_dict(file_path: str) -> Dict:
    with open(file_path, "rb") as f:
        id_class_dict = pickle.load(f)
    
    print(f"✓ Loaded ID-class mapping")
    print(f"  - Total samples: {len(id_class_dict)}")
    print(f"  - Unique classes: {len(set(id_class_dict.values()))}")
    
    return id_class_dict

