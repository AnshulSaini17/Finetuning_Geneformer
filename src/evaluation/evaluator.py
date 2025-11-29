"""
Model evaluation utilities
"""

import os
from typing import Dict
from geneformer import Classifier
from ..models.model_patch import apply_model_patch


def evaluate_model(
    model_directory: str,
    id_class_dict_file: str,
    test_data_file: str,
    output_directory: str,
    output_prefix: str,
    model_version: str = "V1",
    plot_results: bool = True
) -> Dict:
    """
    Evaluate trained model on test set
    
    Args:
        model_directory: Path to trained model checkpoint
        id_class_dict_file: Path to ID-class mapping pickle
        test_data_file: Path to test dataset
        output_directory: Directory to save evaluation results
        output_prefix: Prefix for output files
        model_version: Model version ("V1", "V2")
        plot_results: Whether to generate plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("Evaluating Model")
    print("="*60)
    
    # Apply model loading patch if using V1
    if model_version == "V1":
        apply_model_patch()
    
    # Initialize evaluator
    evaluator = Classifier(
        classifier="cell",
        model_version=model_version
    )
    
    # Run evaluation
    print(f"\nModel: {model_directory}")
    print(f"Test data: {test_data_file}")
    
    test_metrics = evaluator.evaluate_saved_model(
        model_directory=model_directory,
        id_class_dict_file=id_class_dict_file,
        test_data_file=test_data_file,
        output_directory=output_directory,
        output_prefix=output_prefix,
        predict=True,
    )
    
    print("\n✓ Evaluation complete")
    print(f"Test metrics: {test_metrics}")
    
    # Generate plots if requested
    if plot_results:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        evaluator.plot_conf_mat(
            conf_mat_dict=test_metrics,
            output_directory=output_directory,
            output_prefix=output_prefix,
        )
        print("  ✓ Confusion matrix saved")
        
        # Prediction plots
        predictions_file = f"{output_directory}/{output_prefix}_ksplit1_predictions.pkl"
        if os.path.exists(predictions_file):
            evaluator.plot_predictions(
                predictions_file=predictions_file,
                id_class_dict_file=id_class_dict_file,
                output_directory=output_directory,
                output_prefix=output_prefix,
            )
            print("  ✓ Prediction plots saved")
    
    print("="*60 + "\n")
    
    return test_metrics


def print_metrics(metrics: Dict):
    """
    Pretty print evaluation metrics
    
    Args:
        metrics: Dictionary of metrics from evaluation
    """
    print("\n" + "="*60)
    print("Model Performance")
    print("="*60)
    
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {value}")
    
    print("="*60 + "\n")

