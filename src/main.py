"""
Main training script for Geneformer fine-tuning
"""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils import load_config, setup_environment, print_config
from src.data.dataset_loader import load_dataset, save_dataset_to_disk, analyze_dataset
from src.models.classifier import create_classifier, get_training_args
from src.training.trainer import prepare_training_data, train_model, get_output_directory
from src.evaluation.evaluator import evaluate_model, print_metrics


def main(args):
    """
    Main training pipeline
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*80)
    print(" "*20 + "GENEFORMER FINE-TUNING")
    print("="*80 + "\n")
    
    # Load configuration
    config = load_config(args.config)
    if args.verbose:
        print_config(config)
    
    # Setup environment
    setup_environment(config)
    
    # Get configuration values
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    logging_config = config.get("logging", {})
    
    model_name = model_config.get("name", "ctheodoris/Geneformer")
    model_version = model_config.get("version", "V1")
    freeze_layers = model_config.get("freeze_layers", 0)
    
    dataset_file = args.data or data_config.get("dataset_file", "dataset.arrow")
    cell_types = data_config.get("cell_types", [])
    max_cells = data_config.get("max_cells", 50000)
    state_key = data_config.get("cell_state_key", "cell_type")
    
    output_prefix = logging_config.get("output_prefix", "classifier")
    base_output_dir = logging_config.get("output_dir", "outputs")
    
    # Create output directory
    output_dir = args.output_dir or get_output_directory(base_output_dir)
    print(f"📁 Output directory: {output_dir}\n")
    
    # Step 1: Load dataset
    print("Step 1/5: Loading dataset")
    dataset = load_dataset(dataset_file)
    
    if args.verbose:
        analyze_dataset(dataset, label_column=state_key)
    
    # Step 2: Save dataset to disk
    print("\nStep 2/5: Preparing dataset")
    dataset_dir = os.path.join(output_dir, "dataset_temp")
    save_dataset_to_disk(dataset, dataset_dir)
    
    # Step 3: Initialize classifier
    print("\nStep 3/5: Initializing classifier")
    training_args = get_training_args(config, output_dir)
    
    classifier = create_classifier(
        cell_types=cell_types,
        training_config=training_args,
        max_cells=max_cells,
        freeze_layers=freeze_layers,
        model_version=model_version,
        state_key=state_key
    )
    
    # Step 4: Prepare training data
    if not args.skip_prepare:
        print("\nStep 4/5: Preparing training data")
        id_class_dict_file = prepare_training_data(
            classifier=classifier,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            output_prefix=output_prefix
        )
    else:
        print("\nStep 4/5: Skipping data preparation (using existing)")
    
    # Step 5: Train model
    if not args.skip_train:
        print("\nStep 5/5: Training model")
        metrics = train_model(
            classifier=classifier,
            model_directory=model_name,
            prepared_data_dir=output_dir,
            output_prefix=output_prefix,
            output_dir=output_dir
        )
        
        print(f"Training metrics: {metrics}")
    else:
        print("\nStep 5/5: Skipping training")
    
    # Evaluation (optional)
    if args.evaluate:
        print("\nEvaluating model...")
        test_data_file = f"{output_dir}/{output_prefix}_labeled_test.dataset"
        id_class_dict_file = f"{output_dir}/{output_prefix}_id_class_dict.pkl"
        model_checkpoint = f"{output_dir}/ksplit1"
        
        if os.path.exists(model_checkpoint):
            test_metrics = evaluate_model(
                model_directory=model_checkpoint,
                id_class_dict_file=id_class_dict_file,
                test_data_file=test_data_file,
                output_directory=output_dir,
                output_prefix=output_prefix,
                model_version=model_version,
                plot_results=True
            )
            
            print_metrics(test_metrics)
        else:
            print(f"⚠️  Model checkpoint not found at {model_checkpoint}")
    
    print("\n" + "="*80)
    print(f"✓ Complete! Results saved to: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Geneformer for cell type classification"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        help="Path to dataset file (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: timestamped in outputs/)"
    )
    
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip data preparation (use existing prepared data)"
    )
    
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    main(args)

