"""
Helper script to use distilled model for fine-tuning

This script helps you load and use your friend's distilled model (model_best.pt)
instead of the full Geneformer 10M model.
"""

import torch
from transformers import BertForSequenceClassification, BertConfig
import os


def load_distilled_model(
    checkpoint_path,
    num_classes=3,
    vocab_size=25426,  # Same as V1
    hidden_size=256,   # Ask your friend for these specs
    num_hidden_layers=6,
    num_attention_heads=4,
    intermediate_size=1024
):
    """
    Load distilled model from .pt checkpoint
    
    Args:
        checkpoint_path: Path to model_best.pt
        num_classes: Number of output classes for classification
        vocab_size: Should match V1 (25426)
        hidden_size: Distilled model hidden size (ask your friend)
        num_hidden_layers: Number of layers (ask your friend)
        num_attention_heads: Number of attention heads (ask your friend)
        intermediate_size: FFN intermediate size (ask your friend)
        
    Returns:
        Loaded model ready for fine-tuning
    """
    print(f"Loading distilled model from: {checkpoint_path}")
    
    # Check if file exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"✓ Checkpoint loaded")
    
    # Inspect checkpoint structure
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("  Structure: {'model_state_dict': ...}")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("  Structure: {'state_dict': ...}")
        else:
            # Assume the checkpoint IS the state dict
            state_dict = checkpoint
            print("  Structure: Direct state dict")
    else:
        raise ValueError("Unexpected checkpoint format")
    
    # Print some keys to understand structure
    print(f"  Total keys: {len(state_dict)}")
    sample_keys = list(state_dict.keys())[:5]
    print(f"  Sample keys: {sample_keys}")
    
    # Create model configuration
    # These parameters should match the distilled model!
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        num_labels=num_classes,
    )
    
    print(f"\n✓ Model config created:")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Layers: {num_hidden_layers}")
    print(f"  - Attention heads: {num_attention_heads}")
    print(f"  - Vocab size: {vocab_size}")
    
    # Create model from config
    model = BertForSequenceClassification(config)
    
    # Load the distilled weights
    # Handle different key formats
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"\n✓ Distilled model weights loaded successfully!")
    except Exception as e:
        print(f"\n⚠️  Error loading weights: {e}")
        print("Trying to match key names...")
        
        # Sometimes keys need 'bert.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('bert.') and not key.startswith('classifier.'):
                new_key = 'bert.' + key
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ Weights loaded with key mapping")
    
    # Get model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1e6:.1f} MB")
    
    return model, config


def save_distilled_model_for_geneformer(model, config, output_dir):
    """
    Save distilled model in Geneformer-compatible format
    
    Args:
        model: Loaded distilled model
        config: Model configuration
        output_dir: Directory to save model
        
    Returns:
        Path to saved model directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and config
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    print(f"✓ Model saved to: {output_dir}")
    print(f"  - pytorch_model.bin")
    print(f"  - config.json")
    
    return output_dir


def inspect_checkpoint(checkpoint_path):
    """
    Inspect a .pt checkpoint to understand its structure
    
    Args:
        checkpoint_path: Path to .pt file
    """
    print(f"Inspecting: {checkpoint_path}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("Checkpoint Type:", type(checkpoint))
    
    if isinstance(checkpoint, dict):
        print("\nTop-level Keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} items")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Try to find state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"\nState Dict Keys ({len(state_dict)} total):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {key}: {state_dict[key].shape}")
        if len(state_dict) > 10:
            print(f"  ... and {len(state_dict) - 10} more")
        
        # Try to infer architecture
        print("\n🔍 Inferring Architecture:")
        
        # Look for embedding layer
        if 'embeddings.word_embeddings.weight' in state_dict:
            vocab_size, hidden_size = state_dict['embeddings.word_embeddings.weight'].shape
            print(f"  Vocab size: {vocab_size}")
            print(f"  Hidden size: {hidden_size}")
        elif 'bert.embeddings.word_embeddings.weight' in state_dict:
            vocab_size, hidden_size = state_dict['bert.embeddings.word_embeddings.weight'].shape
            print(f"  Vocab size: {vocab_size}")
            print(f"  Hidden size: {hidden_size}")
        
        # Count layers
        layer_keys = [k for k in state_dict.keys() if 'layer.' in k or 'encoder.layer.' in k]
        if layer_keys:
            # Extract layer numbers
            import re
            layer_nums = set()
            for key in layer_keys:
                match = re.search(r'layer\.(\d+)', key)
                if match:
                    layer_nums.add(int(match.group(1)))
            num_layers = max(layer_nums) + 1 if layer_nums else 0
            print(f"  Number of layers: {num_layers}")
        
        # Look for attention heads
        attn_keys = [k for k in state_dict.keys() if 'attention.self.query.weight' in k]
        if attn_keys:
            key = attn_keys[0]
            attn_size, hidden = state_dict[key].shape
            num_heads = hidden // 64  # Typical head_dim is 64
            print(f"  Attention heads: ~{num_heads}")


if __name__ == "__main__":
    # Example usage
    print("Usage:")
    print("1. Inspect your friend's model:")
    print("   inspect_checkpoint('model_best.pt')")
    print("")
    print("2. Load and save for Geneformer:")
    print("   model, config = load_distilled_model('model_best.pt')")
    print("   save_distilled_model_for_geneformer(model, config, 'distilled_model/')")

