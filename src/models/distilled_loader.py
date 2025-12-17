"""
Load and convert distilled models to Geneformer-compatible format
"""

import os
import torch
from transformers import BertModel, BertConfig


def load_distilled_model(
    model_path: str,
    output_dir: str,
    vocab_size: int = 25426,
    hidden_size: int = None,
    num_layers: int = None,
    num_heads: int = None,
    intermediate_size: int = None
):
    """
    Load a distilled model from .pt file and convert to HuggingFace format
    
    Args:
        model_path: Path to .pt file (e.g., model_best.pt)
        output_dir: Where to save converted model
        vocab_size: Vocabulary size (default: 25426 for Geneformer V1)
        hidden_size: If None, auto-detect from checkpoint
        num_layers: If None, auto-detect from checkpoint
        num_heads: If None, auto-detect from hidden_size
        intermediate_size: If None, auto-detect from checkpoint
    
    Returns:
        str: Path to converted model directory
    """
    print(f"\n🔧 Loading distilled model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint; use weights_only to avoid needing training-time deps (e.g. omegaconf)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        full_state_dict = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
    else:
        full_state_dict = checkpoint
    
    print(f"✓ Loaded checkpoint with {len(full_state_dict)} keys")
    
    # Auto-detect architecture from state dict
    detected_vocab_size = vocab_size
    detected_hidden_size = hidden_size
    detected_intermediate_size = intermediate_size
    detected_num_layers = num_layers
    
    # Find embeddings to get vocab size and hidden size
    for key in full_state_dict.keys():
        if 'word_embeddings.weight' in key:
            shape = full_state_dict[key].shape
            detected_vocab_size = shape[0]
            detected_hidden_size = shape[1] if hidden_size is None else hidden_size
            print(f"✓ Detected vocab_size: {detected_vocab_size}")
            print(f"✓ Detected hidden_size: {detected_hidden_size}")
            break
    
    # Count layers
    if num_layers is None:
        import re
        layer_nums = set()
        for key in full_state_dict.keys():
            match = re.search(r'layer\.(\d+)', key)
            if match:
                layer_nums.add(int(match.group(1)))
        detected_num_layers = max(layer_nums) + 1 if layer_nums else 6
        print(f"✓ Detected num_layers: {detected_num_layers}")
    else:
        detected_num_layers = num_layers
    
    # Find intermediate size
    if intermediate_size is None:
        for key in full_state_dict.keys():
            if 'intermediate.dense.weight' in key:
                detected_intermediate_size = full_state_dict[key].shape[0]
                print(f"✓ Detected intermediate_size: {detected_intermediate_size}")
                break
    else:
        detected_intermediate_size = intermediate_size
    
    # Calculate attention heads
    if num_heads is None:
        detected_num_heads = max(1, detected_hidden_size // 64)  # Assume head_dim=64
        print(f"✓ Calculated num_attention_heads: {detected_num_heads}")
    else:
        detected_num_heads = num_heads
    
    # Strip 'model.' prefix and keep only 'bert.*' (remove MLM head)
    bert_state_dict = {}
    for key, value in full_state_dict.items():
        # Remove 'model.' prefix if present
        new_key = key[6:] if key.startswith('model.') else key
        
        # Only keep bert.* keys (skip cls.predictions.* or other heads)
        if new_key.startswith('bert.'):
            # Remove 'bert.' prefix (HuggingFace BertModel expects no prefix)
            final_key = new_key[5:]
            bert_state_dict[final_key] = value
    
    print(f"✓ Extracted {len(bert_state_dict)} BERT encoder keys")
    print(f"  (Removed {len(full_state_dict) - len(bert_state_dict)} non-encoder keys)")
    
    # Create config
    config = BertConfig(
        vocab_size=detected_vocab_size,
        hidden_size=detected_hidden_size,
        num_hidden_layers=detected_num_layers,
        num_attention_heads=detected_num_heads,
        intermediate_size=detected_intermediate_size or detected_hidden_size * 4,
        max_position_embeddings=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=2,
        pad_token_id=0,
    )
    
    # Create model
    base_model = BertModel(config)
    
    # Load weights
    missing, unexpected = base_model.load_state_dict(bert_state_dict, strict=False)
    
    print(f"\n{'='*60}")
    if len(missing) <= 2 and len(unexpected) == 0:
        # Pooler weights are okay to be missing
        print("✅ Model loaded successfully!")
        if missing:
            print(f"   Note: {len(missing)} pooler weights will be randomly initialized (okay)")
    else:
        print(f"⚠️  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"   Missing keys: {missing[:5]}")
        if unexpected:
            print(f"   Unexpected keys: {unexpected[:5]}")
    print(f"{'='*60}")
    
    # Save in HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    base_model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    print(f"\n✅ Distilled model saved to: {output_dir}")
    
    # Calculate size
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"📊 Parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"🚀 Ready for fine-tuning!")
    
    return output_dir


def is_distilled_model_ready(model_dir: str) -> bool:
    """
    Check if a distilled model directory is ready to use
    
    Args:
        model_dir: Path to model directory
    
    Returns:
        bool: True if model is ready (has config.json and model files)
    """
    if not os.path.exists(model_dir):
        return False
    
    has_config = os.path.exists(os.path.join(model_dir, 'config.json'))
    has_model = (
        os.path.exists(os.path.join(model_dir, 'model.safetensors')) or
        os.path.exists(os.path.join(model_dir, 'pytorch_model.bin'))
    )
    
    return has_config and has_model

