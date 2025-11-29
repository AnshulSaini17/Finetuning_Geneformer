"""
Model loading patch for Geneformer V1 from HuggingFace Hub

This patch is ONLY needed when loading from HuggingFace:
  model_directory = "ctheodoris/Geneformer"

NOT needed when using local paths:
  model_directory = "/path/to/Geneformer-V1-10M"

The patch checks the model_directory and only activates for HuggingFace Hub URLs.
"""

import geneformer.perturber_utils as pu
from transformers import BertForSequenceClassification
import torch


# Store original function
_original_load_model = pu.load_model
_patch_applied = False


def patched_load_model(model_type, num_classes, model_directory, mode, quantize=False):
    """
    Patched version of load_model that loads from correct V1-10M subfolder
    
    This fixes the issue where Geneformer tries to load from wrong location
    on HuggingFace Hub. Only activates for "ctheodoris/Geneformer", not local paths.
    """
    if model_directory == "ctheodoris/Geneformer" and model_type == "CellClassifier":
        # Only patch HuggingFace Hub loading, not local paths
        print(f"Loading V1-10M model from subfolder...")
        
        model = BertForSequenceClassification.from_pretrained(
            model_directory,
            subfolder="Geneformer-V1-10M",
            num_labels=num_classes,
            output_hidden_states=(mode == "eval"),
            output_attentions=False,
        )
        
        if mode == "eval":
            model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    else:
        # Use original function for other cases
        return _original_load_model(model_type, num_classes, model_directory, mode, quantize)


def apply_model_patch():
    """Apply the model loading patch"""
    global _patch_applied
    
    if not _patch_applied:
        pu.load_model = patched_load_model
        _patch_applied = True
        print("✓ Model loading patch applied (V1-10M subfolder)")
    

def restore_original():
    """Restore original model loading function"""
    global _patch_applied
    
    if _patch_applied:
        pu.load_model = _original_load_model
        _patch_applied = False
        print("✓ Original model loading restored")

