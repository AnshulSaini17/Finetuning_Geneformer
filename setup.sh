#!/bin/bash
# Setup script for Geneformer fine-tuning

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Installing Geneformer..."
pip install --no-deps git+https://huggingface.co/ctheodoris/Geneformer

echo ""
echo "✓ Installation complete!"
echo ""
echo "To verify:"
echo "  python -c 'from geneformer import Classifier; print(\"✓ Geneformer ready!\")'"

