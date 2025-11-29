# Geneformer Fine-tuning for Cardiomyopathy Classification

Fine-tuning [Geneformer](https://huggingface.co/ctheodoris/Geneformer) V1 model for classifying cardiomyocyte subtypes in heart disease.

## Task

**Downstream Classification Task:** Distinguish between three cardiomyopathy conditions:
- **Non-Failing (NF)** - Healthy heart tissue
- **Hypertrophic Cardiomyopathy (HCM)** - Heart muscle thickening
- **Dilated Cardiomyopathy (DCM)** - Heart chamber enlargement

Using single-cell RNA-seq data from human cardiomyocytes.

## Dataset

**Source:** [Genecorpus-30M Example Dataset](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset)

- **Format:** Pre-tokenized Arrow file (938 MB)
- **Vocabulary:** Geneformer V1 (~20k gene tokens)
- **Ready to use:** No additional tokenization needed

Download:
```bash
# Using Hugging Face CLI
huggingface-cli download ctheodoris/Genecorpus-30M \
  --repo-type dataset \
  --include "example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/*" \
  --local-dir ./data
```

Or download manually from the [HuggingFace dataset page](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset).

## Setup

```bash
# Clone repository
git clone https://github.com/AnshulSaini17/Geneformer_finetuning.git
cd Geneformer_finetuning

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
bash setup.sh
```

**Requirements:**
- Python 3.10+
- CUDA-capable GPU recommended (but works on CPU)
- 16GB+ RAM

## Usage

### 1. Configure Training

Edit `configs/config.yaml`:

```yaml
data:
  dataset_file: "path/to/cell_type_train_data.dataset"
  cell_types:
    - "Cardiomyocyte1"  # or your specific labels
    - "Cardiomyocyte2"
    - "Cardiomyocyte3"
  max_cells: 50000

training:
  num_epochs: 3
  learning_rate: 5e-5
  batch_size: 16  # Adjust based on GPU
```

### 2. Run Training

```bash
# Basic training
python src/main.py

# With evaluation and detailed logs
python src/main.py --evaluate --verbose

# Custom configuration
python src/main.py --config configs/my_config.yaml --data /path/to/dataset
```

### 3. Command Line Options

```
--config PATH       Configuration file (default: configs/config.yaml)
--data PATH         Dataset path (overrides config)
--output-dir PATH   Output directory (default: timestamped)
--evaluate          Run evaluation after training
--verbose           Show detailed logs
--skip-prepare      Skip data preparation (use existing)
```

## Output

Training results are saved to `outputs/<timestamp>/`:

```
outputs/20251128_120000/
├── ksplit1/                              # Trained model checkpoint
│   ├── pytorch_model.bin
│   ├── config.json
│   └── training_args.bin
├── cardiomyocyte_classifier_conf_mat.png # Confusion matrix
├── cardiomyocyte_classifier_predictions_*.png
└── cardiomyocyte_classifier_id_class_dict.pkl
```

## Project Structure

```
Geneformer_finetuning/
├── src/
│   ├── main.py                    # Main training script
│   ├── data/
│   │   └── dataset_loader.py      # Data loading utilities
│   ├── models/
│   │   ├── classifier.py          # Model initialization
│   │   └── model_patch.py         # V1 model loading patch
│   ├── training/
│   │   └── trainer.py             # Training pipeline
│   └── evaluation/
│       └── evaluator.py           # Evaluation & visualization
├── configs/
│   └── config.yaml                # Training configuration
├── notebooks/
│   └── demo.ipynb                 # Example notebook
└── requirements.txt
```

## Configuration

Key parameters in `configs/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.version` | `V1` | Geneformer version (V1: 10M params) |
| `data.max_cells` | `50000` | Maximum cells for training |
| `training.num_epochs` | `3` | Training epochs |
| `training.batch_size` | `16` | Per-device batch size |
| `training.learning_rate` | `5e-5` | Learning rate |
| `training.bf16` | `true` | Use BF16 precision (GPU) |

## Model Details

- **Base Model:** [Geneformer V1-10M](https://huggingface.co/ctheodoris/Geneformer)
- **Architecture:** BERT-based transformer for gene expression
- **Vocabulary Size:** ~20,000 gene tokens
- **Max Sequence Length:** 2,048 tokens
- **Pre-training:** 30M single-cell transcriptomes

## Training Time

| GPU | Batch Size | Time (50k cells, 3 epochs) |
|-----|------------|----------------------------|
| T4  | 16         | ~30-45 min                 |
| V100| 32         | ~20-30 min                 |
| A100| 64         | ~10-15 min                 |

## Key Features

✅ **Smart Model Loading** - Automatically handles HuggingFace V1 subfolder (local paths work too)  
✅ **Configuration-based** - Easy parameter tuning via YAML  
✅ **Modular Code** - Clean, reusable components  
✅ **Works Everywhere** - Local GPU, Colab, cloud platforms  
✅ **Verified** - Tested against working Colab notebook  
✅ **Documentation** - Complete guides and examples

## GPU & Compute Options

This code works on:
- **Local GPU** (NVIDIA with CUDA)
- **Institution/Lab GPU** servers
- **Cloud platforms** (AWS, Azure, GCP)
- **Google Colab** (if you need free GPU access)
- **CPU** (slower but functional)

### Google Colab Setup

If you don't have a GPU, use Google Colab:

```python
!git clone https://github.com/AnshulSaini17/Geneformer_finetuning.git
%cd Geneformer_finetuning
!bash setup.sh
```

## Troubleshooting

### Out of Memory

Reduce batch size in `configs/config.yaml`:

```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 2
```

### CUDA Not Available

Code automatically uses CPU if no GPU found (slower but works).

### Data Loading Issues

Ensure dataset path is correct:
```bash
ls -lh path/to/cell_type_train_data.dataset/
# Should show: dataset.arrow, dataset_info.json, state.json
```

## Citation

If you use Geneformer in your research, please cite:

```bibtex
@article{theodoris2023transfer,
  title={Transfer learning enables predictions in network biology},
  author={Theodoris, Christina V and Xiao, Ling and Chopra, Anant and 
          Chaffin, Mark D and Al Sayed, Zeina R and Hill, Matthew C and 
          Mantineo, Helene and Brydon, Elizabeth M and Zeng, Zexian and 
          Liu, X Shirley and others},
  journal={Nature},
  volume={618},
  number={7965},
  pages={616--624},
  year={2023},
  publisher={Nature Publishing Group}
}
```

## Related Work

- **Geneformer Paper:** [Transfer learning enables predictions in network biology](https://www.nature.com/articles/s41586-023-06139-9)
- **Model on HuggingFace:** [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- **Dataset:** [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M)

## License

MIT License - See LICENSE file for details.

## Contributing

Issues and pull requests are welcome! See [DATA_GUIDE.md](DATA_GUIDE.md) for information on using custom datasets.

---

**Note:** This repository contains only code. The dataset must be downloaded separately from HuggingFace (see Dataset section above).
