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
- CUDA-capable GPU recommended (works on CPU but much slower)
- 16GB+ RAM

## Usage

### Option A: Fine-tune with Distilled/Pre-trained Model

**Have a distilled model from a collaborator?** Use it directly:

```bash
python src/main.py --distilled-model path/to/model_best.pt --evaluate
```

See [DISTILLED_PIPELINE_GUIDE.md](DISTILLED_PIPELINE_GUIDE.md) for details.

### Option B: Fine-tune Original Geneformer

### 1. Configure Training

Edit `configs/config.yaml`:

```yaml
data:
  dataset_file: "path/to/cell_type_train_data.dataset"  # Path to downloaded dataset
  cell_state_key: "disease"                    # Column name for labels
  cell_types:                                    # Cardiomyopathy classifications
    - "nf"              # NF - Healthy
    - "hcm"  # HCM
    - "dcm"       # DCM
  max_cells: None               # Max number of cells to use
  train_test_split: 0.2          # Test set proportion

# Training hyperparameters
training:
  num_epochs: 0.9                
  learning_rate: 0.000804        
  batch_size: 12    # Adjust based on GPU
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
--config PATH           Configuration file (default: configs/config.yaml)
--data PATH             Dataset path (overrides config)
--distilled-model PATH  Path to distilled .pt model (e.g., model_best.pt)
--output-dir PATH       Output directory (default: timestamped)
--evaluate              Run evaluation after training
--verbose               Show detailed logs
--skip-prepare          Skip data preparation (use existing)
```

## Output

Training results are saved to `outputs/<timestamp>/`:

### Original Geneformer:
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

### Distilled Model:
```
outputs/20251128_120000/
├── distilled_geneformer/                         # Converted model
├── cardiomyocyte_classifier_distilled_*.dataset  # Training data (with _distilled suffix)
├── cardiomyocyte_classifier_distilled_conf_mat.png
└── cardiomyocyte_classifier_distilled_*.pkl
```

**Note:** Distilled model results use `_distilled` suffix to prevent overwriting original results.

## Model Details

- **Base Model:** [Geneformer V1-10M](https://huggingface.co/ctheodoris/Geneformer)
- **Architecture:** BERT-based transformer for gene expression
- **Vocabulary Size:** ~20,000 gene tokens
- **Max Sequence Length:** 2,048 tokens
- **Pre-training:** 30M single-cell transcriptomes

## Training Time

Automatically uses GPU if available (CPU fallback):

| GPU | Batch Size | Time (50k cells, 3 epochs) |
|-----|------------|----------------------------|
| A100| 32-64      | ~10-15 min                 |
| V100| 32         | ~20-30 min                 |
| T4  | 16         | ~30-45 min                 |
| CPU | 4-8        | ~3-4 hours                 |

**No GPU?** Use Google Colab (see section below).

## Key Features

✅ **Smart Model Loading** - Automatically handles HuggingFace V1 subfolder (local paths work too)  
✅ **Configuration-based** - Easy parameter tuning via YAML  
✅ **Modular Code** - Clean, reusable components  
✅ **Works Everywhere** - Local GPU, Colab, cloud platforms  
✅ **Verified** - Tested against working Colab notebook  
✅ **Documentation** - Complete guides and examples

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


