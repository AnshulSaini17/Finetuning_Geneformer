# Data Preparation Guide

## 📊 Cardiomyopathy Classification Dataset

### Recommended Dataset (Pre-tokenized)

**Source:** [Genecorpus-30M Cell Classification Example](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset)

- **Task:** Cardiomyocyte subtype classification
- **Size:** 938 MB
- **Format:** Arrow dataset (ready to use)
- **Vocabulary:** Geneformer V1 (~20k gene tokens)
- **Status:** Already tokenized - no preprocessing needed

### Download Instructions

**Method 1: Hugging Face CLI (Recommended)**

```bash
# Install HF CLI
pip install huggingface-hub[cli]

# Download dataset
huggingface-cli download ctheodoris/Genecorpus-30M \
  --repo-type dataset \
  --include "example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/*" \
  --local-dir ./data

# The dataset will be in:
# ./data/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/
```

**Method 2: Manual Download**

1. Visit: https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset
2. Download all three files:
   - `dataset.arrow` (938 MB)
   - `dataset_info.json`
   - `state.json`
3. Place in a folder: `cell_type_train_data.dataset/`

**Method 3: Python Script**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ctheodoris/Genecorpus-30M",
    repo_type="dataset",
    allow_patterns="example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset/*",
    local_dir="./data"
)
```

### Update Configuration

After downloading, update `configs/config.yaml`:

```yaml
data:
  dataset_file: "data/example_input_files/cell_classification/cell_type_annotation/cell_type_train_data.dataset"
  cell_types:
    - "Cardiomyocyte1"  # Update based on your labels
    - "Cardiomyocyte2"
    - "Cardiomyocyte3"
```

---

## 🔧 Using Your Own Data (Advanced)

If you have your own single-cell RNA-seq data:

### Requirements

Your dataset should include:
- **Gene expression data** (.loom, .h5ad, or similar)
- **Cell type annotations** for classification
- Single-cell resolution

### Tokenization Process

```python
from geneformer import TranscriptomeTokenizer

# Initialize tokenizer
tk = TranscriptomeTokenizer(
    custom_attr_name_dict={"cell_type": "cell_type"},
    nproc=4
)

# Tokenize your data
tk.tokenize_data(
    data_directory="path/to/your/data/",
    output_directory="output/",
    output_prefix="my_dataset",
    file_format="h5ad"  # or "loom"
)
```

**Output:** Tokenized Arrow dataset compatible with this codebase.

**See:** [Geneformer Tokenization Guide](https://huggingface.co/ctheodoris/Geneformer/blob/main/examples/tokenizing_scRNAseq_data.ipynb)

---

## 📋 Dataset Format

### Required Structure

```python
from datasets import load_from_disk

dataset = load_from_disk("cell_type_train_data.dataset")

# Example entry:
{
    'input_ids': [15234, 8765, 2341, ...],  # Gene token IDs (List[int])
    'length': 2048,                          # Sequence length (int)
    'cell_type': 'Cardiomyocyte1',          # Label (str)
    # Optional metadata:
    'disease': 'DCM',
    'individual': 'patient_001',
    ...
}
```

### Vocabulary

- **V1 Tokenizer:** ~20,000 gene tokens
- **Coverage:** Most expressed human genes
- **Pre-trained:** On 30M single-cell transcriptomes

---

## 🧪 Test Data (For Code Testing Only)

Create a small dummy dataset to test the code:

```python
from datasets import Dataset
import random

# Minimal test dataset
dummy_data = {
    'input_ids': [[random.randint(0, 20000) for _ in range(2048)] for _ in range(1000)],
    'length': [2048] * 1000,
    'cell_type': random.choices(['Type1', 'Type2', 'Type3'], k=1000),
}

dataset = Dataset.from_dict(dummy_data)
dataset.save_to_disk("test_dataset")
```

**Note:** Won't train a useful model but validates the pipeline works.

---

## 📚 Additional Resources

- **Geneformer Paper:** [Nature 2023](https://www.nature.com/articles/s41586-023-06139-9)
- **HuggingFace Model:** [ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- **Dataset Repository:** [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M)
- **Tokenization Examples:** [Geneformer Examples](https://huggingface.co/ctheodoris/Geneformer/tree/main/examples)

---

## ❓ FAQ

**Q: Why V1 and not V2?**  
A: V1 (10M params, 20k vocab) is sufficient for most fine-tuning tasks and faster to train than V2 (104M/316M params).

**Q: Can I use other cell types?**  
A: Yes! Update `cell_types` in config.yaml with your specific labels.

**Q: How much data do I need?**  
A: For fine-tuning: 10k-100k+ cells recommended. More data generally improves results.

**Q: What about data privacy?**  
A: Never upload patient data to public repos. Use public datasets or follow institutional data policies.

---

**Need help?** Open an issue on [GitHub](https://github.com/AnshulSaini17/Geneformer_finetuning/issues) with details about your data format!
