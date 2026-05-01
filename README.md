Early Fake News Detection on Social Media
---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-412991?style=for-the-badge&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**A multimodal deep learning system for fake news detection using CLIP encoders and Label-Preserving Contrastive Learning (LPCL).**

*Rushil Dharwal (BITS Pilani)*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Components](#model-components)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

This project implements a **multimodal fake news detection system** that jointly analyses the textual headline and associated image of a news item. It is the second and final phase of a two-stage research project:

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Text-only baseline using CLIP text encoder + linear classifier | ✅ Complete |
| **Phase 2** | Full multimodal architecture with NLI-style cross-modal fusion + LPCL | ✅ Complete |
| **Phase 3** | Interactive GUI application (single & batch classification) | ✅ Complete |

### Motivation

Manual fact-checking cannot scale to the volume of content generated on social media. Text-only detectors miss a critical pattern: **cross-modal inconsistency**, where an image contradicts or misrepresents the accompanying text. This system addresses that gap by leveraging CLIP's shared embedding space to detect semantic misalignment between modalities.

---

## Architecture

```
News Headline ──► CLIP Text Encoder ──► L2 Norm ──────────────────────┐
                                                                        ▼
                                                              NLI Fusion Layer ──► Concat [ft; fv; z; s] ──► MLP Classifier ──► REAL / FAKE
                                                                        ▲                     ▲
News Image ────► CLIP Vision Encoder ──► L2 Norm ─────┬───────────────┘                     │
                                                        │                                     │
                                                        └──► Cosine Similarity ───────────────┘
```

### NLI-Style Fusion

Inspired by Natural Language Inference models, the fusion layer explicitly models both shared content and modal divergence:

| Signal | Formula | Captures |
|--------|---------|---------|
| Element-wise interaction | `i = ft ⊙ fv` | What is shared between modalities |
| Absolute difference | `d = \|ft − fv\|` | What differs between modalities |
| Projection | `z = GELU(Wf · [ft; fv; i; d] + bf)` | Compact 512-D fused representation |

The final classifier input is a **1537-dimensional** vector:

```
c = [ft ; fv ; z ; s]  ∈ ℝ^(512 + 512 + 512 + 1)
```

where `s` is the scalar cosine similarity score — an explicit, interpretable cross-modal alignment signal.

---

## Key Features

- **Dual CLIP Encoders** — Shared ViT-B/32 backbone pre-trained on 400M image-text pairs
- **NLI-Style Fusion** — Cross-modal interaction and difference signals concatenated into 2048-D joint representation
- **LPCL Contrastive Loss** — In-batch InfoNCE alignment loss replacing unsupervised GMM refinement
- **Multi-Layer GELU Classifier** — `1537 → 512 → 128 → 2` with LayerNorm and Dropout
- **Confidence Thresholding** — Predictions below 0.65 confidence are routed to expert review
- **Interactive GUI** — CustomTkinter app with single-item and batch analysis modes
- **Batch Reporting** — Export results as PDF, PNG, or JSON

---

## Dataset

This project uses the **[Fakeddit(Multimodal Only)](https://www.kaggle.com/code/vanshikavmittal/fakeddit-multimodal-fake-news-classification)** i.e. modified version of (https://github.com/entitize/fakeddit) dataset (Nakamura et al., 2020) — a large-scale multimodal benchmark collected from Reddit.

| Split | Total Samples | Used |
|-------|--------------|------|
| Training | ~564,212 | 100,000 |
| Validation | ~59,365 | 30,000 |
| Test | ~59,339 | 500 |

**Primary fields used:** `clean_title` (text input), `2_way_label` (0 = Real, 1 = Fake), `id`, `image_url`, `domain`

Images are downloaded separately using the provided URLs and stored as `{id}.jpg` files. The dataset class filters to only rows where the corresponding image is available on disk.

> **Why Fakeddit?** Unlike LIAR (text-only), PHEME (Twitter API-dependent), or FakeNewsNet (API-restricted), Fakeddit is large-scale, publicly accessible, multimodal, and supports binary classification out of the box.
Recent ones are not as big as fakeddit dataset

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- ~10 GB disk space for dataset images

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
customtkinter
matplotlib
numpy
open_clip_torch
openpyxl
pandas
Pillow
python-docx
requests
scikit-learn
torch
tqdm
transformers
```

---

## Usage

### 1. Download Dataset Images

```bash
python data/download_images.py
```

### 2. Train the Model

```bash
python train_model.py
```

### 3. Evaluate

```bash
python evaluate.py
```

### 4. Run the GUI Application

```bash
python interface/app.py
```

### 5. Batch Analysis via JSON

```bash
# Sample JSON format
[
    {
        "id": "sampleid1",
        "text": "sample headline",
        "image_url": "https://sample.com".
        "predicted": 1,
        "predicted_label": "Fake",
        "true": 1,
        "true_label": "Fake",
        "confidence": 0.9586,
        "expert_review": false
    },
]
```

Load this file in the GUI's batch analysis tab, or process programmatically:

```python
from interface.app import FakeNewsApp
# See interface/sample_jsons/ for examples
```

---

## Model Components

### Feature Extraction (`models/image_encoder.py`, `models/text_encoder.py`)

```python
# Both encoders share the CLIP ViT-B/32 backbone
text_features  = F.normalize(self.text_encoder(text_tokens),  dim=1)  # → ℝ^512
image_features = F.normalize(self.image_encoder(images),      dim=1)  # → ℝ^512
```

L2 normalization ensures dot products equal cosine similarity, stabilizing downstream operations.

### NLI Fusion Layer (`models/multimodal_model.py`)

```python
self.fusion_layer = nn.Sequential(
    nn.Linear(512 * 4, 512),
    nn.GELU(),
    nn.Dropout(0.4)
)

interaction = text_features * image_features                         # Element-wise product
difference  = torch.abs(text_features - image_features)             # Absolute difference
fusion = self.fusion_layer(
    torch.cat([text_features, image_features, interaction, difference], dim=1)
)
```

### Multi-Layer Classifier (`models/classifier.py`)

```python
self.net = nn.Sequential(
    nn.Linear(1537, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.5),
    nn.Linear(512,  128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.5),
    nn.Linear(128,    2)
)
```

GELU activation is preferred over ReLU for smoother gradients in transformer-based models:

```
GELU(x) = x · Φ(x) ≈ 0.5x · (1 + tanh(√(2/π)(x + 0.044715x³)))
```

### LPCL Loss Function (`training/trainer.py`)

Label-Preserving Contrastive Learning replaces unsupervised GMM refinement with a supervised in-batch InfoNCE loss:

```python
def in_batch_lpcl_loss(text_feat, image_feat, temperature):
    logits = (text_feat @ image_feat.T) * temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2

# Combined loss in training loop
cls_loss   = criterion(logits, labels)        # CrossEntropyLoss(label_smoothing=0.1)
align_loss = in_batch_lpcl_loss(text_feat, image_feat, temperature)
loss       = (cls_loss + 0.1 * align_loss) / accumulation_steps
```

The temperature `τ` is a learnable parameter clamped for numerical stability:

```python
self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
temperature = torch.clamp(self.logit_scale.exp(), min=1e-3, max=100)
```

---

## Training

### Configuration (Advanced)

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Encoder learning rate | 1 × 10⁻⁶ |
| Classifier head learning rate | 1 × 10⁻⁵ |
| Weight decay | 0.05 |
| Scheduler | OneCycleLR (cosine, 10% warmup) |
| Batch size | 6 |
| Gradient accumulation steps | 4 |
| Effective batch size | 24 |
| Precision | bfloat16 / float16 (AMP) |
| Early stopping patience | 3 epochs |
| Label smoothing | ε = 0.1 |
| LPCL weight | 0.1 |
| Dropout | 0.4 (fusion), 0.5 (classifier) |

### Training Progression

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|---------|-------------|
| 1 | ~0.5591 | ~0.4006 | ~87.66% |
| 2 | ~0.4005 | ~0.3936 | **~88.62%** ← best checkpoint |
| 3 | ~0.3474 | ~0.4048 | ~88.59%|

The best checkpoint is saved as `multimodal_model.pt`.

---

## Results

### Evaluation Metrics (500 Test Samples)

| Metric | Value |
|--------|-------|
| Accuracy | **~89.00%** |
| Precision | ~87.72% |
| Recall | ~92.59% |
| F1-Score | ~90.09% |
| Expert Review Ratio | ~0.0520% |

### Text-Only vs. Multimodal Comparison

| Feature | Text-Only (Phase 1) | Multimodal (Phase 2) |
|---------|--------------------|--------------------|
| Modalities | Text only | Text + Image |
| Classifier | `Linear 512→2 (ReLU)` | `MLP 1537→512→128→2 (GELU)` |
| Loss | CrossEntropy | CrossEntropy + LPCL |
| Cross-modal similarity | — | Cosine similarity |
| Val. Accuracy | ~87% | ~89% |

> **Key insight:** The primary gain of the multimodal system is not raw accuracy but the ability to detect **image-text inconsistency** — a structural capability that text-only models fundamentally cannot provide.

### Error Analysis

**False Negatives** (fake → predicted real): Most common when the image is neutral or decorative, leaving only text as the discriminative signal.

**False Positives** (real → predicted fake): Occur with sensationalist but genuine headlines paired with visually atypical images.

**Expert Review** (~12%): Cases where confidence falls below 0.65 are flagged rather than blindly classified — a practical safety mechanism for real-world deployment.

---

## Project Structure

```
fake-news-detection/
|-- data/
|   |-- dataset_loader.py
|   |-- download_images.py
|   `-- preprocessing.py
|-- Fakeddit/
|   |-- images/
|   `-- multimodal_*.tsv
|-- interface/
|   |-- _temp_images/
|   |-- logs/
|   |-- sample_jsons/
|   |-- app.py
|   `-- history.json
|-- models/
|   |-- classifier.py
|   |-- image_encoder.py
|   |-- multimodal_model.py
|   `-- text_encoder.py
|-- training/
|   `-- trainer.py
|--train_model.py
|-- evaluate.py
|-- multimodal_model.pt
`-- requirements.txt
└── README.md
```
---

## Future Work

- [ ] Replace Fakeddit with a more recent multimodal fake news dataset
- [ ] Web deployment via Flask or FastAPI
- [ ] Complete the procedure for routing to the expert
- [ ] Cross-attention fusion to replace NLI concatenation with a more expressive inter-modal interaction mechanism
- [ ] Adversarial robustness evaluation

---

## References

1. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2020). FakeNewsNet: A data repository with news content, social context, and spatiotemporal information for studying fake news on social media. *Big Data, 8(3)*, 171–188.

2. Chen, X., Wu, J., & Wang, Y. (2022). Propagation structure-aware graph transformer for robust and interoperable fake news detection. *AAAI Conference on Artificial Intelligence.*

3. Xu, M., Li, F., Miao, Z., Han, Z., Wang, L., & Wang, G. (2025). Detecting fake news on social media via multimodal semantic understanding and enhanced transformer architectures.

4. Nakamura, K., Levy, S., & Wang, W. Y. (2020). r/Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection. *ICWSM.*

5. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML 2021.*

6. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017.*

7. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT.*

---

<div align="center">

Made with ❤️ at **BITS Pilani** · April 2026

</div>
