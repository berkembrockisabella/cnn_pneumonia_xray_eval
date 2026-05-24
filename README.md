# Evaluation of CNN Architectures for Pneumonia Classification in Chest X-Ray Images

Applied deep learning project for binary classification of **pneumonia** in chest X-ray images, comparing ResNet, DenseNet, and EfficientNet architectures.


---

## Motivation

Pneumonia remains a major public health concern, demanding rapid and precise diagnosis for effective treatment and prevent complications. Deep learning-based diagnostic support systems have shown strong potential for accelerating and standardizing clinical screening.
This project evaluates well-established CNN architectures — **ResNet50V2**, **DenseNet**, and **EfficientNet** — comparing their performance on binary classification (NORMAL × PNEUMONIA) under four experimental conditions: no augmentation (baseline), with augmentation, with histogram equalization, and with adaptive equalization (CLAHE).

---

## Dataset

Only considering NORMAL and PNEUMONIA categories.

1. Chest X-Ray Images (Pneumonia)
- **Source:** *Chest X-Ray Images (Pneumonia)* (Kaggle)
- **Size:** **5,863 images**
- **Access:** <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

2. Chest X-Ray 8
- **Source:** *ChestX-ray8* — National Institute of Health (NIH)
- **Size:** **62,353**
- **Access:** <https://arxiv.org/abs/1705.02315>

3. COVID-19 Image Data Collection
- **Source:** *COVID-19 image data collection* (IEEE)
- **Size:** **764 images**
- **Access:** <https://github.com/ieee8023/covid-chestxray-dataset>

> **Balancing note:** the NORMAL class from ChestX-ray8 (~60k images) was capped at 5,000 samples to avoid skewing the unified dataset. Residual class imbalance is handled via `class_weight` during training.

---

## Repository structure

```
.
├── notebooks/
│   └── preprocessing.ipynb   # Exploratory Data Analysis and Preprocessing visualization
├── outputs/                  # Experiment outputs (CSV results, analysis tables)
│   └── graphs/               # Distributions, augmentation, preprocessing examples
│   └── logs/                 # Result records and metrics
│   └── models/               # Final trained models
├── src/
│   └── source code/
│       ├── main.py           # Experiment orchestration
│       ├── preprocessing.py  # Data pipeline (DataPipeline)
│       └── training.py       # Training, validation and fold evaluation
└── requirements.txt          # Project requirements
```
 
> The dataset is not committed to the repository.

---

## Methodology
 
**Data splitting**
- 15% reserved as a fixed holdout test set (separated before any fold)
- Remaining 85% split via `StratifiedKFold` with K=5
- Per fold: ~68% train | ~17% validation | 15% test

**Experiments**
 
| Experiment | Augmentation | Equalization |
|---|---|---|
| Baseline | No | No |
| Preprocessed | Yes | No |
| Hist. Equalization | Yes | Global histogram |
| CLAHE | Yes | Adaptive (CLAHE) |
 
**Augmentation applied** (when active): ±15° rotation, ±5% horizontal/vertical shift, ±10% zoom, horizontal flip.
 
---

## Credits

Authors: Ana Flávia Martins Dos Santos; Isabella Vanderlinde Berkembrock; Michele Cristina Otta; Yejin Chung
Affiliation: PUCPR — Pontifícia Universidade Católica do Paraná (Curitiba, Brazil)