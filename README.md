# Detecting Lung Cancer via CT Image Analysis using Deep Learning Models

MSc Dissertation Project, Rushikesh Kothawade, University of Leeds

A single Jupyter notebook (`Detecting_Lung_Cancer_via_CT_Image_Analysis_using_Deep_Learning_Models_code.ipynb`, 112 cells, 16 markdown, 96 code) implementing the full experimental pipeline: five deep learning architectures, a soft-voting ensemble, Grad-CAM explainability, Monte Carlo Dropout uncertainty quantification, and cross-dataset generalisation evaluation.

---

## 1. Pipeline at a glance

| Stage | What happens | Key output |
|---|---|---|
| Run setup | Stamps every artifact with a unique run ID, mounts Google Drive | `RESULTS_DIR`, `manifest` dict |
| Data acquisition | Downloads both datasets from Kaggle via API | `PRIMARY_DIR`, `SECONDARY_DIR` populated |
| EDA | Mean intensity, per-class entropy, PCA variance on primary training split | `eda_intensity_entropy.png` |
| Preprocessing | Resize (224×224 / 300×300), ImageNet normalisation, augmentation (train split only) | `train_loader`, `valid_loader`, `test_loader` |
| SMOTE utility | Defined here, applied later on the secondary dataset | `apply_smote_to_image_folder()` |
| Model training | 5 architectures, per-model hyperparameters, timed logging | 5 checkpoints (`<model>_best.pt`) |
| Evaluation | Accuracy, precision/recall/F1, confusion matrix, ROC-AUC per model + ensemble | `test_results.json`, `per_class_metrics.csv` |
| McNemar's testing | All-pairs statistical significance testing | `mcnemar_all_pairs.csv` |
| Grad-CAM | Saliency maps for the best individual model | `gradcam/` overlays, entropy stats |
| MC Dropout | T=50 stochastic passes, uncertainty-error correlation, triage threshold | `uncertainty_quartile_table.csv`, `mc_dropout_triage_summary.json` |
| Cross-dataset validation | Zero-shot transfer + SMOTE fine-tune on IQ-OTH/NCCD | `table6_cross_dataset_results.csv` |
| Final manifest | Consolidates all run metadata | `manifest.json` |

---

## 2. Notebook structure

Section headers as they actually appear in the notebook, with their markdown cell index.

| Cell # | Section | Notes |
|---|---|---|
| 0 | Run identity | Stamps every artifact with a run ID |
| 2 | Title / overview | Project description |
| 3 | 1. Environment Setup | Seeding, device detection, package installs |
| 7 | 0b. Pin `RESULTS_DIR` to this run | Prevents different runs overwriting each other's outputs |
| 9 | 1. Library + hardware versions | Logs torch/torchvision/sklearn versions, GPU name |
| 11 | 2. Data Acquisition | Kaggle API download of both datasets |
| 17 | 3. Exploratory Data Analysis (EDA) | Reproduces dissertation Section 3.1.4 |
| 22 | 4. Preprocessing Pipeline | Reproduces dissertation Section 3.2.1 |
| 27 | 2. Primary dataset class counts | Train/valid/test class distribution logging |
| 31 | 5. SMOTE Utility | Defined here, applied later in cross-dataset chapter |
| 40 | 3. Parameter counts per architecture | Logs total/trainable params for all 5 models |
| 46 | 4/5. Timed training + best-epoch logging wrapper | `timed_train()` wraps `train_model()` for all architectures |
| 56 | 6. Per-class metrics + confusion matrices | Evaluation on held-out test set |
| 72 | 7. All-pairs McNemar's test | Statistical significance across every model pair |
| 99 | 8. Zero-shot sanity check | Class distribution + confusion matrix, cross-dataset |
| 110 | 9. Final run manifest | Consolidated run metadata export |

Grad-CAM (`class GradCAM`, code cell 62) and Monte Carlo Dropout (`enable_mc_dropout`, code cell 69) are both implemented between the Section 6 and Section 7 markdown headers (cells 56–72) but are not marked with their own top-level markdown headers in this notebook.

---

## 3. Models compared

| Model | Parameters | Pretrained | Fine-tuning strategy | Input size |
|---|---|---|---|---|
| Custom CNN | 410,308 | No (from scratch) | Single-phase, 50 epochs max | 224×224 |
| ResNet-50 | 23,516,228 | ImageNet | Two-phase: head only (5 ep) → full unfreeze (low LR) | 224×224 |
| VGG-16 | 138,459,972 | ImageNet | Single-phase, extended custom head | 224×224 |
| DenseNet-121 | 6,957,956 | ImageNet | Two-phase: head only (5 ep) → full unfreeze (low LR) | 224×224 |
| EfficientNet-B3 | 10,702,380 | ImageNet | Single-phase, label-smoothing loss | 300×300 |

Every model's optimiser, learning rate, scheduler, batch size, and regularisation setting is defined independently in `MODEL_REGISTRY` in the notebook, the single source of truth for all training hyperparameters.

---

## 4. Datasets

Both datasets are pulled from Kaggle at runtime and are **not** stored in this repository.

| Dataset | Role | Classes | Size | Source |
|---|---|---|---|---|
| Chest CT-Scan Images | Primary : train/validation/test | Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal | 1,000 images (613 / 72 / 315 split) | [Kaggle: mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) |
| IQ-OTH/NCCD Lung Cancer Dataset | Secondary : cross-dataset validation only | Normal, Benign, Malignant | 1,097 images | [Kaggle: hamdallak/the-iqothnccd-lung-cancer-dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) |


You will need a Kaggle API token (`kaggle.json`) see the [Kaggle API docs](https://www.kaggle.com/docs/api). The notebook prompts for this on first run.

---

## 5. Results

All figures below are taken directly from this notebook's saved execution output (run ID `20260816_115645`).

### 5.1 Individual models + ensemble (primary test set, n=315)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Custom CNN (baseline) | 46.35% | 0.5072 | 0.5375 | 0.4961 | 0.6794 |
| ResNet-50 | 62.86% | 0.7437 | 0.6041 | 0.6077 | 0.8570 |
| EfficientNet-B3 | 53.65% | 0.6509 | 0.5761 | 0.5194 | 0.8104 |
| VGG-16 | 80.95% | 0.8673 | 0.8630 | 0.8422 | 0.9860 |
| **DenseNet-121** | **84.76%** | 0.8680 | 0.8550 | **0.8597** | 0.9619 |
| **Ensemble** (VGG-16 + DenseNet-121 + ResNet-50) | **86.67%** | 0.8985 | 0.9033 | **0.8897** | **0.9859** |

DenseNet-121 is confirmed in the notebook's own output as the best individual model by test accuracy. The ensemble's +1.91pp gain over DenseNet-121 is **not statistically significant** under McNemar's test (χ²=0.4808, p=0.4881).

### 5.2 Cross-dataset generalisation (IQ-OTH/NCCD)

| Condition | Accuracy | Precision | Recall | F1 | Benign Recall |
|---|---|---|---|---|---|
| Ensemble, zero-shot (full secondary set) | 51.32% | 0.4052 | 0.4122 | 0.3528 | 0.0000 |
| DenseNet-121, zero-shot (full secondary set) | 38.92% | 0.3076 | 0.3334 | 0.2226 | 0.0000 |
| SMOTE + DenseNet-121, fine-tuned (held-out N=329) | 84.80% | 0.8212 | 0.7348 | 0.7620 | 0.4444 |

The dataset's majority-class baseline is 51.14%, zero-shot ensemble accuracy is statistically indistinguishable from majority-class guessing, and zero-shot DenseNet-121 falls below it. Benign recall is structurally 0.0 under zero-shot (the primary model has no Benign output class); SMOTE-based fine-tuning of the classification head alone recovers it to 0.4444, still the weakest of the three per-class recalls post-fine-tune.

### 5.3 MC Dropout triage (VGG-16, T=50, 90th-percentile threshold)

| Uncertainty Quartile | Entropy Range | % of Test Set | Accuracy | n |
|---|---|---|---|---|
| Q1 (Low) | [0.000, 0.010) | 25.08% | 100.00% | 79 |
| Q2 | [0.010, 0.276) | 24.76% | 96.15% | 78 |
| Q3 | [0.276, 0.688) | 25.08% | 74.68% | 79 |
| Q4 (High) | [0.688, 1.345] | 25.08% | 49.37% | 79 |

At the calibrated review threshold: **165/315 (52.4%)** of test cases flagged for review, error rate within flagged subset 36.4%, catching **60 of 63 total model errors (95.2%)**. Extrapolated to 1,000 scans: 476 auto-classified, 524 flagged (191 genuine errors caught, 333 correct-but-flagged).

### 5.4 Comparison with published benchmarks

| Study | Dataset / Task | Best Accuracy | AUC | Explainable | Uncertainty-Aware |
|---|---|---|---|---|---|
| Hamdalla F.Al-Yasriy (2020) | IQ-OTH/NCCD, 3-class | 93.54% | No | No | No |
| Al-Huseiny & Sajit (2021) | IQ-OTH/NCCD, 3-class | 94.38% | No | No | No |
| Ardila et al. (2019) | NLST, binary | No | 0.944 | No | No |
| **This study : Ensemble** | Chest CT, 4-class + cross-dataset | **86.67%** | **0.9859** | **Yes** | **Yes** |

This comparison table is generated directly within the notebook and notes that the two published IQ-OTH/NCCD studies have no confirmed patient-level split, a leakage risk this dissertation discusses in Chapter 2.

---

## 6. Environment & requirements

Developed and executed on **Google Colab**, using a **Tesla T4 GPU** runtime with Google Drive mounted for persistent storage across sessions.

| Package | Version confirmed in notebook output |
|---|---|
| Python | Colab default |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| scikit-learn | 1.6.1 |
| imbalanced-learn | latest via pip |
| statsmodels | latest via pip |
| opencv-python | Colab default |
| pandas / numpy / matplotlib | Colab default |
| kaggle | latest via pip |

Install the packages not pre-installed in Colab:

```bash
pip install -q imbalanced-learn kaggle statsmodels
```

To run outside Colab: set `USE_DRIVE = False` in the Section 1 setup cell to use a local working directory instead of a mounted Drive path, and ensure a CUDA-capable GPU is available, training five architectures plus MC Dropout's 50x inference cost is not practical on CPU alone.

---

## 7. Reproducibility

- Cell 0 stamps every run with a unique `RUN_ID` (timestamp-based); all figures, tables, checkpoints, and JSON summaries for that run are written under `results/<RUN_ID>/`, so different executions never overwrite one another. The results reported above are from run `20260816_115645`.
- A `manifest` dictionary (finalised in the Section 9 cell) accumulates library versions, GPU info, dataset class counts, per-model parameter counts, and convergence stats throughout the run.
- Random seed fixed at `SEED = 42` across Python's `random`, NumPy, and PyTorch (including CUDA), with `torch.backends.cudnn.deterministic = True`. Exact bit-for-bit reproducibility across different GPU types is not guaranteed.

---

## 8. Known limitations

| Limitation | Detail |
|---|---|
| Case-grouping fallback | The patient/case-grouping step for the IQ-OTH/NCCD adaptation/test split cannot parse this dataset's filename convention and falls back to one group per image the fine-tuning split (Section 5.2 above) is image-level, not confirmed patient-level |
| SMOTE on raw pixels | Interpolation happens on flattened pixel vectors (~150k dimensions), not a learned feature embedding acknowledged directly in the notebook's own Section 5 markdown as "a defensible baseline, not a strong one" |
| MC Dropout threshold calibration | The 90th-percentile review threshold is calibrated on the training set, not a held-out validation split |
| Retrospective evaluation only | All results are on public benchmark datasets with fixed labels; no prospective clinical validation |

Full discussion of each limitation is in the accompanying dissertation (Chapters 3, 5, and 6).

---

## 9. Citation

Kothawade, R. (2026) Detecting Lung Cancer via CT Image Analysis using Deep Learning Models. MSc Dissertation, University of Leeds.

## License

Code is made available for academic and research purposes. Dataset licenses are governed by their respective Kaggle sources (Section 4).

