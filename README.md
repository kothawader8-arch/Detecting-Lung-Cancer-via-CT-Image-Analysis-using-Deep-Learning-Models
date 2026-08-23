# Detecting-Lung-Cancer-via-CT-Image-Analysis-using-Deep-Learning-Models
MSc Dissertation Project, Rushikesh Kothawade, University of Leeds

A single, self-contained pipeline (originally developed as a Google Colab notebook, `Detecting Lung Cancer via CT Image Analysis using Deep Learning Models.ipynb`, ~2,550 lines) that trains and compares five deep learning architectures on chest CT scan classification, builds a soft-voting ensemble, applies Grad-CAM explainability and Monte Carlo Dropout uncertainty quantification, and evaluates cross-dataset generalisation on an independent, differently-sourced dataset.

---

## 1. What this pipeline does, in one table

| Stage | What happens | Key output |
|---|---|---|
| Data acquisition | Downloads both datasets from Kaggle via API | `PRIMARY_DIR`, `SECONDARY_DIR` populated |
| EDA | Mean intensity, entropy, PCA variance on primary training split | `eda_intensity_entropy.png` |
| Preprocessing | Resize, ImageNet normalisation, augmentation (train split only) | `train_loader`, `valid_loader`, `test_loader` |
| Model training | 5 architectures trained with per-model hyperparameters | 5 checkpoints (`<model>_best.pt`) |
| Evaluation | Accuracy, precision/recall/F1, confusion matrix, ROC-AUC per model | `test_results.json`, per-class CSVs |
| Ensemble | Soft-voting over top-3 models by validation F1 | `ensemble_config.json` |
| Significance testing | McNemar's test, all pairwise combinations | `mcnemar_all_pairs.csv` |
| Grad-CAM | Saliency maps for the best individual model | `gradcam/` overlays, entropy stats |
| MC Dropout | T=50 stochastic passes, uncertainty-error correlation, triage threshold | `uncertainty_quartile_table.csv`, `mc_dropout_triage_summary.json` |
| Cross-dataset validation | Zero-shot transfer + SMOTE fine-tune on IQ-OTH/NCCD | `cross_dataset/table6_cross_dataset_results.csv` |

---

## 2. Code file structure

The pipeline is organised into nine numbered sections within a single script/notebook. Line numbers refer to `part1_setup_data_instrumented.py` (2,554 lines total).

| # | Section | Lines (approx.) | Purpose |
|---|---|---|---|
| 0 | Run identity & environment setup | 1–126 | Stamps every run with a unique `RUN_ID`, mounts Google Drive, pins `RESULTS_DIR`, logs library/GPU versions |
| 2 | Data Acquisition | 127–238 | Downloads both Kaggle datasets, sanity-checks folder structure via `find_class_root()` |
| 3 | Exploratory Data Analysis | 239–369 | Mean pixel intensity, per-class entropy, PCA variance check (training split only) |
| 4 | Preprocessing Pipeline | 370–529 | `ImageFolder` datasets, train/eval transforms, `DataLoader`s, class-count logging |
| 5 | SMOTE Utility | 530–756 | Defines `apply_smote_to_image_folder()` (used later in cross-dataset fine-tuning); model architecture definitions (`CustomCNN`, `build_resnet50`, etc.) and `MODEL_REGISTRY` also live in this block |
| — | Training loop | 757–1141 | Parameter counts, per-model training with early stopping, checkpoint saving |
| 6 | Per-class metrics + confusion matrices | 1142–1403 | Evaluation on held-out test set for all 5 models + ensemble |
| 7 | All-pairs McNemar's test | 1404–1457 | Statistical significance testing across every model pair |
| — | Grad-CAM | 1458–1729 | `GradCAM` class, per-architecture target-layer resolution, qualitative grid, entropy-spread analysis |
| — | Monte Carlo Dropout | 1730–1916 | `enable_mc_dropout()`, `mc_dropout_predict()`, quartile analysis, triage threshold calibration |
| — | Cross-dataset generalisation | 1917–2547 | Zero-shot evaluation, case-grouped split, SMOTE-balanced fine-tuning of DenseNet-121's head |
| 9 | Final run manifest | 2548–2554 | Consolidates all run metadata into a single exportable summary |

---

## 3. Models compared

| Model | Parameters | Pretrained | Fine-tuning strategy | Input size |
|---|---|---|---|---|
| Custom CNN | 410,308 | No (from scratch) | Single-phase, 50 epochs max | 224×224 |
| ResNet-50 | 23,516,228 | ImageNet | Two-phase: head only (5 ep) → full unfreeze (low LR) | 224×224 |
| VGG-16 | 138,459,972 | ImageNet | Single-phase, extended custom head | 224×224 |
| DenseNet-121 | 6,957,956 | ImageNet | Two-phase: head only (5 ep) → full unfreeze (low LR) | 224×224 |
| EfficientNet-B3 | 10,702,380 | ImageNet | Single-phase, label-smoothing loss | 300×300 |

Each model's optimiser, learning rate, scheduler, batch size, and regularisation settings are defined independently in `MODEL_REGISTRY` (see code) — this is the single source of truth for every hyperparameter used in training, and is reproduced in full in the dissertation's Appendix A.

---

## 4. Datasets

Both datasets are pulled from Kaggle at runtime and are **not** stored in this repository.

| Dataset | Role | Classes | Size | Source |
|---|---|---|---|---|
| Chest CT-Scan Images | Primary — train/validation/test | Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal | 1,000 images (613 / 72 / 315 split) | [Kaggle: mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) |
| IQ-OTH/NCCD Lung Cancer Dataset | Secondary — cross-dataset validation only | Normal, Benign, Malignant | 1,097 images | [Kaggle: hamdallak/the-iqothnccd-lung-cancer-dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset) |

**Note:** there are multiple re-uploads of the IQ-OTH/NCCD dataset on Kaggle under different slugs — verify the slug above matches the version you intend to use before running.

You will need a Kaggle API token (`kaggle.json`) — see the [Kaggle API docs](https://www.kaggle.com/docs/api). The notebook prompts for this on first run.

---

## 5. Results summary

| Model | Test Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Custom CNN (baseline) | 46.35% | 0.5072 | 0.5375 | 0.4961 | 0.6794 |
| ResNet-50 | 62.86% | 0.7437 | 0.6041 | 0.6077 | 0.8570 |
| EfficientNet-B3 | 53.65% | 0.6509 | 0.5761 | 0.5194 | 0.8104 |
| VGG-16 | 80.95% | 0.8673 | 0.8630 | 0.8422 | 0.9860 |
| **DenseNet-121** | **84.76%** | 0.8680 | 0.8550 | **0.8597** | 0.9619 |
| **Ensemble** (VGG-16 + DenseNet-121 + ResNet-50) | **86.67%** | 0.8985 | 0.9033 | **0.8897** | **0.9859** |

The ensemble's gain over DenseNet-121 alone (+1.91pp) is **not statistically significant** under McNemar's test (χ²=0.4808, p=0.4881). DenseNet-121 and VGG-16 are likewise not significantly distinguishable from each other (p=0.2010), despite a 3.81pp accuracy gap.

### Cross-dataset generalisation (IQ-OTH/NCCD)

| Condition | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Ensemble, zero-shot (no fine-tune) | 51.32% | 0.4052 | 0.4122 | 0.3528 |
| DenseNet-121, zero-shot (no fine-tune) | 38.92% | 0.3076 | 0.3334 | 0.2226 |
| SMOTE + DenseNet-121, fine-tuned | 84.80% | 0.8212 | 0.7348 | 0.7620 |

*The dataset's majority-class baseline is 51.14% — zero-shot ensemble performance is statistically indistinguishable from naive majority-class guessing, and zero-shot DenseNet-121 performs below it.*

### MC Dropout triage (VGG-16, T=50, 90th-percentile threshold)

| Uncertainty Quartile | Entropy Range | % of Test Set | Accuracy |
|---|---|---|---|
| Q1 (Low) | H < 0.010 | 25.08% | 100.00% |
| Q2 | 0.010–0.276 | 24.76% | 96.15% |
| Q3 | 0.276–0.688 | 25.08% | 74.68% |
| Q4 (High) | H > 0.688 | 25.08% | 49.37% |

At the calibrated review threshold (90th percentile of training-set entropy), **52.4%** of test cases are flagged for review, capturing **95.2%** of the model's total errors (60 of 63).

---

## 6. Environment & requirements

Developed and executed on **Google Colab**, using a **Tesla T4 GPU** runtime with Google Drive mounted for persistent storage.

| Package | Version used |
|---|---|
| Python | 3.x (Colab default) |
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

To run outside Colab: set `USE_DRIVE = False` in the setup cell to use a local working directory instead of a mounted Drive path, and ensure a CUDA-capable GPU is available — training five architectures plus MC Dropout's 50x inference cost is not practical on CPU alone.

---

## 7. Reproducibility

- Every run is stamped with a unique `RUN_ID` (timestamp-based) at execution start; all figures, tables, checkpoints, and JSON summaries for that run are written under `results/<RUN_ID>/`, so different executions never overwrite one another.
- A `manifest` dictionary accumulates library versions, GPU info, dataset class counts, per-model parameter counts, and convergence stats throughout the run.
- Random seed fixed at `SEED = 42` across Python's `random`, NumPy, and PyTorch (including CUDA), with `torch.backends.cudnn.deterministic = True`. Exact bit-for-bit reproducibility across different GPU types is not guaranteed.

---

## 8. Known limitations

| Limitation | Detail |
|---|---|
| Case-grouping fallback | `extract_case_id()` cannot parse IQ-OTH/NCCD's filename convention and falls back to one group per image — the fine-tuning adaptation/test split is image-level, not confirmed patient-level |
| SMOTE on raw pixels | Interpolation happens on flattened pixel vectors, not a learned feature embedding — a defensible but weaker baseline |
| MC Dropout threshold calibration | The 90th-percentile review threshold is calibrated on the training set, not a held-out split |
| Retrospective evaluation only | All results are on public benchmark datasets with fixed labels; no prospective clinical validation |

Full discussion of each limitation, and its implications for the reported results, is in the accompanying dissertation (Chapters 3, 5, and 6).

---

## 9. Citation

> Kothawade, R. (2026) *Lung Cancer Detection from CT Scans: A Deep Learning Framework with Explainability and Uncertainty Quantification*. MSc Dissertation, University of Leeds.

## License

Code is made available for academic and research purposes. Dataset licenses are governed by their respective Kaggle sources (Section 4).

