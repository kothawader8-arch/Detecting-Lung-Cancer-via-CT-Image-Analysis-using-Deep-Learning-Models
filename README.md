# Detecting-Lung-Cancer-via-CT-Image-Analysis-using-Deep-Learning-Models
MSc Dissertation Project  Rushikesh Kothawade, University of Leeds

A deep learning framework for multi-class lung cancer classification from chest CT images, combining five model architectures, a soft-voting ensemble, Grad-CAM explainability, Monte Carlo Dropout uncertainty quantification, and cross-dataset generalisation evaluation.

Overview

This repository implements the full experimental pipeline for classifying chest CT scan slices into four categories: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal and evaluating cross-dataset generalisation on an independent, differently-sourced 3-class dataset (Normal / Benign / Malignant).

The project addresses three widely cited gaps in the CT-based lung cancer detection literature: the near-total separation between high accuracy and explainability/uncertainty-aware work, the scarcity of multi-class (subtype-level) studies relative to binary cancerous/non-cancerous framings, and the general absence of cross-dataset validation in prior published results on these benchmarks.

Key Results
Model	Test Accuracy	Macro F1	AUC-ROC
Custom CNN (baseline)	46.35%	0.4961	0.6794
ResNet-50	62.86%	0.6077	0.8570
EfficientNet-B3	53.65%	0.5194	0.8104
VGG-16	80.95%	0.8422	0.9860
DenseNet-121	84.76%	0.8597	0.9619
Ensemble (VGG-16 + DenseNet-121 + ResNet-50)	86.67%	0.8897	0.9859
The ensemble's gain over DenseNet-121 alone (+1.91pp) is not statistically significant under McNemar's test (χ²=0.4808, p=0.4881).
MC Dropout (VGG-16, T=50 passes) flags 52.4% of test cases for review while catching 95.2% of the model's total errors.
Zero-shot transfer to the IQ-OTH/NCCD dataset collapses to 51.32% (ensemble) / 38.92% (DenseNet-121 alone) — both near or below the dataset's 51.14% majority-class baseline. Lightweight SMOTE-based fine-tuning of DenseNet-121's classification head recovers accuracy to 84.80% (macro F1 = 0.7620).

Full results, per-class metrics, and discussion are in the accompanying dissertation report.

Repository Structure
├── part1_setup_data_instrumented.ipynb   # Environment setup, data acquisition, EDA, preprocessing
├── part2_...ipynb                        # Model architectures, training loops (see notebook for exact split)
├── part3_...ipynb                        # Evaluation, ensemble construction, McNemar's testing
├── part4_...ipynb                        # Grad-CAM explainability
├── part5_...ipynb                        # Monte Carlo Dropout uncertainty quantification
├── part6_...ipynb                        # Cross-dataset generalisation (zero-shot + SMOTE fine-tuning)
├── README_instrumentation.md             # Notes on run-identity stamping and reproducibility instrumentation
└── results/
    └── <RUN_ID>/                         # Per-run outputs: figures, tables, checkpoints, JSON summaries

Note: update the file list above to match the actual notebook filenames once all parts are uploaded,this pipeline was developed and run as a multi-part Colab notebook sequence (Part 1 shown in the excerpt this README was derived from).

Datasets

Both datasets are pulled from Kaggle at runtime and are not included in this repository.

Dataset	Role	Classes	Size	Source
Chest CT-Scan Images	Primary:training/validation/test	Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal	1,000 images (613/72/315 split)	Kaggle: mohamedhanyyy/chest-ctscan-images
IQ-OTH/NCCD Lung Cancer Dataset	Secondary:cross-dataset generalisation only	Normal, Benign, Malignant	1,097 images	Kaggle: hamdallak/the-iqothnccd-lung-cancer-dataset

You will need a Kaggle API token (kaggle.json) to download the datasets — see Kaggle API documentation for how to generate one. The notebooks prompt for this token on first run and cache it for the session.

Verify the secondary dataset slug before running,there are multiple re-uploads of the IQ-OTH/NCCD dataset on Kaggle under different slugs; confirm the one above matches the version you intend to cite.

Environment & Requirements

Developed and executed on Google Colab using a Tesla T4 GPU runtime.

python >= 3.10
torch >= 2.11.0+cu128
torchvision >= 0.26.0+cu128
scikit-learn >= 1.6.1
imbalanced-learn
statsmodels
opencv-python
pandas
numpy
matplotlib
Pillow
kaggle

Install via:

bash
pip install -q imbalanced-learn kaggle statsmodels

(torch, torchvision, scikit-learn, pandas, numpy, matplotlib, opencv-python are pre-installed in the Colab runtime.)

To run outside Colab, set USE_DRIVE = False in the setup cell to use a local working directory instead of a mounted Google Drive path, and ensure a CUDA-capable GPU is available (training on CPU is not recommended given the five-architecture comparison and MC Dropout's 50x inference cost).

Reproducibility

Every run is stamped with a unique RUN_ID (timestamp-based) at execution start, and all figures, tables, checkpoints, and JSON summaries for that run are written under results/<RUN_ID>/, so outputs from different executions never overwrite one another. A manifest dictionary accumulates library versions, GPU info, dataset class counts, per-model parameter counts, and convergence stats throughout the run and can be inspected or exported for full run provenance.

The random seed is fixed (SEED = 42) across Python's random, NumPy, and PyTorch (including CUDA), with torch.backends.cudnn.deterministic = True, though exact bit-for-bit reproducibility across different GPU types is not guaranteed.

Pipeline Stages
Setup & Data Acquisition:environment setup, Kaggle dataset download, folder-structure validation.
Exploratory Data Analysis:mean pixel intensity, per-class entropy, PCA variance check on the primary training split.
Preprocessing: resize (224×224, or 300×300 for EfficientNet-B3), ImageNet normalisation, domain-informed augmentation (flip, rotation, brightness/contrast jitter, Gaussian blur, random resized crop) on the training split only.
Model Training:five architectures (Custom CNN, ResNet-50, VGG-16, DenseNet-121, EfficientNet-B3), each with its own optimiser, scheduler, and (for the transfer-learning models) a two-phase frozen-then-unfrozen fine-tuning strategy where applicable.
Evaluation:accuracy, macro precision/recall/F1, confusion matrices, one-vs-rest ROC-AUC on the held-out primary test set.
Ensemble Construction: soft voting over the top-3 models by validation macro-F1, weighted proportionally to validation F1.
Statistical Testing: McNemar's test, both ensemble-vs-best and all-pairs, on paired test-set prediction outcomes.
Grad-CAM: gradient-weighted class activation mapping applied to the best individual model, with architecture-specific target-layer resolution for all five model families.
Monte Carlo Dropout: T=50 stochastic forward passes on the best dropout-capable architecture, producing per-sample predictive entropy, quartile-stratified accuracy analysis, and an operational triage threshold calibrated at the 90th percentile of training-set entropy.
Cross-Dataset Generalisation: zero-shot evaluation of the primary-trained ensemble and best individual model on IQ-OTH/NCCD (via max-pooling the three malignant-subtype outputs into a single "Malignant" class), followed by SMOTE-rebalanced, patient/case-grouped fine-tuning of the classification head only.
Known Limitations (see dissertation for full discussion)
The case-grouping step for the IQ-OTH/NCCD adaptation/test split does not reliably infer patient identity from this dataset's filename convention and falls back to image-level grouping, the fine-tuning results should be read as image-level, not confirmed patient-level, held-out performance.
SMOTE is applied to flattened raw pixel vectors rather than a learned feature embedding, a weaker but more direct baseline.
MC Dropout's review threshold is calibrated on the training set rather than a held-out validation split.
All evaluation is retrospective, on public benchmark datasets with fixed labels; no prospective clinical validation has been conducted.
Citation

If referencing this work, please cite the accompanying MSc dissertation:

Kothawade, R. (2026) Lung Cancer Detection from CT Scans: A Deep Learning Framework with Explainability and Uncertainty Quantification. MSc Dissertation, University of Leeds.

License

This code is made available for academic and research purposes. Dataset licenses are governed by their respective Kaggle sources (see links above).
