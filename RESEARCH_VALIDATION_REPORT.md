# Research Validation Report

This report summarizes the findings and validation metrics of the **Research Integrity and Experiment Validation Pass** conducted on the bone radiography osteoporosis classification model.

---

## 1. Confirmed Issues & Completed Fixes

### Fix 1: Experiment 2 Resolution Correction
* **Confirmed Issue**: The model checkpoint `best_resnet50_exp2.pth` was trained on images resized to $256 \times 256$ pixels, but evaluated in `evaluate_exp2.py` at $224 \times 224$ pixels. This mismatch introduces spatial distortion and invalidates the test metrics.
* **Fix Completed**: We developed `src/evaluate_exp2_corrected.py` to evaluate the model at the correct $256 \times 256$ resolution. 
* **Corrected Results**: Overall test accuracy remained at **84.50%** ($109/129$). However, the class metrics shifted: Osteoporosis recall increased from **$79.17\%$ to $83.33\%$**, and false negatives (Osteoporosis misclassified as Normal) decreased from 6 to 4. Corrected metrics are saved in `results/exp2_corrected/`.

### Fix 2: Reproducibility & Training Seeding
* **Confirmed Issue**: The original training scripts (`train.py` and `train_exp3.py`) did not set random seeds for Python, NumPy, or PyTorch, resulting in non-deterministic training runs.
* **Fix Completed**: We added an explicit `set_seed(seed=42)` configuration to both training scripts inside the audit package (`osteoporosis_research_audit/src/`), locking PyTorch manual seeds, CUDA seeds, and forcing CuDNN deterministic operations.

### Fix 3: Explainability & Interpretability
* **Confirmed Issue**: The original codebase contained no explainability mechanisms (such as Grad-CAM or heatmaps), rendering the model a "black box" that could not be clinically audited.
* **Fix Completed**: We developed a hook-based Grad-CAM analyzer (`src/gradcam_analysis.py`) targeting the final bottleneck block (`model.layer4[-1]`) of the ResNet-50 backbone. Saliency maps were generated for correct predictions and key Normal ↔ Osteoporosis mistakes, saved under `results/gradcam/`.

---

## 2. Patient-Level 5-Fold Cross-Validation Findings (Primary Dataset Only)

To evaluate the generalization performance of the architecture in a strictly controlled patient-level environment, we executed a stratified 5-fold cross-validation on the **Primary Dataset only** ($N = 239$ unique patients/images). In this dataset, the patient structure is verified (exactly 1 image per patient), and zero patient overlap exists between train and validation folds.

### Baseline vs. Experiment 1 Configuration (Mean ± SD)

| Metric | Frozen ResNet-50 Baseline (Model A) | Fine-Tuned ResNet-50 (Model B / Exp 1 Config) | Relative Change |
| :--- | :---: | :---: | :---: |
| **Accuracy** | $40.58\% \pm 11.13\%$ | **$51.00\% \pm 7.44\%$** | **+10.42%** (absolute) |
| **Macro F1** | $0.2769 \pm 0.0395$ | **$0.3885 \pm 0.0358$** | **+0.1116** (absolute) |
| **Macro Precision** | $0.2879 \pm 0.0583$ | **$0.4191 \pm 0.0617$** | **+0.1312** (absolute) |
| **Macro Recall** | $0.3252 \pm 0.0501$ | **$0.4028 \pm 0.0233$** | **+0.0776** (absolute) |
| **Weighted F1** | $0.3883 \pm 0.0944$ | **$0.5093 \pm 0.0614$** | **+0.1210** (absolute) |
| **Macro ROC-AUC** | $0.5543 \pm 0.0329$ | **$0.5760 \pm 0.0362$** | **+0.0217** (absolute) |

### Selected Configuration
Based on validation performance, the **Experiment 1 configuration** was selected and locked:
* Unfreezes layers 3 & 4 of the ResNet-50 backbone.
* Uses discriminative learning rates ($1e-5$ for backbone, $1e-4$ for classifier head).
* Optimizer: AdamW.
* Loss: Weighted Cross-Entropy (class weights calculated dynamically on train fold).

---

## 3. Final Locked Test Evaluation Results

After locking the configuration, a final model was trained from scratch using seed 42 on the training split `train_final.csv` for a maximum of 25 epochs, with `val_final.csv` used to evaluate validation loss after each epoch and select the best checkpoint (**Correction, terminology correction pass:** this sentence previously read "...using `val_final.csv` for early stopping." No training run was halted before its planned epoch count — the model trained the full 25 epochs every time, and the checkpoint with the lowest validation loss, which occurred at epoch 25, was retained. **Superseded, not to be cited: "early stopping" — use "best-checkpoint selection based on validation loss."**). A single evaluation was then executed on the locked test set `test_final.csv` ($N = 129$).

### FINAL LOCKED TEST EVALUATION RESULTS
* **Overall Test Accuracy**: **86.82%** ($112 / 129$ correct classifications)
* **95% Confidence Intervals for Accuracy**:
  * **95% Bootstrap CI (10,000 resamples)**: **[80.62%, 92.25%]**
  * **95% Wald CI**: **[80.98%, 92.66%]**
* **Macro Precision**: **0.8663**
* **Macro Recall**: **0.8878**
* **Macro F1-Score**: **0.8709**
* **Weighted F1-Score**: **0.8675**
* **Macro ROC-AUC**: **0.9346**

### Confusion Matrix
$$\begin{pmatrix} 41 & 3 & 8 \\ 0 & 29 & 0 \\ 2 & 4 & 42 \end{pmatrix}$$

### Per-Class ROC-AUC (One-vs-Rest)
* **Normal Class**: **0.9173**
* **Osteopenia Class**: **0.9628**
* **Osteoporosis Class**: **0.9236**

> **Correction (numerical correction pass):** this section previously read Normal 0.9340 / Osteopenia 0.9155 / Osteoporosis 0.9542. Those values did not match `results/final_test/final_test_metrics.json` and were traced and independently recomputed twice from `final_test_predictions.csv`'s raw per-class probability columns (see `ROC_AUC_DISCREPANCY_AUDIT.md`), both times reproducing 0.9173 / 0.9628 / 0.9236 to full floating-point precision. The Macro ROC-AUC figure above (0.9346) was already correct and is unchanged. **Superseded values, preserved for the historical record and not to be cited: Normal 0.9340 [SUPERSEDED — use 0.9173], Osteopenia 0.9155 [SUPERSEDED — use 0.9628], Osteoporosis 0.9542 [SUPERSEDED — use 0.9236].**

---

## 4. Final Error Analysis

An audit of the 17 misclassified cases on the test set revealed the following patterns:
* **Dominant Confusion**: **Normal predicted as Osteoporosis (8 cases)** and **Osteoporosis predicted as Osteopenia (4 cases)** represent the most common errors.
* **High-Confidence Mistakes**: **10 out of the 17 errors** are high-confidence mistakes (confidence $\ge 70\%$). This indicates the model is making confident, systematically incorrect predictions (e.g. `OS41.jpg` predicted as osteopenia with $97.51\%$ confidence).
* **Clinical False Negatives**: `Osteoporosis 28.png` was misclassified as Normal with extremely high confidence (**95.56%**). This is a critical error where osteoporotic bone is predicted as completely healthy.

---

## 5. Final Grad-CAM Interpretability Findings

Saliency maps generated targeting the final block (`model.layer4[-1]`) on correct and incorrect predictions indicate the following:
* **Joint and Trabecular Focus**: In correct classifications, Grad-CAM visualizations suggested that the model's attention was concentrated near specific bone joint spaces, tibial plateaus, and trabecular regions.
* **Peripheral distraction**: In high-confidence errors (such as the Normal scan `Normal 114.jpg` predicted as osteoporosis), visualizations indicated that attention shifted away from the bone anatomy towards soft tissue boundaries and image margins.
* **Disclaimer**: Saliency heatmaps suggest feature correlation and output influence; **they do not prove clinical reasoning or causality**.

---

## 6. Remaining Limitations & Peer Review Guidelines

1. **Patient Leakage on Combined Dataset**: The secondary dataset ($712$ images) lacks patient identifiers. **Patient-level separation could not be established for the complete combined dataset.** This remains a threat to validity that must be disclosed in the paper.
2. **No External Generalization**: All results are generated on splits of the same local dataset. Testing on an independent external cohort is required before claiming clinical utility.
3. **Statistical Significance**: No paired statistical significance test (such as McNemar's test) has been conducted to compare the test performances of Experiment 1, 2, and 3. Therefore, no claims of statistical significance should be made.
