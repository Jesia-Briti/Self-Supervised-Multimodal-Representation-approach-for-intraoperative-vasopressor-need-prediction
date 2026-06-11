# Multimodal Foundation Model for Hypotension Forecasting

Predicts hypotension a few minutes before it happens from three signals recorded together: PPG, ABP, and ECG. A foundation model is pretrained on a large unlabelled dataset, then fine-tuned for the clinical task.

## Datasets

- **Pretraining:** MIMIC-BP. Unlabelled PPG/ABP/ECG triplets.
- **Downstream:** VitalDB. Vasopressor-treated cases as positives and matched control cases as negatives, aligned by exact time offset before the event.

## Files

- `pretraining-code_patch30.ipynb` — self-supervised pretraining on MIMIC-BP. Saves the encoder weights used downstream.
- `downstream_patch30.ipynb` — loads the pretrained encoder and trains the prediction head. 5-fold cross-validation across horizons of 3, 5, 7, 10, 12, and 15 minutes before event.
- `build_master_dataset.py` — assembles the downstream dataset. Splits patients 80/10/10 into train/val/test with no patient leakage.

## Model

Each signal has its own small transformer. PPG and ABP use simple patching; ECG uses a learnable wavelet front-end. The three streams meet in a shared cross-modal transformer. Inputs are 30-second windows at 125 Hz, cut into 125 patches.

## Pretraining

Mixed objective: reconstruct masked regions in time and frequency, match a PPG morphology loss anchored on key fiducial points, and stay contrastively consistent across masked views and across modalities.

## Downstream task

Two heads on top of the pretrained encoder: a binary classifier for whether the patient will need a vasopressor soon, and three regression heads producing a 15-step MAP/SBP/DBP trajectory toward the event. Focal loss for the classifier, Huber for the trajectories. Class imbalance handled with weighted sampling and augmentation of positives. Final probabilities are temperature-calibrated and the test scores come from a 5-fold ensemble with bootstrap confidence intervals.

## How to run

1. Run `build_master_dataset.py` to assemble the train/val/test folders.
2. Run `pretraining-code_patch30.ipynb` on MIMIC-BP to produce the encoder checkpoint.
3. Run `downstream_patch30.ipynb` pointing at that checkpoint.

Paths are set up for Kaggle. Seed is 42, patient splits are locked.
