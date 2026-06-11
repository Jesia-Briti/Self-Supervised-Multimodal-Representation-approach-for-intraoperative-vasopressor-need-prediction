# Multimodal Representation Model for Hypotension Forecasting

Predicts hypotension a few minutes before it happens from three signals recorded together: PPG, ABP, and ECG. A foundation model is pretrained on a large unlabelled dataset, then fine-tuned for the clinical task.

## Datasets

- **Pretraining:** MIMIC-BP. Unlabelled PPG/ABP/ECG triplets.
- **Downstream:** VitalDB. Vasopressor-treated cases as positives and matched control cases as negatives, aligned by exact time offset before the event.

## Files

- `pretraining-code.ipynb` — self-supervised pretraining on MIMIC-BP. Saves the encoder weights used downstream.
- `downstream_code.ipynb` — loads the pretrained encoder and trains the prediction head. 5-fold cross-validation across horizons of 3, 5, 7, 10, 12, and 15 minutes before event.
- `build_master_dataset.py` — assembles the downstream dataset. Splits patients 80/10/10 into train/val/test with no patient leakage.


