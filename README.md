# Federated ECG Arrhythmia Classification

This project classifies heartbeats from ECG signals using **federated learning (FL)**.

Simple idea:
- Each patient is treated as one separate client (like one hospital).
- Each client trains on its own ECG data locally.
- Only **model weights** are shared with the server, not raw ECG data.
- Server averages the weights and sends the updated global model back.
- This repeats for multiple rounds.

Then we compare:
- **Federated model** (clients collaborate through weight sharing)
- **Local-only model** (each patient trains alone, no collaboration)

---

## What problem this solves

ECG data is private. Hospitals/patients cannot easily share raw recordings.

Federated learning allows collaboration **without sharing raw patient data**.

---

## Dataset

We use the **MIT-BIH Arrhythmia Database**, already included in:

`./mit-bih-arrhythmia-database-1.0.0/`

Current run uses these 5 patients:
- `100`, `101`, `103`, `105`, `108`

Each heartbeat is classified into one of 5 AAMI classes:
- Normal
- Supraventricular
- Ventricular
- Fusion
- Unknown

---

## Project files

- `data_prep.py` - reads MIT-BIH files and prepares training data
- `model.py` - 1D CNN model (`ECGNet`)
- `train_utils.py` - training, testing, and train/test split functions
- `flower_client.py` - Flower client logic (local train + evaluate)
- `run_federated.py` - main script (run this file)
- `requirements.txt` - required Python packages

---

## How to run

### 1) Clone and go to project folder

```bash
git clone https://github.com/YOUR_USERNAME/federated-ecg-fl.git
cd federated-ecg-fl
```

Dataset is already included in `mit-bih-arrhythmia-database-1.0.0/` — no extra download needed.

### 2) Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install packages

```bash
pip install -r requirements.txt
```

### 4) (Optional) Check data loading

```bash
python data_prep.py
```

### 5) Run main experiment

```bash
python run_federated.py
```

---

## What happens when you run

1. Loads ECG data for selected patients
2. Splits each patient's beats into train (80%) and test (20%)
3. Runs federated training for **10 rounds**
4. Trains local-only baseline for each patient
5. Prints accuracy comparison in terminal
6. Saves two plots:
   - `federated_vs_local_comparison.png`
   - `worst_patients_federated_progress.png`

---

## Important settings (`run_federated.py`)

```python
PATIENT_IDS = ['100', '101', '103', '105', '108']
NUM_ROUNDS = 10
EPOCHS_PER_ROUND = 5
NUM_WORST_TO_HIGHLIGHT = 5
```

- Change `PATIENT_IDS` to use different patients
- Increase `NUM_ROUNDS` for longer federated training
- `NUM_WORST_TO_HIGHLIGHT` controls how many weakest local patients are shown in the extra comparison

---

## How to read results

You may see:
- Federated accuracy higher than local-only (good case for FL)
- Federated accuracy slightly lower than local-only (also possible)

This can happen because:
- Patient data is very different from each other (non-IID)
- Local model may fit one patient better
- Federated model is optimized for all patients together

We especially check **worst local-performing patients** to see if federated learning helps weaker cases.

---

## Privacy point (main project message)

In this setup:
- Raw ECG beats stay on each client side
- Only model weights are exchanged
- So collaboration is possible with better privacy protection

---

## Future work idea

Add secure communication for weight sharing (for example post-quantum key exchange) and compare overhead vs this baseline FL setup.
