# load MIT-BIH files and make (X, y) for training

import wfdb
import numpy as np


# AAMI standard - group many beat types into 5 classes
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}

CLASS_TO_INDEX = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

WINDOW = 180  # half second each side, total beat length = 360 samples


def load_patient_record(record_id, data_dir):
    path = f"{data_dir}/{record_id}"

    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')

    signal = record.p_signal[:, 0]  # use first channel only
    return signal, annotation


def extract_beats(signal, annotation, window=WINDOW):
    beats = []
    labels = []

    for sample_index, symbol in zip(annotation.sample, annotation.symbol):
        if sample_index - window < 0 or sample_index + window > len(signal):
            continue
        if symbol not in AAMI_MAP:
            continue

        beat_window = signal[sample_index - window: sample_index + window]  # one heartbeat = one training sample
        beats.append(beat_window)
        labels.append(CLASS_TO_INDEX[AAMI_MAP[symbol]])

    return np.array(beats, dtype=np.float32), np.array(labels, dtype=np.int64)


def normalize_beats(beats):
    # different patients have different signal strength, so normalize each beat
    mean = beats.mean(axis=1, keepdims=True)
    std = beats.std(axis=1, keepdims=True) + 1e-8
    return (beats - mean) / std


def build_client_dataset(record_id, data_dir):
    signal, annotation = load_patient_record(record_id, data_dir)
    X, y = extract_beats(signal, annotation)
    X = normalize_beats(X)
    return X, y


if __name__ == "__main__":
    DATA_DIR = "./mit-bih-arrhythmia-database-1.0.0"
    record_id = "100"

    X, y = build_client_dataset(record_id, DATA_DIR)

    print(f"Patient {record_id}")
    print(f"  Number of beats extracted: {len(X)}")
    print(f"  Shape of one beat: {X[0].shape}")
    print(f"  Class counts:")
    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"    {CLASS_NAMES[idx]:18s}: {count}")
