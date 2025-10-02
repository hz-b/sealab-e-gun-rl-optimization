import os
import numpy as np
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, fbeta_score
)
import matplotlib.pyplot as plt
from surrogate import H5Dataset  # Your custom dataset

def load_and_split_dataset(data_path, seed=None):
    full_dataset = H5Dataset(data_path, raw=True)

    # Determine validity
    isnan_mask = torch.isnan(full_dataset.y_norm[:, :4]).any(dim=1)
    validity = (~isnan_mask).float()
    print(f"[Data] {validity.sum().item()} / {len(full_dataset)} valid data points.")

    X = full_dataset.x_norm.numpy()
    y = validity.numpy().astype(int)

    # Shuffle and split
    total_size = len(X)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_size)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    splits = {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx]
    }
    return splits

def train_random_forest(data_path, run_idx=1, seed=None, fit=True):
    splits = load_and_split_dataset(data_path, 10000)
    X_train, y_train = splits["X_train"], splits["y_train"]

    clf = RandomForestClassifier(
        n_estimators=2,#00,
        max_depth=None,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced"
    )
    if fit:
        clf.fit(X_train, y_train)

    os.makedirs("outputs", exist_ok=True)
    model_path = f"outputs/random_forest_model_run{run_idx}.joblib"
    if fit:
        joblib.dump(clf, model_path)
    print(f"[Run {run_idx}] Model: {model_path}")
    return model_path, splits["X_test"], splits["y_test"]

def test_random_forest(model_path, X_test, y_test, run_idx=1, threshold=0.7, plot=True):
    clf = joblib.load(model_path)
    y_probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)

    print(f"\n[Run {run_idx}] Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_prec = average_precision_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    if plot:
        # --- Confusion Matrix ---
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Invalid", "Valid"])
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(f"outputs/random_forest_cm_run{run_idx}.pdf")
        plt.close()

        # --- Precision-Recall Curve (PRC) ---
        precisions, recalls, prc_thresholds = precision_recall_curve(y_test, y_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label="PR Curve", color="blue")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/random_forest_prc_run{run_idx}.pdf")
        plt.close()

        # --- Precision, Recall, F2 vs. Threshold ---
        thresholds = np.linspace(0, 1, 200)
        precisions_thr = []
        recalls_thr = []
        f2_scores = []

        for t in thresholds:
            y_pred_t = (y_probs >= t).astype(int)
            p = precision_score(y_test, y_pred_t, zero_division=0)
            r = recall_score(y_test, y_pred_t)
            f2 = fbeta_score(y_test, y_pred_t, beta=2)
            precisions_thr.append(p)
            recalls_thr.append(r)
            f2_scores.append(f2)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, recalls_thr, label='Recall', color='green')
        plt.plot(thresholds, precisions_thr, label='Precision', color='blue')
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Selected Threshold = {threshold:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"outputs/random_forest_threshold_metrics_run{run_idx}.pdf")
        plt.close()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "avg_precision": avg_prec,
        "confusion_matrix": cm,
    }

def main():
    data_path = "datasets/bbp_ds_2m_merged_v2.h5"
    runs = 3
    total_cm = np.zeros((2, 2), dtype=int)  # Assuming binary classification

    for run_idx in range(runs):
        #model_path, X_test, y_test = train_random_forest(data_path, run_idx=run_idx, seed=42 + run_idx)
        model_path = f"outputs/random_forest_model_run{run_idx}.joblib"
        metrics = test_random_forest(model_path, X_test, y_test, run_idx=run_idx, threshold=0.7)
        

        # Accumulate confusion matrix
        total_cm += metrics["confusion_matrix"]

    # === Final aggregated confusion matrix ===
    total_cm = total_cm/runs
    fig, ax = plt.subplots(figsize=(4, 3))  # Change (width, height) as needed


    disp = ConfusionMatrixDisplay(confusion_matrix=total_cm.astype(int), display_labels=["Invalid", "Valid"])
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    plt.savefig("outputs/random_forest_cm_aggregated.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()
