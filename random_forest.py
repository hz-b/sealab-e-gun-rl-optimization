import os
import numpy as np
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
    splits = load_and_split_dataset(data_path, seed)
    X_train, y_train = splits["X_train"], splits["y_train"]

    clf = RandomForestClassifier(
        n_estimators=100,
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
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_prec = average_precision_score(y_test, y_probs)

    if plot:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Invalid", "Valid"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"[Run {run_idx}] Confusion Matrix")
        plt.savefig("outputs/random_forest_cm.pdf")

        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {avg_prec:.3f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"[Run {run_idx}] Precision-Recall Curve")
        plt.legend()
        plt.grid()
        plt.savefig("outputs/random_forest_prc.pdf")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "avg_precision": avg_prec,
    }

def main():
    data_path = "datasets/bbp_ds_2m_merged_v2.h5"

    # === Train Phase ===
    for run_idx in range(3):
        model_path, X_test, y_test = train_random_forest(data_path, run_idx=run_idx, seed=42+run_idx)

        # === Test Phase ===
        metrics = test_random_forest(model_path, X_test, y_test, run_idx=1, threshold=0.7)
        
        print("\n=== Final Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()

