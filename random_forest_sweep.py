import os
import numpy as np
import torch
import joblib
from torch.utils.data import random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from surrogate import H5Dataset  # Same as before

def main(data_path):
    # Load and preprocess dataset
    full_dataset = H5Dataset(data_path, raw=True)

    # Determine validity
    isnan_mask = torch.isnan(full_dataset.y_norm[:, :4]).any(dim=1)
    validity = (~isnan_mask).float()
    print(f"{validity.sum().item()} / {len(full_dataset)} data points are valid.")

    # Extract input features and labels
    X = full_dataset.x_norm.numpy()
    y = validity.numpy().astype(int)

    # Split dataset
    total_size = len(X)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    indices = np.random.permutation(total_size)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"  # helps with imbalance
    )
    clf.fit(X_train, y_train)
    model_path = "outputs/random_forest_model.joblib"
    joblib.dump(clf, model_path)

    # Evaluate on test set
    y_probs = clf.predict_proba(X_test)[:, 1]
    threshold = 0.7  # Can be adjusted based on desired precision/recall
    y_pred = (y_probs > threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Invalid", "Valid"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    avg_precision = average_precision_score(y_test, y_probs)

    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.show()

class RandomForest:
    def __init__(self, model_path):
        self.clf = joblib.load(model_path)
    def __call__(self, x):
        x_np = x.detach().cpu().numpy()
        probs = self.clf.predict_proba(x_np)[:, 1]
        return torch.tensor(probs, device=x.device)
        
if __name__ == "__main__":
    data_path = "datasets/bbp_ds_2m_merged_v2.h5"
    main(data_path)
