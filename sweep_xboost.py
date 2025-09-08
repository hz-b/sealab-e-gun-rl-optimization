from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    log_loss,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)
from collections import defaultdict
from surrogate import H5Dataset
import torch
import numpy as np
import xgboost as xgb
import pickle

def sweep_hyperparameter(data_path, param_name, values, base_params, num_runs=5, num_boost_round=200, threshold=0.5):
    results = defaultdict(list)

    for val in values:
        print(f"\n🔍 {param_name} = {val}")
        f1_list = []
        logloss_list = []
        confusion_entries = []

        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}", end="\r")

            full_dataset = H5Dataset(data_path, raw=True)
            isnan_mask = torch.isnan(full_dataset.y_norm[:, :4]).any(dim=1)
            validity = (~isnan_mask).float()

            X = full_dataset.x_norm.numpy()
            y = validity.numpy().astype(int)

            total_size = len(X)
            train_size = int(0.6 * total_size)
            val_size = int(0.2 * total_size)
            test_size = total_size - train_size - val_size

            indices = np.random.permutation(total_size)
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            dtest = xgb.DMatrix(X_test, label=y_test)

            params = base_params.copy()
            params[param_name] = val

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=False
            )

            y_probs = model.predict(dtest)
            y_pred = (y_probs > threshold).astype(int)

            f1 = f1_score(y_test, y_pred)
            logloss = log_loss(y_test, y_probs)
            cm = confusion_matrix(y_test, y_pred)

            f1_list.append(f1)
            logloss_list.append(logloss)
            confusion_entries.append(cm.ravel())

        results[val] = {
            "f1": (np.mean(f1_list), np.std(f1_list)),
            "logloss": (np.mean(logloss_list), np.std(logloss_list)),
            "confusion": {
                name: (np.mean([cm[i] for cm in confusion_entries]), np.std([cm[i] for cm in confusion_entries]))
                for i, name in enumerate(["tn", "fp", "fn", "tp"])
            }
        }

    return results

def multi_param_sweep(data_path, param_grid, base_params=None, num_runs=3, num_boost_round=500, threshold=0.5):
    if base_params is None:
        base_params = {
            "tree_method": "hist",
            "device": "cuda",
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "max_depth": 8,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 0.5,
        }

    all_results = {}

    for param_name, values in param_grid.items():
        print(f"\n=== Sweeping hyperparameter: {param_name} ===")
        sweep_results = sweep_hyperparameter(
            data_path=data_path,
            param_name=param_name,
            values=values,
            base_params=base_params,
            num_runs=num_runs,
            num_boost_round=num_boost_round,
            threshold=threshold
        )
        all_results[param_name] = sweep_results

    return all_results


def format_combined_latex_table(results_dict, param_name):
    """
    Create one LaTeX table for a single hyperparameter sweep, showing all metrics.
    """
    header = (
        "\\begin{tabular}{c|c|c|c|c|c|c}\n"
        f"{param_name} & F1 & LogLoss & TN & FP & FN & TP \\\\\n"
        "\\hline"
    )
    lines = [header]

    for val, metrics in results_dict.items():
        f1_mean, f1_std = metrics["f1"]
        ll_mean, ll_std = metrics["logloss"]

        tn_mean, tn_std = metrics["confusion"]["tn"]
        fp_mean, fp_std = metrics["confusion"]["fp"]
        fn_mean, fn_std = metrics["confusion"]["fn"]
        tp_mean, tp_std = metrics["confusion"]["tp"]

        line = (
            f"{val} & "
            f"${f1_mean:.4f} \\pm {f1_std:.4f}$ & "
            f"${ll_mean:.4f} \\pm {ll_std:.4f}$ & "
            f"${tn_mean:.1f} \\pm {tn_std:.1f}$ & "
            f"${fp_mean:.1f} \\pm {fp_std:.1f}$ & "
            f"${fn_mean:.1f} \\pm {fn_std:.1f}$ & "
            f"${tp_mean:.1f} \\pm {tp_std:.1f}$ \\\\"
        )
        lines.append(line)

    lines.append("\\end{tabular}")
    return "\n".join(lines)

def generate_combined_tables_for_all_sweeps(all_results):
    for param_name, result_dict in all_results.items():
        print(f"\n📄 Combined Table for {param_name}:")
        print(format_combined_latex_table(result_dict, param_name))


if __name__ == "__main__":
    data_path = "datasets/bbp_ds_2m_merged_v2.h5"

    param_grid = {
        "max_depth": [4, 6, 8],
        "eta": [0.01], 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "cosample_bytree": [0.6, 0.8, 1.0],
    }

    results = multi_param_sweep(
        data_path=data_path,
        param_grid=param_grid,
        num_runs=3,
        num_boost_round=500,
        threshold=0.5
    )

    generate_combined_tables_for_all_sweeps(results)

    with open('outputs/xboost_results.pkl', 'wb') as handle:
        pickle.dump(results, handle)
