from surrogate import BerlinPro2
from evaluate_nn import model_paths_to_model_dict
import lightning
import pickle

def make_latex_table_from_metrics(results_dict, prefix="feature_rmse/"):
    import re

    name_map = {
        'feature_rmse/Horizontal_beam_size_[mm]': r'$s_x$ [mm]',
        'feature_rmse/Vertical_beam_size_[mm]': r'$s_y$ [mm]',
        'feature_rmse/Horizontal_beam_position_[mm]': r'$p_x$ [mm]',
        'feature_rmse/Vertical_beam_postion_[mm]': r'$p_y$ [mm]',
        'feature_rmse_<_30/Horizontal_beam_size_[mm]': r'$s_x$ [mm]',
        'feature_rmse_<_30/Vertical_beam_size_[mm]': r'$s_y$ [mm]',
        'feature_rmse_<_30/Horizontal_beam_position_[mm]': r'$p_x$ [mm]',
        'feature_rmse_<_30/Vertical_beam_postion_[mm]': r'$p_y$ [mm]',
    }

    def clean_metric_name(name):
        return name_map.get(name, name.replace('_', r'\_'))

    model_names = list(results_dict.keys())
    first_model_metrics = results_dict[model_names[0]][0]

    base_metrics = [
        k for k in first_model_metrics
        if k.startswith(prefix) and not k.endswith('_std')
    ]

    column_names = [clean_metric_name(k) for k in base_metrics]
    header = ['Model'] + column_names

    # Begin LaTeX table
    table = "\\begin{tabular}{l" + "c" * len(column_names) + "}\n"
    table += "\\toprule\n"
    table += " & ".join(header) + " \\\\\n"
    table += "\\midrule\n"

    for model in model_names:
        row_metrics = results_dict[model][0]
        row = [model]
        for k in base_metrics:
            if k in row_metrics:
                mean = row_metrics[k]
                std = row_metrics.get(f"{k}_std", 0.0)
                val_str = f"${mean:.3f} \\pm {std:.3f}$"
            else:
                val_str = "---"  # Placeholder if metric is missing
            row.append(val_str)
        table += " & ".join(row) + " \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"

    return table



def eval_model_paths(model_paths):
    model_dict = model_paths_to_model_dict(model_paths, prefix="outputs/berlinpro_surrogate/berlinpro_surrogate", model_class=BerlinPro2)
    result_dict = {}
    for key, item in model_dict.items():
        trainer = lightning.Trainer()
        result_dict[key] = trainer.test(item)
    with open('outputs/evaluate_surrogate.pkl', 'wb') as file:
        pickle.dump(result_dict, file)
    
    print(make_latex_table_from_metrics(result_dict, prefix="feature_rmse/"))
    
    print(make_latex_table_from_metrics(result_dict, prefix="feature_rmse_<_30/"))
    
if __name__ == "__main__":
    coarse_model_paths = {
        "decay_95": "2aucg3nj",
        "decay_99": "aafdqga6",
        "patience_1000": "l2j8nqm5",
        "patience_250": "zxzqaoju",
        "bs_128": "mhjyksv8",
        "bs_512": "04h5e6cj",
        "lr1e-5": "i7i94bon",
        "lr1e-4": "kwwe5kc0",
        "lr1e-3": "uqd8xout",
        "layers_3": "xff6j98d",
        "layers_10": "8b63drv1",
        "layers_15": "oocb5yj5",
        "blow_100": "6lybs98e",
        "blow_200": "rijhlctg",
        "shrink_lin": "gziloeus",
         }
    eval_model_paths(coarse_model_paths)
         
    fine_model_paths = {
        "decay_95": "iaikkt43",
        "decay_99": "9is16kyc",
        "patience_1000": "1okzr8ya",
        "patience_250": "8mmqoqxr",
        "bs_128": "kvpvki5b",
        "bs_512": "qoo80gkn",
        "lr1e-5": "o8cv2nxc",
        "lr1e-4": "xp881xkx",
        "lr1e-3": "g2ppqwaf",
        "layers_3": "q7v3hf5q",
        "layers_10": "k06stu84",
        "layers_15": "mt65l5zc",
        "blow_100": "ji3v2cme",
        "blow_200": "ogc5jqhn",
        "shrink_lin": "jb7ne3v8",
         }
    eval_model_paths(fine_model_paths)
    
