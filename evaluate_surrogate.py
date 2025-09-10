from surrogate import BerlinPro2
from evaluate_nn import model_paths_to_model_dict
import lightning

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



def eval_model_paths(model_dict):
    model_dict = model_paths_to_model_dict(model_paths, prefix="outputs/berlinpro_surrogate/berlinpro_surrogate", model_class=BerlinPro2)
        result_dict = {}
        for key, item in model_dict.items():
            trainer = lightning.Trainer()
            result_dict[key] = trainer.test(item)
        
        print(make_latex_table_from_metrics(result_dict, prefix="feature_rmse/"))
        
        print(make_latex_table_from_metrics(result_dict, prefix="feature_rmse_<_30/"))
    
if __name__ == "__main__":
    coarse_model_paths = {
        "Reference": "itjvgcf9",
        "decay_95": "abpmsgcz",
        "decay_99": "8izpysj6",
        "decay_9": "rqmh11bb",
        "patience_1000": "q9wf54fi",
        "patience_500": "3vme6uaf",
        "bs_1024": "phe9bmtn",
        "bs_512": "fe726nbo",
        "lr1e-5": "6kqms9m0",
        "lr1e-4": "5dcl96iu",
        "layers_3": "qqff5fih",
        "layers_15": "f13x79r4",
        "layers_10": "t1067zfd",
        "blow_100": "sqerbdb5",
        "blow_200": "tzy028j4",
        "shrink_lin": "jt6b67yb",
         }
    eval_model_paths(coarse_model_paths)
         
    fine_model_paths = {
        "Reference": "3yh6g4h0",
        "decay_95": "bf38qw38",
        "decay_99": "a4zsln77",
        "decay_9": "qrqlyl9b",
        "patience_1000": "3zr1oaic",
        "patience_500": "3h7hocny",
        "bs_1024": "cuju5zez",
        "bs_512": "wbiecjlb",
        "bs_32": "i3qkzpx6",
        "lr1e-5": "izpfk6il",
        "lr1e-4": "vrmk31q6",
        "layers_3": "9wefmhwx",
        "layers_15": "1lnp6rj4",
        "layers_10": "w0zo4xwk",
        "blow_100": "ib344ki4",
        "blow_200": "v0pu6pr7",
        "shrink_lin": "kc0sg1nl",
         }
    eval_model_paths(fine_model_paths)
    
