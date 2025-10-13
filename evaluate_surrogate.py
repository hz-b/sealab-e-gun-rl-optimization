from surrogate import BerlinPro2
from evaluate_nn import model_paths_to_model_dict
import lightning
import pickle

def make_latex_table_from_metrics(results_dict, prefix="feature_rmse/", multiplier=1, accuracy=2):
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
        'test_loss/test_loss': r'\ac{MSE}',
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

    # Find the best (lowest) mean value per metric
    best_values = {}
    for k in base_metrics:
        min_val = float('inf')
        for model in model_names:
            mean = results_dict[model][0].get(k, float('inf'))
            if mean < min_val:
                min_val = mean
        best_values[k] = min_val

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

                mean_str = f"{mean * multiplier:.{accuracy}f}"
                std_str = f"{std * multiplier:.{accuracy}f}"

                # Bold only the mean if it's the best
                if mean == best_values[k]:
                    val_str = f"$\\textbf{{{mean_str}}} \\pm {std_str}$"
                else:
                    val_str = f"${mean_str} \\pm {std_str}$"
            else:
                val_str = "---"
            row.append(val_str)
        table += " & ".join(row) + " \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"

    return table


def eval_model_paths(model_paths, label, load=False):
    if load:
        with open('outputs/evaluate_surrogate_'+label+'.pkl', 'rb') as file:
            result_dict = pickle.load(file)
    else:
        model_dict = model_paths_to_model_dict(model_paths, prefix="outputs/berlinpro_surrogate/berlinpro_surrogate", model_class=BerlinPro2)
        result_dict = {}
        for key, item in model_dict.items():
            trainer = lightning.Trainer()
            result_dict[key] = trainer.test(item)
        with open('outputs/evaluate_surrogate_'+label+'.pkl', 'wb') as file:
             pickle.dump(result_dict, file)
    
    print(make_latex_table_from_metrics(result_dict, prefix="feature_rmse/"))
    
    print(make_latex_table_from_metrics(result_dict, prefix="feature_rmse_<_30/"))
    
    print(make_latex_table_from_metrics(result_dict, prefix="test_loss/", multiplier=1000, accuracy=3))
    
if __name__ == "__main__":
    load = False
    coarse_model_paths = {
        r"$\mathrm{Plat}_{1\,000}$": "l2j8nqm5",
        r"$\mathrm{Plat}_{250}$": "zxzqaoju",
        r"$\mathrm{BS}_{128}$": "mhjyksv8",
        r"$\mathrm{BS}_{512}$": "04h5e6cj",
        r"$\mathrm{LR}\,10^{-5}$": "i7i94bon",
        r"$\mathrm{LR}\,10^{-4}$": "kwwe5kc0",
        r"$\mathrm{LR}\,10^{-3}$": "uqd8xout",
        r"$\mathrm{Log}_3$": "xff6j98d",
        r"$\mathrm{Log}_{10}$": "8b63drv1",
        r"$\mathrm{Log}_{15}$": "oocb5yj5",
        r"Small": "6lybs98e",
        r"Big": "rijhlctg",
        r"$\mathrm{Lin}_5$ ": "gziloeus",
         }
    eval_model_paths(coarse_model_paths, "coarse", load=load)
         
    fine_model_paths = {
        r"$\mathrm{Plat}_{1\,000}$": "1okzr8ya",
        r"$\mathrm{Plat}_{250}$": "8mmqoqxr",
        r"$\mathrm{BS}_{128}$": "kvpvki5b",
        r"$\mathrm{BS}_{512}$": "qoo80gkn",
        r"$\mathrm{LR}\,10^{-5}$": "o8cv2nxc",
        r"$\mathrm{LR}\,10^{-4}$": "xp881xkx",
        r"$\mathrm{LR}\,10^{-3}$": "g2ppqwaf",
        r"$\mathrm{Log}_3$": "q7v3hf5q",
        r"$\mathrm{Log}_{10}$": "k06stu84",
        r"$\mathrm{Log}_{15}$": "mt65l5zc",
        r"Small": "ji3v2cme",
        r"Big": "ogc5jqhn",
        r"$\mathrm{Lin}_5$": "jb7ne3v8",
         }
    eval_model_paths(fine_model_paths, "fine", load=load)
    
