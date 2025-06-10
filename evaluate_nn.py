import os
import pickle
import errno
import glob
import torch
from tqdm.auto import tqdm
from scipy.stats import ttest_ind
import re
from critic import Critic
from model import RandomDataModule, RandomModel

def get_checkpoint_path(identifier, prefix="outputs/berlinpro", suffix="checkpoints"):
    """
    Get the checkpoint file path with the highest step for a given identifier.
    
    Args:
        identifier (str): The identifier for the checkpoint folder.
        prefix (str): The base directory where checkpoints are stored.
        suffix (str): The subdirectory containing checkpoint files.
        
    Returns:
        str: Full path to the checkpoint file with the highest step, or None if no files found.
    """
    base_path = os.path.join(prefix, identifier, suffix)
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), base_path)
    
    highest_step = -1
    highest_ckpt = None
    step_pattern = re.compile(r"step=(\d+)\.ckpt$")
    
    for file_name in os.listdir(base_path):
        match = step_pattern.search(file_name)
        if match:
            step = int(match.group(1))
            if step > highest_step:
                highest_step = step
                highest_ckpt = file_name
    
    if highest_ckpt:
        return os.path.join(base_path, highest_ckpt)
    return None


def evaluate_model_dict_to_result_dict(model_dict):
    result_dict = {}
    for scenario_name, scenario_subset in tqdm(metrics_dict.items()):
        result_dict[scenario_name] = {model_key: evaluate(model, scenario_subset) for model_key, model in model_dict.items()}
    return result_dict

@staticmethod
def significant_confidence_levels(group_A, group_B, confidence=0.99):
    ci = ttest_ind(group_A.flatten().cpu(), group_B.flatten().cpu(), equal_var=False).confidence_interval(confidence_level=confidence)
    confidence_interval = (ci.low.item(), ci.high.item())
    return not (confidence_interval[0] < 0. and confidence_interval[1] > 0.), confidence_interval


def statistics(result_dict, reference_key="ref"):
    min_mean = float('inf')
    statistics_dict = {}
    for key, value in result_dict.items():
        total_loss = value.mean(dim=1)
        mean = total_loss.mean()
        statistics_dict[key] = (mean.item(), total_loss.std().item())
        if mean < min_mean:
            min_mean_key = key
            min_mean = mean

    for key, value in result_dict.items():
         statistics_dict[key] =  statistics_dict[key] + (key==min_mean_key,) + significant_confidence_levels(value, result_dict[reference_key])
         diff = (result_dict[key] - result_dict[min_mean_key]).flatten().abs().cpu()
         mean = torch.mean(diff)
         std_dev = torch.std(diff)
    return statistics_dict

def model_paths_to_model_dict(model_paths):
    models_dict = {}
    for key, identifier in model_paths.items():
        path = get_checkpoint_path(identifier)
        models_dict[key] = RandomModel.load_from_checkpoint(
        checkpoint_path=path,
        map_location=None,
        )
    return models_dict

def evaluate(seed_shift, model, critic_net, num_samples=100000):
    output_list = []
    batch_size = 32
    seed = 42+50000000*seed_shift
    datamodule = RandomDataModule(num_samples, model.input_dim, model.output_dim, batch_size, seed, device=critic_net.model.device, val_samples=100000, val_seed=seed+20000000)

    for state in tqdm(datamodule.train_dataloader(), leave=False):
        with torch.no_grad():
            policy_action = model(state)
            output_list.append(critic_net(policy_action, state))
    if len(output_list) > 0:
        output_tensor = torch.cat(output_list)
    return output_tensor

def evaluate_model_dict_to_result_dict(model_dict, num_samples=100000):
    result_dict = {}
    for i, (key, model) in tqdm(enumerate(model_dict.items()), total=len(model_dict)):
        result_dict[key] = evaluate(i, model, critic_net, num_samples)
    return result_dict

def scientific(value: float, precision: int = 2) -> str:
    """
    Convert a float to a LaTeX-formatted scientific notation string.

    Args:
        value (float): The number to format.
        precision (int): Number of decimal places for the base.

    Returns:
        str: LaTeX-formatted string, e.g., "$1.03 \\times 10^{-2}$"
    """
    sci_str = f"{value:.{precision}e}"
    base, exp = sci_str.split("e")
    exp = exp.lstrip("+")
    if exp.startswith("-"):
        exp = "-" + exp[1:].lstrip("0")
    else:
        exp = exp.lstrip("0")
    exp = exp or "0"
    return f"{base} \\times 10^{{{exp}}}"
    
@staticmethod
def result_dict_to_latex(statistics_dict, reference_key="ref", statistics_table=True):
    subtrahent = 0 if statistics_table else 1
    if len(result_dict) < 4:
        alignment = "l" * (len(statistics_dict)-subtrahent)
        table_environment = "tabular"
    else:
        
        alignment = r"""*{"""+str(len(statistics_dict)-subtrahent)+r"""}{>{\centering\arraybackslash}X}"""
        table_environment = "tabularx"
    
    if table_environment =="tabularx":
        text_width =  r"""{\textwidth}"""
    else:
        text_width = ""

    output_string = (
        r"""
    \begin{"""+table_environment+r"""}"""+text_width+r"""{p{2cm}|"""+
    alignment    
        + r"""}
    \hline"""
        + "\n"
    )

    keys = statistics_dict.keys()
    
    output_string += r"Model & \ac{MSE} $\pm\sigma$"
    if statistics_table:
        output_string += r"& (\acs{CI})"
    output_string += r"\\" + "\n" 
    output_string += r"\hline" + "\n"
    
    for model_key, (mean, std_dev, is_best, is_significant, p_value) in statistics_dict.items():
        model_row_element = scientific(mean)
        if is_best:
            model_row_element = r"\mathbf{" + model_row_element + r"}"
        model_row_element = "$"+model_row_element+r" \pm "+scientific(std_dev)+"$ "
        if not model_key == reference_key:
            if statistics_table:
                model_row_element += "& $(" + scientific(p_value[0]) + "," + scientific(p_value[1]) + ")$"
            if is_significant:
                model_row_element += r"$\dagger$"
        else:
            if statistics_table:
                model_row_element += "& ---"
        #print(model_row_element)
        output_string += model_key + " & " + model_row_element + r" \\" + "\n"

    output_string += r"""\hline
    \end{"""+table_environment+r"""}"""
    return output_string

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic_net = Critic(device=device)
    model_paths = {
        "Reference": "2vzv2nk0",
        "ScaledSigmoid": "mwjxf0st",
        "Lin_4": "ehu98hh8",
        "Lin_5": "hhea93wl",
        "Log_4": "5wj45jrp",
        "Mish": "33vt5lny",
        "Small": "mb1yxtff",
        "Big": "iwm6scit",
        "BatchNorm": "1ckq79p7",
        "BS_16": "0sirvbb5",
        "BS_64": "j6t73cyh",
        "LR_1e-3": "1ncaidn5",
        "LR_1e-5": "5ln6pbe1",
        "L2": "959mgp0l",
        "AdamW": "2xh15ihg",
        "Plat_3": "5dwghjbi",
        "Plat_5": "vv4ng2p2",
         }
    
    model_dict = model_paths_to_model_dict(model_paths)
    result_dict = evaluate_model_dict_to_result_dict(model_dict, num_samples=10)
    
    with open("outputs/result_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)
    
    statistics_dict = statistics(result_dict)
    print(result_dict_to_latex(statistics_dict))
