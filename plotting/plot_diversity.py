import sys
import os
current_script_path = os.path.abspath(__file__)
plotting_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(plotting_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import argparse
from matplotlib import cm
import matplotlib.ticker as mticker
from datasets import load_dataset
from matplotlib.colors import Normalize
from matplotlib.colors import rgb2hex
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from src.evaluate_models import (
    inference_scaling_aggregator_average,
    inference_scaling_aggregator_majority_voting,
    inference_scaling_aggregator_exact_match,
    compute_similarity
)

from matplotlib.patches import Patch
from scipy import interpolate 
def binning_by_config(
    mse_arr: np.ndarray,
    similarity_arr: np.ndarray,
    config_arr: np.ndarray,
    group_n: int = 1 
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Group and bin results according to `config_setting`.

    The first and last unique config values each form an individual bin (n = 1),
    while the intermediate unique config values are merged into consecutive bins
    of size `group_n`.

    Args:
        mse_arr: MSE values, aligned element-wise with `similarity_arr` and `config_arr`.
        similarity_arr: Cosine similarity values corresponding to each configuration.
        config_arr: Configuration identifiers used as the grouping key.
        group_n: Bin size for intermediate consecutive config groups (group_n ≥ 1).

    Returns:
        bin_similarity_mean: Mean similarity value for each bin (x-axis).
        bin_mse_mean: Mean MSE value for each bin (y-axis).
        bin_mse_std: Standard deviation of MSE within each bin (error bars).
        baselines: Median MSE value for each bin.
    """
    if not (len(mse_arr) == len(similarity_arr) == len(config_arr)):
        raise ValueError("mse_arr、similarity_arr、config_arr must match")
    if not isinstance(group_n, int) or group_n < 1:
        raise ValueError("group_n must ≥1")
    
    unique_configs = np.unique(config_arr)
    n_unique = len(unique_configs)
    if n_unique == 0:
        return np.array([]), np.array([]), np.array([]), []
    
    config_groups = []
    if n_unique == 1:
        config_groups.append([unique_configs[0]])
    else:
        config_groups.append([unique_configs[0]])
        
        middle_configs = unique_configs[1:-1]  
        n_middle = len(middle_configs)
        for i in range(0, n_middle, group_n):
            middle_group = middle_configs[i:i+group_n]
            config_groups.append(middle_group.tolist())
        
        config_groups.append([unique_configs[-1]])
    
    bin_similarity_mean = []
    bin_mse_mean = []
    bin_mse_std = []
    baselines = []
    len_g = len(config_groups)

    for k , config_group in enumerate(config_groups):
        group_mask = np.zeros_like(config_arr, dtype=bool)
        for config in config_group:
            group_mask |= (config_arr == config)  
        
        similarity_in_group = similarity_arr[group_mask]
        mse_in_group = mse_arr[group_mask]
        
        if len(mse_in_group) == 0:
            continue
        
        mse_in_group=np.delete(np.array(mse_in_group), np.argmax(mse_in_group))

        if k==len_g-1 or k==0:
            mse_mean = np.max(mse_in_group)
        else:
            mse_mean = np.percentile(mse_in_group, q=25, interpolation='linear')

        sim_mean = np.mean(similarity_in_group)
        mse_std = np.std(mse_in_group)
        mse_max = np.max(mse_in_group)

        bin_similarity_mean.append(sim_mean)
        bin_mse_mean.append(mse_mean)
        bin_mse_std.append(mse_std)
        baselines.append(mse_max)
    
    bin_similarity_mean = np.array(bin_similarity_mean)
    bin_mse_mean = np.array(bin_mse_mean)
    bin_mse_std = np.array(bin_mse_std)
    
    if len(bin_similarity_mean) > 0:
        sorted_idx = np.argsort(bin_similarity_mean)
        bin_similarity_mean = bin_similarity_mean[sorted_idx]
        bin_mse_mean = bin_mse_mean[sorted_idx]
        bin_mse_std = bin_mse_std[sorted_idx]
        baselines = [baselines[idx] for idx in sorted_idx]
    
    return bin_similarity_mean, bin_mse_mean, bin_mse_std, baselines


def plot_perturbation_curve(
    task_name: str,
    similarities: list[list[float]],  
    perturb_perfs: list[list[float]],  
    perturb_stds: list[list[float]], 
    none_baselines: list[float], 
    curve_labels: list[str] = None,
    y_label: str = "MSE",
    o_dir: str = ""
):
    SMOOTHING_SIGMA = 1
    
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['errorbar.capsize'] = 5
    plt.rcParams['legend.frameon'] = True 
    plt.rcParams['legend.framealpha'] = 0.9 
    plt.rcParams['legend.fontsize'] = 10 

    n_curves = len(similarities)
    if n_curves == 0:
        raise ValueError("Input data must not be empty; at least one curve is required")
    data_groups = [perturb_perfs, perturb_stds, none_baselines]
    data_names = ["perturb_perfs", "perturb_stds", "none_baselines"]
    for idx, (data, name) in enumerate(zip(data_groups, data_names)):
        if len(data) != n_curves:
            raise ValueError(f"Length mismatch: {name} has length {len(data)}, "
                             f"but similarities has length {n_curves}")
    for i in range(n_curves):
        group_len = len(similarities[i])
        if len(perturb_perfs[i]) != group_len or len(perturb_stds[i]) != group_len:
            raise ValueError(
            f"Inner length mismatch in group {i + 1}: "
            f"similarities has length {group_len}, "
            f"but perturb_perfs/perturb_stds lengths do not match."
        )
    
    if curve_labels is None:
        curve_labels = [f"Curve {i+1}" for i in range(n_curves)]
    if len(curve_labels) != n_curves:
        raise ValueError(
        f"Curve label length {len(curve_labels)} does not match "
        f"the number of curves ({n_curves})."
    )

    color_palette = [
        (
         '#E1D5E7',  
         '#6A0DAD', 
         '#D32F2F', 
         '#6A0DAD'   
        ),
        (
         '#D5E8D4',  
         '#2E8B57', 
         '#FF6347',  
         '#2E8B57'   
        ),
        (
         '#DCEAFB',  
         '#4169E1',  
         '#FFD700',  
         '#A9A9A9'   
        ),
        (
         '#FFF2CC',  
         '#DAA520', 
         '#9932CC', 
         '#C0C0C0' 
        )
    ]
    def get_group_colors(idx):
        return color_palette[idx % len(color_palette)]

    all_processed_data = []

    for i in range(n_curves):
        similarity = similarities[i]
        perturb_perf = perturb_perfs[i]
        perturb_std = perturb_stds[i]

        sorted_idx = np.argsort(similarity)
        x_raw = np.array(similarity)[sorted_idx]
        y_raw = np.array(perturb_perf)[sorted_idx]
        std_raw = np.array(perturb_std)[sorted_idx]

        x_dense = np.linspace(x_raw.min(), x_raw.max(), 300)
        
        f_y = interp1d(x_raw, y_raw, kind='linear', fill_value="extrapolate")
        f_std = interp1d(x_raw, std_raw, kind='linear', fill_value="extrapolate")
        y_dense = f_y(x_dense)
        std_dense = f_std(x_dense)
        
        y_smooth_mean = gaussian_filter1d(y_dense, sigma=SMOOTHING_SIGMA)
        std_smooth = gaussian_filter1d(std_dense, sigma=SMOOTHING_SIGMA)
        
        y_smooth_upper = y_smooth_mean + std_smooth
        y_smooth_lower = y_smooth_mean - std_smooth

        all_processed_data.append({
            "x_raw": x_raw,
            "y_raw": y_raw,
            "std_raw": std_raw,
            "x_dense": x_dense,
            "y_smooth_mean": y_smooth_mean,
            "y_smooth_upper": y_smooth_upper,
            "y_smooth_lower": y_smooth_lower,
            "baseline": none_baselines[i],
            "label": curve_labels[i],
            "colors": get_group_colors(i)
        })

    fig, ax = plt.subplots(figsize=(7, 5))

    for processed_data in all_processed_data:
        color_fill, color_curve, color_data, color_baseline = processed_data["colors"]
        
        ax.fill_between(
            processed_data["x_dense"],
            processed_data["y_smooth_lower"], 
            processed_data["y_smooth_upper"], 
            color=color_fill, 
            alpha=0.4, 
            edgecolor='none'
        )

        ax.plot(
            processed_data["x_dense"],
            processed_data["y_smooth_mean"], 
            color=color_curve, 
            linewidth=2.5, 
            linestyle='-', 
            label=f"Perturbed sample on {processed_data['label']}"
        )

        ax.errorbar(
            processed_data["x_raw"],
            processed_data["y_raw"],
            yerr=processed_data["std_raw"],
            fmt='o', 
            color=color_data, 
            markersize=7, 
            elinewidth=1, 
            capsize=2,
            capthick=1,
            zorder=10,
        )

        ax.axhline(
            y=processed_data["baseline"], 
            color=color_baseline, 
            linestyle='--', 
            linewidth=2.5, 
            label=f"None perturbation on {processed_data['label']}"
        )

    ax.set_xlabel('Cosine Similarity', fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')

    all_data_min = []
    all_data_max = []
    for processed_data in all_processed_data:
        data_min = min(processed_data["y_smooth_lower"].min(), (processed_data["y_raw"] - processed_data["std_raw"]).min())
        data_max = max(processed_data["y_smooth_upper"].max(), (processed_data["y_raw"] + processed_data["std_raw"]).max())
        all_data_min.append(data_min)
        all_data_max.append(data_max)
    global_min = min(all_data_min)
    global_max = max(all_data_max)
    y_margin = (global_max - global_min) * 0.1
    y_btm = max(0, global_min - y_margin) if "MSE" in y_label else global_min - y_margin
    y_top = global_max + y_margin

    all_x_min = [processed_data["x_raw"].min() for processed_data in all_processed_data]
    all_x_max = [processed_data["x_raw"].max() for processed_data in all_processed_data]
    x_btm = min(all_x_min) - 0.002
    x_top = max(all_x_max) + 0.002
    
    ax.set_xlim(x_btm, x_top)
    ax.set_ylim(y_btm, y_top)

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.5, linestyle='-', linewidth=0.5, color='#dcdcdc')
    
    handles, labels = ax.get_legend_handles_labels()
    unique_handles_labels = []
    seen_labels = set()
    for handle, label in zip(handles, labels):
        if label not in seen_labels and label != "":
            seen_labels.add(label)
            unique_handles_labels.append((handle, label))
    ax.legend(
        [h for h, l in unique_handles_labels],
        [l for h, l in unique_handles_labels],
        loc='best', 
        frameon=True, 
        fancybox=True
    )

    plt.tight_layout()

    if len(o_dir) > 0:
        os.makedirs(o_dir, exist_ok=True)
        task_path = os.path.join(o_dir, task_name)
    else:
        task_path = task_name

    plt.savefig(f'{task_path}_multi_smooth_perturbation_curve.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{task_path}_multi_smooth_perturbation_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

disturbance_types_9 = ['prefix', 'suffix', 'insert', 'gaussian_noise', 'random_offset_noise', 'missing_data', "task_dependent", 'task_sensitive', 'task_reconstruct']

def get_result(ori_path, model_name, number_samples, disturbance_types=[], lens=[512], temperature=[0.7]):
    print("\n==============EVALUATION!!================")
    all_result={}
    for l in lens:
        for t in temperature:
            output_data_path=ori_path+f"_l{str(l)}_t{str(t)}"
            if os.path.exists(output_data_path):
                # read baseline
                baseline_path=os.path.join(output_data_path, model_name)
                if os.path.exists(baseline_path):
                    baseline_results=[]
                    json_count = len([f for f in os.scandir(baseline_path)])
                    if json_count==0:
                        continue
                    for i in tqdm(range(json_count)):
                        baseline_mse_file=os.path.join(baseline_path,f"{i+1}.json")
                        with open(baseline_mse_file, "r") as f:
                            result = json.load(f)
                            baseline_results.append(result)
                    all_result["baseline"+f"_l{str(l)}_t{str(t)}"] = baseline_results

    if os.path.exists(output_data_path):
        # read disturbed data
        for d_type in disturbance_types:
            results=[]
            disturbed_path=os.path.join(output_data_path, d_type)
            if os.path.exists(disturbed_path):
                print("read: ", d_type)
                for i in tqdm(range(number_samples)):
                    mse_file=os.path.join(disturbed_path,str(i),model_name,"1.json")
                    if not os.path.exists(mse_file):
                        break
                    with open(mse_file, "r") as f:
                        result = json.load(f)
                        results.append(result)
                all_result[d_type] = results
    return all_result


# python plot_output.py -i /root/RobustTSFM/results/predictions_ETTh1 -m chronos-t5-tiny -n 128
def main():
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--data', '-d', type=str, default = "ETTh1")
    parser.add_argument('--output_dir','-dir', type=str, default = "./results_when")
    parser.add_argument('--models', '-m', type=str, default="chronos-t5-tiny,chronos-t5-base")
    parser.add_argument('--save_dir', '-s', type=str, default="./plots")
    parser.add_argument('--temperatures', '-t', type=str, default="0.7")
    parser.add_argument('--length', '-l', type=str, default="512")
    parser.add_argument('--disturbance', '-p', type=str, default="gaussian_noise")
    parser.add_argument('--number','-n', type=int, default = 64)
    parser.add_argument('--seed','-seed', type=int, default = 4)
    args = parser.parse_args()
    length=args.length.split(',')#[32,64,128,256,512,1024]
    temperatures=args.temperatures.split(',') #[0.7]
    disturbance_types=args.disturbance.split(',')  #['suffix',  'missing_data', 'prefix', 'insert', 'gaussian_noise', 'random_offset_noise']  #[]   #
    disturbance_types = [d for d in disturbance_types if d in disturbance_types_9 ]
    models = args.models.split(',')#["chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small"]     #
    metric="MSE"
    similarity_key="cosine_similarity"
    config_key = "eta" if args.disturbance=="gaussian_noise" else "rho"
    save_dir=args.save_dir+args.data

    all_data={}
    for model in models:
        all_data[model]={
            "mse": [],
            "similarity": [],
            "config": []
        }

    for x in range(1,args.seed+1):
        output_data_path=args.output_dir+str(x)+"/predictions_"+args.data
        
        print("read:", output_data_path)
        
        all_result={}
        for model in models:
            all_result = get_result(output_data_path, model, args.number, disturbance_types,lens=length, temperature=temperatures)
            for key in all_result:
                cur_all_result = all_result[key]    # dict_keys(['pred', 'gt', 'MSE', 'MAE', 'RMSE'])
                similarity = compute_similarity(cur_all_result, is_print=False)
                all_data[model]["similarity"].extend(similarity["individual_samples"][similarity_key])
                for sample in cur_all_result:
                    all_data[model]["mse"].append(sample[metric])
                    all_data[model]["config"].append(sample["config"][config_key])

    similarities, perturb_perfs, perturb_stds, none_baselines=[],[],[],[]
    for model in models:
        mse = all_data[model]["mse"]
        config = all_data[model]["config"]
        similarity = all_data[model]["similarity"]

        similarity_result=np.array(similarity)
        mse_result=np.array(mse)
        config_setting=np.array(config)

        sorted_idx = np.argsort(similarity_result)
        similarity_result = similarity_result[sorted_idx]
        mse_result = mse_result[sorted_idx]
        config_setting = config_setting[sorted_idx]

        bin_similarity, bin_mse_mean, bin_mse_std, baselines = binning_by_config(
            mse_result, similarity_result,
            config_arr=config_setting
        )
######################
        bin_similarity=bin_similarity[9:]
        bin_mse_mean=bin_mse_mean[9:]
        bin_mse_std=bin_mse_std[9:]
######################

        similarities.append(bin_similarity.tolist())
        perturb_perfs.append(bin_mse_mean.tolist())
        perturb_stds.append(bin_mse_std.tolist())
        none_baselines.append(baselines[-1])
    print(none_baselines)

    plot_perturbation_curve(
        task_name=args.disturbance,
        similarities=similarities,
        perturb_perfs=perturb_perfs,
        perturb_stds=perturb_stds,
        none_baselines=none_baselines,
        curve_labels=models,
        y_label="MSE",
        o_dir=save_dir
    )
if __name__ == "__main__":
    main()
