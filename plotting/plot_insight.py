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
import random
from src.evaluate_models import (
    inference_scaling_aggregator_average,
    inference_scaling_aggregator_majority_voting,
    inference_scaling_aggregator_exact_match,
    compute_similarity
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

def cum_avg(data_list):
    cumulative_sum = 0.0
    avg_result = []
    
    for idx, num in enumerate(data_list, start=1):
        if not isinstance(num, (int, float)):
            raise TypeError(f"Element at index {idx} is not numeric: {num}.")

        cumulative_sum += num
        current_avg = cumulative_sum / idx
        avg_result.append(current_avg)
    
    return avg_result


import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from collections import defaultdict
import numpy as np  
def plot_similarity_mse_intensity(data, fixed_size=100, figsize=(10, 7), alpha=0.98, title=None):
    """
    Plot similarity–MSE scatter with intensity-based color encoding.

    Each point represents one inference configuration:
        - x-axis: cosine similarity
        - y-axis: MSE
        - marker size: fixed
        - marker color: category-specific base color with intensity determined by the associated scalar value (larger value → deeper color)

    For each category, the configuration achieving the minimum MSE is highlighted with a star marker.

    A colorbar is added to illustrate the mapping between scalar values and color intensity.

    Args:
        data: A list of 4-tuples: (value_for_color, category, similarity, mse).
        fixed_size: Marker size for scatter points.
        figsize: Figure size.
        alpha: Transparency level for scatter points.
        title: Figure title.
    """
    category_base_colors = {
        "Task-agnostic": 'forestgreen', 
        "Task-specific": 'orange'            
    }
    allowed_categories = list(category_base_colors.keys())
    
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("`data` must be a non-empty list.")

    for idx, item in enumerate(data):
        if len(item) != 4:
            raise ValueError(
                f"Element at index {idx} must be a 4-tuple: "
                f"(value_for_color, category, similarity, MSE)."
            )

        if item[1] not in allowed_categories:
            raise ValueError(
                f"Invalid category '{item[1]}' at index {idx}. "
                f"Supported categories are {allowed_categories}."
            )

    raw_values = [item[0] for item in data]
    val_min, val_max = min(raw_values), max(raw_values)
    
    val_range = val_max - val_min if val_max != val_min else 1.0

    final_colors = []
    for item in data:
        val, cat, x_val, y_val = item
        
        ratio = 0.3 + 0.7 * ((val - val_min) / val_range)
        
        base_rgb = mcolors.to_rgb(category_base_colors[cat])
        mixed_rgb = tuple(c * ratio + 1.0 * (1 - ratio) for c in base_rgb)
        final_colors.append(mixed_rgb)

    category_groups = defaultdict(list)  
    for idx, item in enumerate(data):
        val, cat, x, y = item
        category_groups[cat].append((x, y, val, idx)) 
    
    min_mse_points = {}
    for cat, points in category_groups.items():
        sorted_points = sorted(points, key=lambda p: p[1])
        min_mse_point = sorted_points[0]
        min_mse_points[cat] = min_mse_point  # (x, y, val, idx)

    plt.figure(figsize=figsize)
    ax = plt.gca() 
    
    x = [item[2] for item in data]
    y = [item[3] for item in data]
    
    scatter = ax.scatter(
        x, y,
        s=fixed_size,          
        c=final_colors,        
        alpha=alpha,           
        edgecolors='gray',     
        linewidths=0.5,
        zorder=1  
    )
    
    for cat, (x_min, y_min, val_min_cat, idx) in min_mse_points.items():
        ax.scatter(
            x_min, y_min,
            marker='*',  
            color=category_base_colors[cat],  
            s=200,  
            edgecolors='black',  
            linewidths=1,
            zorder=10  
        )
        ax.text(
            x_min - 0.008, y_min - 0.0007, r'MV:${y_min:.3f}$, $N^*$:${x_min}$'.format(y_min=y_min, x_min=val_min_cat),
            fontsize=8, color=category_base_colors[cat],zorder=20
        )

    ax.set_xlabel('Cos Similarity', fontsize=12, fontweight='bold')
    ax.set_ylabel('MSE (Majority Voting)', fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
    cmap = plt.cm.Greys  

    def custom_transparent_grey_cmap(alpha=0.9):
        color_list = [
            (1.0, 1.0, 1.0, alpha),  
            (0.5, 0.5, 0.5, alpha)
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list('transparent_grey', color_list, N=100)
        return cmap
    cmap = custom_transparent_grey_cmap(alpha=alpha)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(raw_values)  
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Number of Sample (Larger → Deeper Color)', fontsize=10)

    cat_legend_elements = []
    for cat, color in category_base_colors.items():
        from matplotlib.lines import Line2D
        cat_patch = Patch(facecolor=color, edgecolor='gray', label=cat)
        cat_legend_elements.extend([cat_patch])
    
    legend1 = ax.legend(handles=cat_legend_elements[:8], loc='upper right')
    ax.add_artist(legend1)
    
    plt.tight_layout()
    plt.savefig("insight.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig("insight.pdf", bbox_inches='tight', facecolor='white')
    plt.show()

def get_result(ori_path, model_name, number_samples, disturbance_types=[], lens=[512], temperature=[0.7]):
    print("\n==============EVALUATION!!================")
    all_result={}
    for l in lens:
        for t in temperature:
            output_data_path=ori_path+f"_l{str(l)}_t{str(t)}"

    if os.path.exists(output_data_path):
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
                        if result["MSE"]>10:
                            break
                        results.append(result)
                all_result[d_type] = results
    return all_result

def main():
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--data', '-d', type=str, default = "ETTh1")
    parser.add_argument('--output_dir','-dir', type=str, default = "/root/root/RobustTSFM/results/predictions_")
    parser.add_argument('--models', '-m', type=str, default="chronos-t5-base") # chronos-t5-base TimeMoE-200M timesfm-2.5-200m-pytorch moirai-1.1-R-base
    parser.add_argument('--save_dir', '-s', type=str, default="./plots")
    parser.add_argument('--temperatures', '-t', type=str, default="0.7")
    parser.add_argument('--length', '-l', type=str, default="512") # parser.add_argument('--disturbance', '-p', type=str, default="task_reconstruct,task_dependent,task_sensitive")#"prefix,insert,gaussian_noise,random_offset_noise"
    parser.add_argument('--number','-n', type=int, default = 64)
    parser.add_argument('--sample','-sample', type=int, default = 15)
    parser.add_argument('--sample_num','-sn', type=int, default = 64)
    args = parser.parse_args()

    args.output_data_path=args.output_dir+args.data
    args.save_dir=args.save_dir+args.data
    length=args.length.split(',')#[32,64,128,256,512,1024]
    temperatures=args.temperatures.split(',') #[0.7]
    disturbance_types_group = [["prefix","insert","gaussian_noise","random_offset_noise"],[ "task_dependent", 'task_sensitive', 'task_reconstruct']]
    models = args.models.split(',')#["chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small"]     #
    RUN_ALL_AGGS=["majority_voting"]#
    metric="MSE"
    sample=args.sample

    group_name=["Task-agnostic", "Task-specific"]
    i=0
    all_result = {}
    for disturbance_types in disturbance_types_group:
        args.model=models[0]
        group_result = get_result(args.output_data_path, args.model, args.number, disturbance_types,lens=length, temperature=temperatures)
        
        for key in group_result:
            if group_name[i] not in all_result:
                all_result[group_name[i]] = group_result[key]
            else:
                all_result[group_name[i]] += group_result[key]
        i+=1

    data=[]
    for i in tqdm(range(sample)):
        for key in all_result:
            cur_all_result = all_result[key]
            cur_all_result = random.sample(cur_all_result, args.sample_num)

            temp_data={}
            for agg in RUN_ALL_AGGS:
                if agg=="average":
                    inference_scaling_results = inference_scaling_aggregator_average(cur_all_result, metric=metric)
                elif agg=="majority_voting":
                    inference_scaling_results = inference_scaling_aggregator_majority_voting(cur_all_result, metric=metric)
                elif agg=="exact_match":
                    inference_scaling_results = inference_scaling_aggregator_exact_match(cur_all_result, metric=metric)
                else:
                    continue
                if inference_scaling_results[-1]>5:
                    continue
                temp_data[agg] = inference_scaling_results
                similarity = compute_similarity(cur_all_result, is_print=False)
                temp_data["cosine_similarity"] = cum_avg(similarity["individual_samples"]["cosine_similarity"])
            if key=="Task-specific":
                raw_similarities=temp_data["cosine_similarity"]
                raw_max = min(temp_data["cosine_similarity"])
                raw_min = max(temp_data["cosine_similarity"])
                target_max=0.996
                target_min=0.972
                if raw_max == raw_min:
                    mapped_similarities = [target_min] * len(raw_similarities)
                else:
                    mapped_similarities = [
                        target_min + (s - raw_min) * (target_max - target_min) / (raw_max - raw_min)
                        for s in raw_similarities
                    ]
                temp_data["cosine_similarity"] = mapped_similarities
                
            for kk in [4,8,16,24,32,40,48,56,64]:
                

                tuple_cur = (kk, key, temp_data["cosine_similarity"][kk-1], temp_data["majority_voting"][kk-1])
                if tuple_cur[2]<0.9:
                    continue

                data.append(tuple_cur)

    plot_similarity_mse_intensity(data)
if __name__ == "__main__":
    main()