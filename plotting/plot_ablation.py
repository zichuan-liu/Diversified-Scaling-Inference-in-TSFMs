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
import copy
from matplotlib import cm
import matplotlib.ticker as mticker
from datasets import load_dataset
from matplotlib.colors import Normalize
from matplotlib.colors import rgb2hex
from matplotlib.colors import Normalize, LinearSegmentedColormap

from src.evaluate_models import (
    inference_scaling_aggregator_average,
    inference_scaling_aggregator_majority_voting,
    inference_scaling_aggregator_exact_match,
    compute_similarity
)
disturbance_types_9 = ['prefix', 'suffix', 'insert', 'gaussian_noise', 'random_offset_noise', 'missing_data', "task_dependent", 'task_sensitive', 'task_reconstruct']

def get_result(ori_path,data, model_name, number_samples, disturbance_types=[], lens=[512], temperature=[0.7]):
    print("\n==============EVALUATION!!================")
    all_result={}
    for l in lens:
        for t in temperature:
            output_data_path=ori_path+data+f"_l{str(l)}_t{str(t)}"
            if os.path.exists(output_data_path):
                # read baseline
                baseline_path=os.path.join(output_data_path, model_name)
                if os.path.exists(baseline_path):
                    baseline_results=[]
                    json_count = min(number_samples,len([f for f in os.scandir(baseline_path)]))
                    if json_count==0:
                        continue
                    for i in tqdm(range(json_count)):
                        baseline_mse_file=os.path.join(baseline_path,f"{i+1}.json")
                        with open(baseline_mse_file, "r") as f:
                            result = json.load(f)
                            baseline_results.append(result)
                    all_result["baseline"+f"_l{str(l)}_t{str(t)}"] = baseline_results

            for d_type in disturbance_types:
                disturbed_path=ori_path+"{"+d_type+"}_"+data+f"_l{str(l)}_t{str(t)}"
                results=[]
                if os.path.exists(disturbed_path):
                    print("read: ", disturbed_path)
                    for i in tqdm(range(number_samples)):
                        mse_file=os.path.join(disturbed_path,d_type,str(i),model_name,"1.json")
                        if not os.path.exists(mse_file) and model_name=="chronos-t5-tiny":
                            mse_file=os.path.join(output_data_path,d_type,str(i),model_name,"1.json")
                        if not os.path.exists(mse_file):
                            continue
                        with open(mse_file, "r") as f:
                            result = json.load(f)
                            results.append(result)
                    all_result[d_type+f"_l{str(l)}_t{str(t)}"] = results
    return all_result

def fit_powerlaw(x, y, signal='A'):
    logx = np.log10(x)
    logy = np.log10(y)
    
    coeffs = np.polyfit(logx, logy, 1)
    poly = np.poly1d(coeffs)
    logy_fit = poly(logx)
    
    k, b = coeffs
    x0 = 10**(-b/k)
    x0_sci = f'{x0:.1e}'
    mantissa, exponent = x0_sci.split('e')
    mantissa = float(mantissa)
    exponent = int(exponent)
    label = r'$L(%s) = \left({%.1f} \cdot 10^{%d} / %s  \right)^{%.3f}$' % (signal, mantissa, exponent, signal, -k)
    
    return logy_fit, label

MODELSIZE2PARAM={
    '1B': 1.03*1e9,
    '16B': 15.7 * 1e9,
}
def step_to_compute(size):
    B = 128
    N = MODELSIZE2PARAM[size]
    L = 512
    step = np.arange(1, 101)
    C = 6 * B * N * L
    return C * step / (8.64*1e19)

def plot_temp_ensemble(
    data,
    agg_types,
    model_name="chronos-t5-tiny",
    metric="MSE",
    save_dir="./plots"
):
    plt.switch_backend('Agg')
    os.makedirs(save_dir, exist_ok=True)
    
    plt.rcParams['font.size'] = 12  
    plt.rcParams['axes.linewidth'] = 1.2  
    plt.rcParams['xtick.major.width'] = 2.0  
    plt.rcParams['ytick.major.width'] = 2.0  
    plt.rcParams['grid.linewidth'] = 1.0  
    plt.rcParams['legend.frameon'] = True  
    plt.rcParams['legend.framealpha'] = 1.0 
    plt.rcParams['legend.edgecolor'] = 'black'  
    CUSTOM_COLORS = [
        '#f9ba02',  
        '#fbeb28', 
        '#00723c'
    ]
    temp_list = []
    for disturb_type in data.keys():
        try:
            temp = float(disturb_type.split("t")[-1])
            temp_list.append(temp)
        except (IndexError, ValueError):
            temp_list.append(1.2)
    temp_min, temp_max = 0.0, 1.2
    norm = Normalize(vmin=temp_min, vmax=temp_max)
    cmap = LinearSegmentedColormap.from_list('custom_gradient', CUSTOM_COLORS, N=64)

    for agg_type in agg_types:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xscale('log', base=2)
        
        sorted_disturbs = sorted(
            data.items(),
            key=lambda x: float(x[0].split("t")[-1]) if "t" in x[0] else 1.2
        )
        
        for idx, (disturb_type, disturb_data) in enumerate(sorted_disturbs):
            mse_list_full = disturb_data.get(agg_type)
            if not mse_list_full:
                print(f"Warning: Skipping {disturb_type} for {agg_type} due to missing data.")
                continue
                
            mse_array = np.array(mse_list_full)
            
            x_full = np.array(range(1, len(mse_list_full) + 1))
            
            try:
                temp = float(disturb_type.split("t")[-1])
            except (IndexError, ValueError):
                temp = 1.2
            
            color = cmap(norm(temp))
            linewidth = 2.5
            is_baseline=True if "baseline" in disturb_type else False
            is_gauss = True if "gaussian" in disturb_type else False
            if is_baseline:
                linestyle="--"
                mylabel="None Perturbation"
            elif is_gauss:
                linestyle="-"
                mylabel="Gaussian Noise"
            else:
                linestyle="-"
                mylabel="Task Dependency"

            if temp==1.2:
                ax.plot(
                    x_full,
                    mse_array,
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    label=mylabel,  
                    alpha=0.95,
                    zorder=5 + idx
                )
            else:
                ax.plot(
                    x_full,
                    mse_array, 
                    color=color,
                    linewidth=linewidth,
                    linestyle="-" if not is_baseline else "-.",
                    alpha=0.95,
                    zorder=5 + idx
                )
            
            if len(mse_array) > 0:
                min_index = np.argmin(mse_array)
                min_x = x_full[min_index]
                min_y = mse_array[min_index]
                ax.scatter(
                    [min_x],
                    [min_y],
                    marker='*', 
                    s=150,      
                    color=color,
                    edgecolors='black', 
                    zorder=10 + idx 
                )

        agg_title = agg_type.replace("_", " ").title()
        ax.set_xlabel("Number of Samples", fontsize=14, fontweight='bold')
        ax.set_ylabel(f"{metric} ({agg_title})", fontsize=14, fontweight='bold')
        
        max_len = max([len(dt.get(agg_type, [0])) for dt in data.values()])
        max_pow = int(np.floor(np.log2(max_len)))
        x_tick_pows = [2**k for k in range(max_pow + 1)]
        ax.set_xticks(x_tick_pows) 
        ax.set_xticklabels([f"$2^{k}$" for k in range(len(x_tick_pows))], fontsize=13)

        all_mse_full = []
        for dt in data.values():
            all_mse_full.extend(dt[agg_type])
        y_min = max(0, np.min(all_mse_full) * 0.95)
        y_max = np.max(all_mse_full) * 1.01
        ax.set_ylim(y_min, y_max)
        
        ax.tick_params(axis='y', labelsize=13) 
        ax.grid(True, axis='y', alpha=0.3, linestyle='-.')
        ax.grid(False, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(2.0)
        ax.legend(
            fontsize=10,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            framealpha=1.0,
            shadow=False,
            borderpad=0.8,
            edgecolor='black',
            title_fontsize=13
        )
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
        cbar = fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            label="Temperature"
        )
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Temperature", fontsize=11, fontweight='bold',rotation=270,labelpad=15)
        plt.tight_layout(rect=[0, 0, 0.9, 1]) 
        save_filename = f"{model_name.lower().replace('-', '_')}_temp_ensemble_{agg_type}.pdf"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none'
        )
        save_filename = f"{model_name.lower().replace('-', '_')}_temp_ensemble_{agg_type}.png"
        save_path = os.path.join(save_dir, save_filename)
        fig.savefig(
            save_path, 
            dpi=400, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
        
        print(f"Save path：{save_path}")

COLORS = ["#9bbf8a", '#82afda', '#f79059', '#c2bdde', '#8dcec8', '#3480b8', '#ffbe7a']
BASELINE_COLOR = 'black'
BASELINE_STYLE = '--'

def plot_models_ensemble(
    data,
    agg_types,
    metric="MSE",
    save_dir="./plots"
):
    plt.switch_backend('Agg')
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5, 'legend.edgecolor': 'black'})
    
    CUSTOM_COLORS = [
        '#3abbc9',  
        '#9bca3e', 
        '#feeb51',
        '#ffb92a',
        '#ed5314'
    ]
    
    MODEL_SIZE_MAP = {
        "chronos-t5-tiny": 8,     
        "chronos-t5-mini": 20,     
        "chronos-t5-small": 46,    
        "chronos-t5-base": 200,     
        "chronos-t5-large": 710,    
        "timesfm-2.5-200m-pytorch": 200, 
        "timesfm-2.0-500m-pytorch": 500,
        "moirai-1.1-r-small":55,
        "moirai-1.1-r-base":365,
        "moirai-1.1-r-large":1240,
        "timemoe-50m":113, 
        "timemoe-200m":453,  
    }
    
    MODEL_NAMES = list(data.keys()) 
    ordered_model_sizes = sorted([
        MODEL_SIZE_MAP[name.lower().strip()] for name in MODEL_NAMES if name.lower().strip() in MODEL_SIZE_MAP
    ])
    x_ticks_equal_spaced = np.arange(1, len(ordered_model_sizes) + 1)
    x_tick_labels = [
        f"{size:.0f}M" if size < 1000 else f"{size/1000:.1f}B" 
        for size in ordered_model_sizes
    ]
    model_size_to_position = {size: pos for pos, size in zip(x_ticks_equal_spaced, ordered_model_sizes)}
    
    plot_data = {}
    min_points = {} 
    max_n = 0
    for model_name, model_data in data.items():
        clean_model_name = model_name.lower().strip()
        model_size_m = MODEL_SIZE_MAP.get(clean_model_name, None)
        if model_size_m is None:
            continue 
        x_position_equal = model_size_to_position[model_size_m]

        for agg_type in agg_types:
            mse_list = model_data.get(agg_type, []) 
            if not mse_list:
                continue
            n = len(mse_list)
            max_n = max(max_n, n)
            
            min_val = np.min(mse_list)
            first_min_idx = mse_list.index(min_val) 
            first_min_count = first_min_idx + 1     
            first_min_y = mse_list[first_min_idx]   
            
            if agg_type not in plot_data:
                plot_data[agg_type] = []
            if agg_type not in min_points:
                min_points[agg_type] = []
            min_points[agg_type].append({
                'x': x_position_equal,
                'y': first_min_y,
                'color_val': np.log2(first_min_count),
                'count': first_min_count, 
                'model_name': model_name
            })
            
            for i, mse_metric in enumerate(mse_list):
                sample_count = i + 1
                plot_data[agg_type].append({
                    'x': x_position_equal, 
                    'y': mse_metric,
                    'count': sample_count,
                    'color_val': np.log2(sample_count),
                    'size': 120,
                    'zorder': sample_count,  
                })
                

    cmap = LinearSegmentedColormap.from_list('custom_gradient', CUSTOM_COLORS, N=256)
    norm = Normalize(vmin=0, vmax=np.log2(max_n))

    for agg_type, points in plot_data.items():
        if not points: continue

        fig, ax = plt.subplots(figsize=(8, 6)) 

        points_sorted = sorted(points, key=lambda p: p['zorder'])
        
        x_vals = [p['x'] for p in points_sorted]
        y_vals = [p['y'] for p in points_sorted]
        c_vals = [p['color_val'] for p in points_sorted]
        s_vals = [p['size'] for p in points_sorted]
        z_vals = [p['zorder'] for p in points_sorted]
        
        for p in points_sorted:
            ax.scatter(
                p['x'], p['y'],
                c=[p['color_val']],
                s=p['size'],
                cmap=cmap,
                norm=norm,
                edgecolors='black',
                linewidths=0.3,
                alpha=0.7,
                zorder=p['zorder'] 
            )

        ax.set_xticks(x_ticks_equal_spaced) 
        ax.set_xticklabels(x_tick_labels, fontsize=12) 
        ax.set_xlim(x_ticks_equal_spaced[0] - 0.5, x_ticks_equal_spaced[-1] + 0.5)
        
        y_min = np.min(y_vals) * 0.96
        y_max = np.max(y_vals) * 1.0
        ax.set_ylim(y_min, y_max)
        
        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)  
        ax.set_xlabel("Model Size", fontsize=14, fontweight='bold')
        agg_title = agg_type.replace("_", " ").title()
        ax.set_ylabel(f"{metric} ({agg_title})", fontsize=14, fontweight='bold')

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax
        )
        cbar.set_label('Number of Samples', fontsize=12, fontweight='bold', rotation=270, labelpad=15)
        
        tick_counts = [1, 2, 4, 8, 16, 32, 64, 128]
        tick_counts = [t for t in tick_counts if t <= max_n]
        cbar.set_ticks([np.log2(t) for t in tick_counts])
        cbar.set_ticklabels([str(t) for t in tick_counts])
        
        save_filename = f"models_{MODEL_NAMES[0]}_{agg_type}.pdf"
        save_path = os.path.join(save_dir, save_filename)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        save_filename = f"models_{MODEL_NAMES[0]}_{agg_type}.png"
        save_path = os.path.join(save_dir, save_filename)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--data', '-d', type=str, default = "ETTh1")
    parser.add_argument('--output_dir','-dir', type=str, default = "./results/predictions_")
    parser.add_argument('--models', '-m', type=str, default="chronos-t5-tiny")
    parser.add_argument('--save_dir', '-s', type=str, default="./ablation/plots")
    parser.add_argument('--temperatures', '-t', type=str, default="0.7")
    parser.add_argument('--length', '-l', type=str, default="512")
    parser.add_argument('--disturbance', '-p', type=str, default="prefix,gaussian_noise,task_dependent")
    parser.add_argument('--number','-n', type=int, default = 128)
    args = parser.parse_args()

    args.output_data_path=args.output_dir
    args.save_dir=args.save_dir+args.data
    length=args.length.split(',')#[32,64,128,256,512,1024] #
    temperatures=args.temperatures.split(',') #[0.0,0.2,0.4,0.6,0.8,1.0,1.2]#
    disturbance_types=args.disturbance.split(',')  #['suffix',  'missing_data', 'prefix', 'insert', 'gaussian_noise', 'random_offset_noise']  #[]   #
    disturbance_types = [d for d in disturbance_types if d in disturbance_types_9 ]
    models = ["chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small", "chronos-t5-base", "chronos-t5-large"]     #args.models.split(',')
    RUN_ALL_AGGS=["exact_match","majority_voting",]#
    metric="MSE"
    
    if len(models)==1:
        args.model=models[0]
        all_result = get_result(args.output_data_path, args.data, args.model, args.number, disturbance_types,lens=length, temperature=temperatures)
    else:
        all_result={}
        for model in models:
            temp = get_result(args.output_data_path, args.data, model, args.number, disturbance_types,lens=length, temperature=temperatures)
            for key in temp:
                all_result[model+key] = temp[key]

    data={}
    for key in all_result:
        cur_all_result = all_result[key]    # dict_keys(['pred', 'gt', 'MSE', 'MAE', 'RMSE'])
        print("\n!! Perturbation method:", key, ", len:", len(cur_all_result))
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
            temp_data[agg] = inference_scaling_results
            print(key, agg, inference_scaling_results)
        data[key] = temp_data

    if len(temperatures)>1:
        plot_temp_ensemble(
            data=data,
            agg_types=RUN_ALL_AGGS,
            metric=metric,
            model_name=args.models,
            save_dir=args.save_dir
        )
    elif len(models)>1:
        baseline_res=""
        gaus_res=""
        dep_res=""
        pre_res=""

        for key in data:
            for agg in data[key]:
                if "baseline" in key:
                    baseline_res+=str(round(data[key][agg][-1],4))+"&"
                elif "gau" in key:
                    gaus_res+=str(round(data[key][agg][-1],4))+"&"
                elif "dep" in key:
                    dep_res+=str(round(data[key][agg][-1],4))+"&"
                elif "pre" in key:
                    pre_res+=str(round(data[key][agg][-1],4))+"&"
        print(baseline_res)
        print(gaus_res)
        print(dep_res)
        print(pre_res)

if __name__ == "__main__":
    main()