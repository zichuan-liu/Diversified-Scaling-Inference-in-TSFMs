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

from src.evaluate_models import (
    inference_scaling_aggregator_average,
    inference_scaling_aggregator_majority_voting,
    inference_scaling_aggregator_exact_match,
    compute_similarity
)
disturbance_types_9 = ['prefix', 'suffix', 'insert', 'gaussian_noise', 'random_offset_noise', 'missing_data', "task_dependent", 'task_sensitive', 'task_reconstruct']
disturbance_types_9_map={
    'prefix': "Prefix Padding", 
    'insert': "Insert Padding", 
    'gaussian_noise': "Gaussian Noise", 
    'random_offset_noise': "Random Offset", 
    'task_sensitive': "Task Sensitivity", 
    'task_dependent': "Task Dependency", 
    'task_reconstruct': "Task Reconstruction", 
}
data_map={
    "ETTh1": "ETTh1",
    "traffic": "Traffic",
    "electricity": "Electricity",
}
def get_result(ori_path, model_name, number_samples, disturbance_types=[], lens=[512], temperature=[0.7]):
    print("\n==============EVALUATION!!================")
    all_result={}
    for l in lens:
        for t in temperature:
            output_data_path=ori_path+f"_l{str(l)}_t{str(t)}"
            if os.path.exists(output_data_path):
                baseline_path=os.path.join(output_data_path, model_name)
                if os.path.exists(baseline_path):
                    baseline_results=[]
                    json_count = len([f for f in os.scandir(baseline_path)])
                    if json_count==0:
                        continue
                    json_count=min(json_count,number_samples)
                    for i in tqdm(range(json_count)):
                        baseline_mse_file=os.path.join(baseline_path,f"{i+1}.json")
                        with open(baseline_mse_file, "r") as f:
                            result = json.load(f)
                            baseline_results.append(result)
                    all_result["baseline"+f"_l{str(l)}_t{str(t)}"] = baseline_results

    if os.path.exists(output_data_path):
        for d_type in disturbance_types:
            results=[]
            disturbed_path=os.path.join(output_data_path, d_type)
            if os.path.exists(disturbed_path):
                print("read: ", d_type)
                for i in tqdm(range(number_samples)):
                    mse_file=os.path.join(disturbed_path,str(i),model_name,"1.json")
                    with open(mse_file, "r") as f:
                        result = json.load(f)
                        results.append(result)
                all_result[d_type] = results
    return all_result

COLORS = [
    '#c6c71c',  
    '#7aaa18',  
    '#2a9513',  
    '#00723c',  
    '#fbeb28', 
    '#f1d520',  
    '#FC9171', 
]

BASELINE_COLOR = '#515151'
BASELINE_STYLE = '--'

def plot_ensemble(
    data,
    agg_type,    
    model_name,  
    metric,       
    save_dir,    
    ax,           
    is_legend_plot=False, 
    global_y_lim=None,   
    dataset_name=None     
):
    
    if not data or not agg_type:
        return [], []
    
    baseline_y_val = None
    baseline_label = 'None Perturbation' 

    first_disturb = next(iter(data.values()))
    
    if agg_type not in first_disturb:
          return [], []
          
    num_samples_original = len(first_disturb[agg_type])
    max_pow = int(np.floor(np.log2(num_samples_original)))
    x_tick_pows = [2**k for k in range(max_pow + 1)]
    x_full = range(1, num_samples_original + 1)
    
    sorted_disturbs = sorted(data.items(), key=lambda x: x[0] == "baseline") 
    
    legend_handles = []
    legend_labels = []

    color_idx = 0
    min_y,max_y=1000,0
    for disturb_type, disturb_data in sorted_disturbs:
        mse_list_full = disturb_data.get(agg_type) 
        if not mse_list_full:
            continue
        
        if "baseline" in disturb_type:
            baseline_y_val = disturb_data[agg_type][0]
            if baseline_y_val is not None:
                line = ax.axhline(
                    y=baseline_y_val, 
                    color=BASELINE_COLOR, 
                    linestyle=BASELINE_STYLE, 
                    linewidth=2, 
                    label=baseline_label, 
                    zorder=10
                )
                if is_legend_plot:
                    legend_handles.append(line)
                    legend_labels.append(baseline_label)
            continue

        plot_color = COLORS[color_idx % len(COLORS)]
        linewidth = 2 
        linestyle = "-"
        mse_plot_data = disturb_data[agg_type]
        zorder_val = 5
        color_idx += 1

        line, = ax.plot(
            x_full,
            mse_plot_data,
            color=plot_color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=disturbance_types_9_map[disturb_type], 
            alpha=0.95,
            zorder=zorder_val
        )
        min_y=min(min_y,np.min(mse_plot_data)-0.01)
        max_y=max(max_y,np.max(mse_plot_data)+0.01)

        if is_legend_plot:
            legend_handles.append(line)
            legend_labels.append(disturbance_types_9_map[disturb_type])

        min_mse = np.min(mse_plot_data)
        min_index = np.where(mse_plot_data == min_mse)[0][0] 
        min_x_val = x_full[min_index]
        
        ax.scatter(
            min_x_val, 
            min_mse, 
            marker='*', 
            s=200,      
            color=plot_color, 
            edgecolor='black', 
            linewidth=0.5, 
            zorder=20    
        )
            
    ax.set_xscale('log', base=2)
    ax.tick_params(axis='y', labelsize=14) 
    ax.set_xticks(x_tick_pows) 
    ax.set_xticklabels([f"$2^{k}$" for k in range(len(x_tick_pows))], fontsize=14)
    ax.set_ylim(min_y, max_y)
    
    ax.grid(True, axis='y', alpha=0.4, linestyle='-.')
    ax.grid(False, axis='x')
    
    return legend_handles, legend_labels

def main_plot_six_subplots(
    all_datasets_data, 
    agg_types,
    metric="MSE",
    model_name="chronos-t5-tiny",
    save_dir="./plots" 
):
    plt.switch_backend('Agg')
    os.makedirs(save_dir, exist_ok=True)
    
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['legend.edgecolor'] = 'black'
    
    dataset_names = list(all_datasets_data.keys()) 
    N_ROWS = len(agg_types) # 2
    N_COLS = len(dataset_names) # 3
    
    fig, axes = plt.subplots(
        nrows=N_ROWS, 
        ncols=N_COLS, 
        figsize=(15, 8)
    )
    axes = axes.flatten() 

    all_y_vals = []
    for ds_name in dataset_names:
        data_set = all_datasets_data[ds_name]
        for dt in data_set.keys():
            for agg in agg_types:
                all_y_vals.extend(data_set[dt].get(agg, []))
    
    global_y_min = max(0, np.min(all_y_vals) * 0.95)
    global_y_max = np.max(all_y_vals) * 1.05
    global_y_lim = (global_y_min, global_y_max)
    
    legend_handles = []
    legend_labels = []

    for i in range(N_ROWS):
        agg_type = agg_types[i]
        for j in range(N_COLS):
            ds_name = dataset_names[j]
            current_data = all_datasets_data[ds_name]
            ax_index = i * N_COLS + j
            ax = axes[ax_index]

            is_legend = (ax_index == 0)
            
            h, l = plot_ensemble(
                data=current_data,
                agg_type=agg_type,
                model_name=model_name,
                metric=metric,
                save_dir=save_dir,
                ax=ax,
                is_legend_plot=is_legend,
                global_y_lim=global_y_lim,
                dataset_name=ds_name
            )
            
            if is_legend:
                legend_handles, legend_labels = h, l

            if i == N_ROWS - 1:
                 ax.set_xlabel("Number of Samples", fontsize=16, fontweight='bold')
            else:
                 ax.set_xlabel("")
                 ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
            
            if j == 0:
                ax_title = agg_type.replace("_", " ").title()
                ax.set_ylabel(f"{metric} ({ax_title})", fontsize=16, fontweight='bold')
            
            ax.set_title(data_map[ds_name], fontsize=16, fontweight='bold')
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_color('black')
            ax.spines['top'].set_linewidth(plt.rcParams['axes.linewidth'])
            ax.spines['right'].set_linewidth(plt.rcParams['axes.linewidth'])
            ax.spines['left'].set_linewidth(plt.rcParams['axes.linewidth'])
            ax.spines['bottom'].set_linewidth(plt.rcParams['axes.linewidth'])
            
    N_COLS_LEGEND = 4 
    
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.001), 
        ncols=N_COLS_LEGEND, 
        fontsize=16,
        frameon=True,
        framealpha=1.0,
        shadow=False,
        borderpad=0.5,
        edgecolor='black',
        title='',
        title_fontsize=16,
        handletextpad=0.5,
        columnspacing=1.5, 
    )
    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.9]) 

    save_filename_png = f"{model_name.lower().replace('-', '_')}_6subplots_ds_agg_min_star.pdf"
    save_path_png = os.path.join(save_dir, save_filename_png)
    fig.savefig(
        save_path_png, 
        facecolor='white',
        edgecolor='none'
    )
    save_filename_png = f"{model_name.lower().replace('-', '_')}_6subplots_ds_agg_min_star.png"
    save_path_png = os.path.join(save_dir, save_filename_png)
    fig.savefig(
        save_path_png, 
        dpi=300, 
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)
    print(f"6-Subplot Image successfully saved to: {save_path_png}")

def main():
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--data', '-d', type=str, default = "ETTh1,traffic,electricity")
    parser.add_argument('--output_dir','-dir', type=str, default = "results/predictions_")
    parser.add_argument('--models', '-m', type=str, default="timesfm-2.5-200m-pytorch")
    parser.add_argument('--save_dir', '-s', type=str, default="./plots")
    parser.add_argument('--temperatures', '-t', type=str, default="0.7")
    parser.add_argument('--length', '-l', type=str, default="512")
    parser.add_argument('--disturbance', '-p', type=str, default="prefix,insert,gaussian_noise,random_offset_noise,task_sensitive,task_dependent,task_reconstruct")
    parser.add_argument('--number','-n', type=int, default = 128)
    args = parser.parse_args()
    all_datasets_data = {}

    args.model=args.models.split(',')[0]
    args.save_dir=args.save_dir+args.model
    length=args.length.split(',')#[32,64,128,256,512,1024]
    temperatures=args.temperatures.split(',') #[0.7]
        
    datas=args.data.split(',')
    disturbance_types=args.disturbance.split(',')
    disturbance_types = [d for d in disturbance_types if d in disturbance_types_9 ]
    models = args.models.split(',')#["chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small"]     #
    RUN_ALL_AGGS=["exact_match","majority_voting",]#
    metric="MSE"
    for cur_data in datas:
        args.output_data_path=args.output_dir+cur_data
        all_result = get_result(args.output_data_path, args.model, args.number, disturbance_types,lens=length, temperature=temperatures)

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
        all_datasets_data[cur_data]=data

    main_plot_six_subplots(
        all_datasets_data=all_datasets_data,
        agg_types=RUN_ALL_AGGS,
        metric=metric,
        model_name=args.models,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
