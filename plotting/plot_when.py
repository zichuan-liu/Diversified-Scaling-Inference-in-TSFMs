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
                    json_count=min(json_count, number_samples)
                    for i in tqdm(range(json_count)):
                        baseline_mse_file=os.path.join(baseline_path,f"{i+1}.json")
                        if not os.path.exists(baseline_mse_file):
                            break
                        with open(baseline_mse_file, "r") as f:
                            result = json.load(f)
                            if result["MSE"]>10:
                                result["MSE"]=10
                            baseline_results.append(result["MSE"])
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
                        if result["MSE"]>10:
                            result["MSE"]=10
                        results.append(result["MSE"])
                all_result[d_type] = results
    return all_result

def main():
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--data', '-d', type=str, default = "ETTh1,traffic,electricity")
    parser.add_argument('--output_dir','-dir', type=str, default = "results/predictions_")
    parser.add_argument('--models', '-m', type=str, default="timesfm-2.5-200m-pytorch")  # chronos-t5-tiny TimeMoE-50M timesfm-2.5-200m-pytorch moirai-1.1-R-small
    parser.add_argument('--temperatures', '-t', type=str, default="0.7")
    parser.add_argument('--length', '-l', type=str, default="512")
    parser.add_argument('--disturbance', '-p', type=str, default="task_dependent,task_sensitive,task_reconstruct")#task_dependent,task_sensitive,task_reconstruct
    parser.add_argument('--number','-n', type=int, default = 12)
    args = parser.parse_args()

    all_data = {}
    datas=args.data.split(',')

    for data in datas:
        args.output_data_path=args.output_dir+data
        length=args.length.split(',')#[32,64,128,256,512,1024]
        temperatures=args.temperatures.split(',')
        disturbance_types=args.disturbance.split(',')
        disturbance_types = [d for d in disturbance_types if d in disturbance_types_9 ]
        models = args.models.split(',')#["chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small"]     #

        args.model=models[0]
        all_result = get_result(args.output_data_path, args.model, args.number, disturbance_types,lens=length, temperature=temperatures)

        all_data[data] = all_result
    
    perturb_data={}
    for data  in datas:
        all_result = all_data[data]
        for perturb in all_result:
            mean_val = np.mean(all_result[perturb])
            std_val = np.std(all_result[perturb])

            str_print = f"{mean_val:.4f}$\pm$\\tiny{std_val:.4f}"
            if perturb not in perturb_data:
                perturb_data[perturb] = str_print
            else:
                perturb_data[perturb] += " & "+str_print

    for perturb in perturb_data:
        print(perturb, ":     ", perturb_data[perturb])
if __name__ == "__main__":
    main()