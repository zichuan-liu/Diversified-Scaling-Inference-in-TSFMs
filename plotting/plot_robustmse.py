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
                            break
                        results.append(result)
                all_result[d_type] = results
    return all_result

def main():
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--data', '-d', type=str, default = "ETTh1")
    parser.add_argument('--output_dir','-dir', type=str, default = "/root/newdata/RobustTSFM/results/predictions_")
    parser.add_argument('--models', '-m', type=str, default="moirai-1.1-R-base") # chronos-t5-base TimeMoE-200M timesfm-2.5-200m-pytorch moirai-1.1-R-base
    parser.add_argument('--save_dir', '-s', type=str, default="./plots")
    parser.add_argument('--temperatures', '-t', type=str, default="0.7")
    parser.add_argument('--length', '-l', type=str, default="512")
    parser.add_argument('--disturbance', '-p', type=str, default="task_reconstruct,task_dependent,task_sensitive")#"prefix,insert,gaussian_noise,random_offset_noise"
    parser.add_argument('--number','-n', type=int, default = 128)
    parser.add_argument('--sample','-sample', type=int, default = 5)
    parser.add_argument('--sample_num','-sn', type=int, default = 64)
    args = parser.parse_args()

    args.output_data_path=args.output_dir+args.data
    args.save_dir=args.save_dir+args.data
    length=args.length.split(',')#[32,64,128,256,512,1024]
    temperatures=args.temperatures.split(',') #[0.7]
    disturbance_types=args.disturbance.split(',')  #['suffix',  'missing_data', 'prefix', 'insert', 'gaussian_noise', 'random_offset_noise']  #[]   #
    disturbance_types = [d for d in disturbance_types if d in disturbance_types_9 ]
    models = args.models.split(',')#["chronos-t5-tiny", "chronos-t5-mini", "chronos-t5-small"]     #
    RUN_ALL_AGGS=["exact_match","majority_voting"]#
    metric="MSE"
    sample=args.sample

    if len(models)==1:
        args.model=models[0]
        all_result = get_result(args.output_data_path, args.model, args.number, disturbance_types,lens=length, temperature=temperatures)
    else:
        all_result={}
        for model in models:
            temp = get_result(args.output_data_path, model, args.number, disturbance_types,lens=length, temperature=temperatures)
            for key in temp:
                all_result[model] = temp[key]
    
    data={}
    for i in range(sample):
        all_sample = [] 
        for key in all_result:
            cur_all_result = all_result[key]
            all_sample.extend(cur_all_result)
        cur_all_result = random.sample(all_sample, args.sample_num)

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
            temp_data[agg] = inference_scaling_results[-1]
        temp_data[metric] = [mse[metric] for mse in cur_all_result]
        data[i] = temp_data

    majority_voting_data = []
    exact_match_data = []
    mse_data = []

    for val in data.values():
        majority_voting_data.append(val['majority_voting'])
        exact_match_data.append(val['exact_match'])
        mse_data.append(val[metric])
    mv_arr = np.array(majority_voting_data)
    em_arr = np.array(exact_match_data)
    mse_arr = np.array(mse_data)

    mv_mean = np.mean(mv_arr)
    mv_std = np.std(mv_arr)

    em_mean = np.mean(em_arr)
    em_std = np.std(em_arr)
    
    mse_mean = np.mean(mse_arr)
    mse_std = np.std(mse_arr)

    print(f"exact_match:        {em_mean:.4f}$\pm$\\tiny{em_std:.4f}")
    print(f"majority_voting:    {mv_mean:.4f}$\pm$\\tiny{mv_std:.4f}")
    print(f"mse:    {mse_mean:.4f}$\pm$\\tiny{mse_std:.4f}")

if __name__ == "__main__":
    main()