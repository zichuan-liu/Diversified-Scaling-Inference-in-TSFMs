import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import warnings
import argparse
warnings.filterwarnings('ignore')
from disturb_function import generate_disturbance_configs, generate_disturbance_when
from evaluate_models import universal_prediction

def get_disturbed_series_by_type(disturbed_data_results, disturb_name):
    series_list = []
    config_list = []
    for res in disturbed_data_results:
        if disturb_name in res.get("disturbance_type"):
            series_list.append(np.array(res.get("perturbed_series"), dtype=np.float32))
            config_list.append(res.get("config_idx"))
    return series_list, config_list


if __name__ == "__main__":
    TARGET_COL = "OT"
    STRIDE = 32  
    DEVICE = "cuda:0"   # cpu
    TOP_P = 1.0  

    parser = argparse.ArgumentParser(description='Training Type Argparse')
    parser.add_argument('--model','-m', type=str, default = "chronos-t5-tiny") # "timesfm-2.5-200m-pytorch"  # TimeMoE-50M
    parser.add_argument('--input_len','-i', type=int, default = 512)
    parser.add_argument('--output_len','-o', type=int, default = 96)
    parser.add_argument('--number','-n', type=int, default = 128)
    parser.add_argument('--special_number','-sn', type=int, default = -1)
    parser.add_argument('--temp','-t', type=float, default = 0.7)
    parser.add_argument('--data','-data', type=str, default = "ETTh1")
    parser.add_argument('--output_dir','-dir', type=str, default = "./results/predictions_")
    parser.add_argument('--disturb','-d', type=str, default = "prefix") #'suffix', 'prefix', 'insert', 'gaussian_noise', 'random_offset_noise', 'missing_data'
    parser.add_argument('--disturb_when','-dw', type=str, default = "false") 
    args = parser.parse_args()
    DATA_PATH = f"dataset/ETT-small/{args.data}.csv"

    TEMPERATURE=args.temp
    CONTEXT_LENGTH=args.input_len
    PREDICTION_LENGTH=args.output_len
    NUM_SAMPLES=args.number
    output_data_path=args.output_dir+args.data
    special_number=args.special_number

    output_data_path=output_data_path+"_l"+str(CONTEXT_LENGTH)+"_t"+str(TEMPERATURE)

    IS_RUN_BASELINE=True
    IS_RUN_DISTURB=True
    if args.disturb_when=='true':
        IS_RUN_BASELINE=False
        disturbance_types=[args.disturb]
        disturbance_configs = generate_disturbance_when(NUM_SAMPLES, args.disturb)
    elif args.disturb=="baseline":
        IS_RUN_DISTURB=False
    elif args.disturb in ['suffix', 'prefix', 'insert', 'gaussian_noise', 'random_offset_noise', 'missing_data', "task_dependent", 'task_sensitive', 'task_reconstruct']:
        disturbance_types=[args.disturb]
        IS_RUN_BASELINE=False
        disturbance_configs = generate_disturbance_configs(NUM_SAMPLES,disturbance_types)
    else:
        disturbance_types=['suffix', 'prefix', 'insert', 'gaussian_noise', 'random_offset_noise', 'missing_data', "task_dependent", 'task_sensitive', 'task_reconstruct']
        IS_RUN_BASELINE, IS_RUN_DISTURB=False,False
        
    MODEL_FULL_NAME=args.model
    MODEL_NAME=MODEL_FULL_NAME.split('-', 1)[0].lower()

    print("===== Loading Dataset =====")
    df = pd.read_csv(DATA_PATH)
    original_series = df[TARGET_COL].values.astype(np.float32)
    original_series = (original_series - original_series.mean()) / (original_series.std() + 1e-8)
    print(f"Original series length: {len(original_series)}")

    if IS_RUN_BASELINE:
        baseline_results = universal_prediction(
                series=original_series,
                model_name=MODEL_NAME,
                model_full_name=MODEL_FULL_NAME,
                num_samples=NUM_SAMPLES,
                device=DEVICE,
                save_path=output_data_path,
                context_length=CONTEXT_LENGTH,
                prediction_length=PREDICTION_LENGTH,
                stride=STRIDE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                special_number=special_number
            )

    if IS_RUN_DISTURB:
        for d_type in disturbance_types:
            print(f"\nProcessing disturbance type: {d_type}")
            for idx, configs in enumerate(disturbance_configs):
                if special_number>0 and idx!=special_number-1:
                    continue
                save_path=os.path.join(output_data_path, d_type, f"{idx}")
                os.makedirs(save_path, exist_ok=True)
                baseline_results = universal_prediction(
                    series=original_series,
                    model_name=MODEL_NAME,
                    model_full_name=MODEL_FULL_NAME,
                    num_samples=1,
                    device=DEVICE,
                    save_path=save_path,
                    context_length=CONTEXT_LENGTH,
                    prediction_length=PREDICTION_LENGTH,
                    stride=STRIDE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    config=configs
                )
    