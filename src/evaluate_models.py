import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
from disturb_function import TimeSeriesDisturbance

model_dict = [
    # "timesfm-2.5-200m-pytorch",   
    # "timesfm-1.0-200m-pytorch",
    # "timesfm-2.0-500m-pytorch",
    # "moirai-1.1-R-small",         
    # "moirai-1.1-R-base",
    # "moirai-1.1-R-large",
    # "moirai-1.0-R-small",
    # "moirai-1.0-R-base",
    # "moirai-1.0-R-large",
    # "moirai-moe-1.0-R-small",
    # "moirai-moe-1.0-R-base",
    # "moirai-2.0-R-small",
    #"chronos-t5-tiny",           
    # "chronos-t5-mini",
    # "chronos-t5-small",
    # "chronos-t5-base",
    # "chronos-t5-large",
    # "chronos-bolt-tiny",
    # "chronos-bolt-mini",
    # "chronos-bolt-small",
    # "chronos-bolt-base",
    # "TimeMoE-50M",              
    # "TimeMoE-200M",               
]

dataset_dict = [
    "dataset/ETT-small/ETTh1.csv",
    "dataset/ETT-small/ETTh2.csv",
    "dataset/ETT-small/ETTm1.csv",
    "dataset/ETT-small/ETTm2.csv",
    "dataset/electricity/electricity.csv",
    "dataset/exchange_rate/exchange_rate.csv",
    "dataset/weather/weather.csv",  
    "dataset/traffic/traffic.csv",
]


def generate_disturbed_data(
    original_series,
    context,
    disturbance_config,
):
    disturber = TimeSeriesDisturbance(context, original_series)
    d_type = disturbance_config.get("disturbance_type", "unknown")
    c = disturbance_config.copy()
    if d_type in ["insert_segment", "insert_mean_segment", "insert_same_segment"] and "insert_ratio" in c:
        insert_pos = int(len(context) * c["insert_ratio"])
        c["insert_position"] = insert_pos
        c.pop("insert_ratio", None)
    disturbed_series = disturber.apply_disturbance(**c)
    disturbed_series = np.asarray(disturbed_series, dtype=np.float32).flatten()
    return disturbed_series.tolist()


def select_model_and_forecast(
    model_name,
    series,
    context_length=192,
    prediction_length=192,
    stride=32,
    device="cpu",
    save_path="prediction",
    config=None,
    **kwargs
):
    """
    Select and run a forecasting model based on model_name.

    Args:
        model_name: str, one of ["timesfm", "moirai", "chronos", "timemoe"]
        series: 1D np.ndarray or list, the time series to forecast on
        context_length: Number of input timesteps
        prediction_length: Number of steps to forecast
        stride: Sliding window stride
        device: GPU device string
        save_path: Path for saving results
        **kwargs: Additional model-specific parameters

    Returns:
        results: dict with keys ['pred', 'gt', 'MSE', 'MAE', 'RMSE']
    """
    model_name_lower = model_name.lower()

    if "timesfm" in model_name_lower:
        from TimesFM import timesfm_forecast

        # TimesFM model
        model_full_name = kwargs.get("model_full_name", "timesfm-2.5-200m-pytorch")
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1.0)
        num_samples = kwargs.get("num_samples", 50)

        results = timesfm_forecast(
            series=series,
            model_name=model_full_name,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            device=device,
            temperature=temperature,
            top_p=top_p,
            save_path=save_path,
            num_samples=num_samples,
            config=config
        )

    elif "moirai" in model_name_lower:
        from Moirai import moirai_forecast

        # Moirai model
        model_full_name = kwargs.get("model_full_name", "moirai-1.1-R-small")
        patch_size = kwargs.get("patch_size", 32)
        batch_size = kwargs.get("batch_size", 8)
        num_samples = kwargs.get("num_samples", 100)
        target_col = kwargs.get("target_col", "OT")

        results = moirai_forecast(
            series=series,
            model_name=model_full_name,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_size=patch_size,
            batch_size=batch_size,
            device=device,
            num_samples=num_samples,
            save_path=save_path,
            target_col=target_col,
            stride=stride,
            config=config
        )

    elif "chronos" in model_name_lower:
        # Chronos model
        from Chronos import chronos_forecast
        import torch
        model_full_name = kwargs.get("model_full_name", "chronos-t5-tiny")
        num_samples = kwargs.get("num_samples", 50)
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1)
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        results = chronos_forecast(
            series=series,
            model_name=model_full_name,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            device=device,
            torch_dtype=torch_dtype,
            save_path=save_path,
            config=config
        )

    elif "timemoe" in model_name_lower:
        from TimeMoE import timemoe_forecast

        # TimeMoE model (Maple728/TimeMoE)
        model_full_name = kwargs.get("model_full_name", "TimeMoE-50M")
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 1)
        # TimeMoE does not support num_samples, batch, etc, just one prediction
        results = timemoe_forecast(
            series=series,
            model_name=model_full_name,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            device=device,
            temperature=temperature,
            top_p=top_p,
            save_path=save_path,
            config=config
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from ['timesfm', 'moirai', 'chronos', 'timemoe']")

    return results

def average_evaluation(results_list, original_series):
    all_preds, gt_arr, min_pred_len = preprocess_results(results_list)
    
    if gt_arr is None:
        gt_arr = np.array(original_series)[-min_pred_len:]
    
    pred_avg = np.mean(all_preds, axis=0)
    mse = round(float(mean_squared_error(gt_arr, pred_avg)), 4)
    mae = round(float(mean_absolute_error(gt_arr, pred_avg)), 4)
    rmse = round(float(np.sqrt(mse)), 4)
    
    return {
        "prediction_avg": pred_avg.tolist(),
        "gt": gt_arr.tolist(),
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "num_models": len(all_preds)
    }

def universal_prediction(
    series,
    model_name="chronos",
    model_full_name="chronos-t5-tiny",
    num_samples=1000,
    device="cpu",
    save_path="prediction",
    context_length=192,
    prediction_length=192,
    stride=32,
    temperature=0.7,
    top_p=1,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=False,
    fix_quantile_crossing=True,
    config=None,
    special_number=-1,
):
    save_path = os.path.join(save_path, model_full_name)
    os.makedirs(save_path, exist_ok=True)
    results_list = []
    mse_list = []

    begin=1
    end=num_samples+1
    if special_number>0:
        begin=special_number
        end=special_number+1
    for i in tqdm(range(begin, end), desc=f"{model_name} Inference Scaling Samples"):
        # For different models, we prepare parameters as needed
        forecast_kwargs = dict(
            model_name=model_name,
            model_full_name=model_full_name,
            series=series,
            context_length=context_length,
            prediction_length=prediction_length,
            stride=stride,
            device=device,
            config=config,
        )
        forecast_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "normalize_inputs": normalize_inputs,
            "use_continuous_quantile_head": use_continuous_quantile_head,
            "force_flip_invariance": force_flip_invariance,
            "infer_is_positive": infer_is_positive,
            "fix_quantile_crossing": fix_quantile_crossing,
            "num_samples": 1,
            "save_path": os.path.join(save_path, f"{i}.json")
        })
        print("cur num:", i, "cur config:", config)
        result = select_model_and_forecast(**forecast_kwargs)
        results_list.append(result)
    return results_list

def inference_scaling_aggregator_average(results_list,metric="MSE",is_print=False):  
    num_samples=len(results_list)
    mse_list=[]
    series=results_list[0]['gt']
    for i in range(1, num_samples + 1):
        ensemble_result = average_evaluation(
            results_list[:i],
            original_series=series,
        )
        mse_list.append(ensemble_result[metric])
        if is_print:
            print(f"Sample {i}: {metric}={ensemble_result[metric]:.4f}")
    return mse_list

def inference_scaling_aggregator_majority_voting(results_list,metric="MSE",is_print=False):  
    num_samples=len(results_list)
    mse_list = []
    for i in range(1, num_samples + 1):
        predictions = [r['pred'] for r in results_list[:i]]  # each r['pred']: (N, prediction_length)
        predictions_np = np.stack(predictions, axis=0)  # shape (i, N, prediction_length)
        majority_voted_pred = np.median(predictions_np, axis=0)
        target = results_list[0]['gt']
        mse = float(mean_squared_error(majority_voted_pred, target))
        mae = float(mean_absolute_error(majority_voted_pred, target))
        rmse = float(np.sqrt(mse))
        ensemble_results = {"MSE": round(mse, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4)}
        mse_list.append(ensemble_results[metric])
    return mse_list

def preprocess_results(results_list):
    if not results_list:
        raise ValueError("results_list is empty")
    
    all_preds = []
    gt_arr = None 
    min_pred_len = float('inf')
    
    for result in results_list:
        pred = None
        current_gt = None
        if isinstance(result, (list, tuple)):
            if len(result) > 0 and isinstance(result[0], dict):
                pred_dict = result[0]
                pred = np.array(pred_dict['pred']).flatten() if 'pred' in pred_dict else None
                if 'gt' in pred_dict:
                    current_gt = np.array(pred_dict['gt']).flatten()
                elif len(result) > 1:
                    if isinstance(result[1], dict) and 'gt' in result[1]:
                        current_gt = np.array(result[1]['gt']).flatten()
                    elif isinstance(result[1], (list, np.ndarray)):
                        current_gt = np.array(result[1]).flatten()
        elif isinstance(result, dict):
            pred = np.array(result['pred']).flatten() if 'pred' in result else None
            current_gt = np.array(result['gt']).flatten() if 'gt' in result else None
        elif isinstance(result, np.ndarray):
            pred = result
        
        if pred is not None and pred.size > 0:
            pred_flat = pred.flatten() if pred.ndim > 1 else pred
            all_preds.append(pred_flat)
            min_pred_len = min(min_pred_len, len(pred_flat))
        if gt_arr is None and current_gt is not None and current_gt.size > 0:
            gt_arr = current_gt.flatten() if current_gt.ndim > 1 else current_gt
    if not all_preds:
        raise ValueError("No valid predictions found")
    
    all_preds = [p[:min_pred_len] for p in all_preds]
    if gt_arr is not None:
        gt_arr = gt_arr[:min_pred_len]
    
    return np.array(all_preds), gt_arr, min_pred_len

def compute_ensemble_metric(preds_stack, gt_arr, metric="MSE"):
    pred_avg = np.mean(preds_stack, axis=0)
    # pred_avg = preds_stack.flatten()
    if metric == "MSE":
        metric_val = mean_squared_error(gt_arr, pred_avg)
    elif metric == "MAE":
        metric_val = mean_absolute_error(gt_arr, pred_avg)
    elif metric == "RMSE":
        metric_val = np.sqrt(mean_squared_error(gt_arr, pred_avg))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return round(metric_val, 4)


def inference_scaling_aggregator_exact_match(results_list,metric="MSE",is_print=False):  
    all_preds, gt_arr, min_pred_len = preprocess_results(results_list)
    num_samples = len(all_preds)
    
    metric_cache = []
    for k in range(num_samples):
        current_preds = all_preds[:k+1]
        metric_val = compute_ensemble_metric(current_preds, gt_arr, metric)
        metric_cache.append(metric_val)
    
    mse_list = []
    for i in range(1, num_samples + 1):
        current_metrics = metric_cache[:i]
        min_metric = float(np.min(current_metrics))
        mse_list.append(min_metric)
        if is_print:
            print(f"Sample {i}: Best (min) {metric}={min_metric:.4f}")
    return mse_list

def compute_similarity(pred_results: list, save_path: str = None, include_dtw=False,is_print=False) -> dict:
    """
    Compute time series similarity metrics between predictions and ground truth.
    """
    from timeseries_similarity import (
        evaluate_predictions_vs_groundtruth,
        compute_all_timeseries_similarities
        )

    if not pred_results:
        print("Warning: No prediction results provided")
        return {}
    
    # Extract predictions and ground truth as numerical arrays
    predictions = []
    ground_truths = []
    
    for pred_result in pred_results:
        pred = pred_result.get('disturb_x', pred_result.get('predictions', []))
        gt = pred_result.get('ori_x', pred_result.get('ground_truth', []))
        
        # Convert to numpy arrays
        pred_arr = np.array(pred).flatten()
        gt_arr = np.array(gt).flatten()
        
        # Ensure same length
        min_len = min(len(pred_arr), len(gt_arr))
        predictions.append(pred_arr[:min_len])
        ground_truths.append(gt_arr[:min_len])
    
    if len(predictions) == 0:
        print("Warning: No valid prediction-ground truth pairs found")
        return {}
    
    # Use the correct time series similarity evaluation
    results = evaluate_predictions_vs_groundtruth(
        predictions,
        ground_truths,
        save_path=save_path,
        include_dtw=include_dtw,
        is_print=is_print
    )
    return results

# Example usage
if __name__ == "__main__":
    # test_all_model()   

    # Load dataset
    df = pd.read_csv("dataset/ETT-small/ETTh1.csv")
    series = ((df["OT"] - df["OT"].mean()) / df["OT"].std()).values.astype(np.float32)
