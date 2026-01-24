import os
import json
import numpy as np
import pandas as pd
import timesfm
import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import warnings
warnings.filterwarnings("ignore")
from evaluate_models import generate_disturbed_data


def timesfm_forecast(
    series,
    model_name="timesfm-2.5-200m-pytorch",
    context_length=512,
    prediction_length=96,
    stride=96,
    device="cuda:0",
    temperature=0.9,
    top_p=0.9,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    force_flip_invariance=True,
    infer_is_positive=False,
    fix_quantile_crossing=True,
    save_path="",
    result_name=None,
    num_samples=1,
    config=None,
):
    if isinstance(series, pd.Series):
        values = series.values.astype(np.float32)
    else:
        values = np.asarray(series, dtype=np.float32)

    # Load TimesFM Model
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        f"google/{model_name}",
        device_map=device,
        torch_compile=False,
        temparature=temperature, 
        top_p=top_p,
        num_samples=num_samples,
    )

    new_context_length = context_length
    if config:
        for key in config:
            if "length" in key:
                new_context_length += config[key]
                break

    model.compile(
        timesfm.ForecastConfig(
            max_context=new_context_length,
            max_horizon=prediction_length,
            normalize_inputs=normalize_inputs,
            use_continuous_quantile_head=use_continuous_quantile_head,
            force_flip_invariance=force_flip_invariance,
            infer_is_positive=infer_is_positive,
            fix_quantile_crossing=fix_quantile_crossing,
        )
    )

    # Sliding-window forecasting with optional input disturbance
    if stride is None:
        stride = prediction_length

    ori_x = []
    disturb_x = []
    forecasts = [] 
    true_values = []

    for start_idx in tqdm.tqdm(range(0, len(values) - context_length - prediction_length + 1, stride), desc="Sliding TimesFM Forecast"):
        context = values[start_idx : start_idx + context_length]
        ori_x.append(context)
        
        if config:
            context = generate_disturbed_data(values, context, config)
        disturb_x.append(context)
        
        target = values[start_idx + context_length : start_idx + context_length + prediction_length]
        
        inputs = [context]
        point_forecast, _ = model.forecast(horizon=prediction_length, inputs=inputs)
        
        pred = point_forecast[0] 
        
        forecasts.append(pred)
        true_values.append(target)

    forecasts = np.array(forecasts)
    true_values = np.array(true_values)
    ori_x = np.array(ori_x)
    disturb_x = np.array(disturb_x)

    # Flatten predictions and targets for metric computation
    y_true = true_values.flatten()
    y_pred = forecasts.flatten()

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    print(f"{model_name} :: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    results = {
        "ori_x": ori_x.tolist(),
        "disturb_x": disturb_x.tolist(),
        "config": config,
        "pred": forecasts.tolist(), 
        "gt": true_values.tolist(),
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
    }

    if save_path and len(save_path) > 0:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f)
            
    return results

if __name__ == "__main__":
    MODEL_NAME = "timesfm-2.5-200m-pytorch"
    CONTEXT_LENGTH = 512
    PREDICTION_LENGTH = 96
    STRIDE = 24
    DEVICE = "cuda:3"
    TEMPERATURE = 0.9
    TOP_P = 0.9
    SAVE_PATH = "prediction"
    NUM_SAMPLES=50

    DATA_PATH = "dataset/ETT-small/ETTh1.csv"
    TARGET_COL = "OT"
    df = pd.read_csv(DATA_PATH)
    series = df[TARGET_COL].values.astype(np.float32)

    series_mean = series.mean()
    series_std = series.std()
    series = (series - series_mean) / series_std

    timesfm_forecast(
        series=series,
        model_name=MODEL_NAME,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        stride=STRIDE,
        device=DEVICE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        save_path=SAVE_PATH,
        num_samples=NUM_SAMPLES,
    )
