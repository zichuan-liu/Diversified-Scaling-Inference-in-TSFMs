import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
import warnings
warnings.filterwarnings("ignore")
from evaluate_models import generate_disturbed_data

def chronos_forecast(
    series,
    model_name="chronos-t5-tiny",
    context_length=512,
    prediction_length=96,
    stride=96,
    num_samples=1,
    temperature=0.7,
    top_p=1,
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    config=None,
    save_path=""
):
    """
    Run Chronos forecasting with sliding window and save the results as a JSON file.

    Args:
        series: 1D np.ndarray or list, the time series to forecast on.
        model_name: HuggingFace model id or local path.
        context_length: Number of timesteps as context window.
        prediction_length: Number of steps to forecast.
        stride: Sliding window stride.
        num_samples: Number of forecast samples per window.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top-p.
        device: GPU device string (e.g. "cuda:0").
        torch_dtype: PyTorch dtype for model.
        save_path: Output path for prediction results JSON.

    Returns:
        results: dict containing predictions, ground truth and metrics.
    """
    # 1. Load Chronos model pipeline
    pipeline = ChronosPipeline.from_pretrained(
        f"amazon/{model_name}",
        # device_map=device if torch.cuda.is_available() else "cpu",
        device_map={"": 0},
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    new_context_length=context_length
    if config:
        for key in config:
            if "length" in key:
                new_context_length+=config[key]
                break
                
    # 2. Ensure the input series is a numpy float32 array
    if isinstance(series, pd.Series):
        series = series.values
    series = np.asarray(series, dtype=np.float32)

    # 3. Perform sliding window forecasting
    ori_x = []
    disturb_x = []
    forecasts = []
    true_values = []
    num_windows = (len(series) - context_length - prediction_length + 1) // stride
    for start_idx in tqdm(range(0, len(series) - context_length - prediction_length + 1, stride), desc="Sliding Chronos Forecast"):
        context = series[start_idx : start_idx + context_length]
        ori_x.append(context)
        if config:
            context = generate_disturbed_data(series, context, config)
        disturb_x.append(context)
        target = series[start_idx + context_length : start_idx + context_length + prediction_length]
        context_tensor = torch.tensor(context, dtype=torch.float32)

        # Model prediction (returns [1, num_samples, prediction_length])
        if temperature<=0:
            temperature=0.001
        forecast = pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
        )

        # Take mean over samples as point forecast: [prediction_length]
        mean_forecast = forecast.mean(dim=1).squeeze().cpu().numpy()
        forecasts.append(mean_forecast)
        true_values.append(target)

    ori_x = np.array(ori_x)
    disturb_x = np.array(disturb_x)

    forecasts = np.array(forecasts)
    true_values = np.array(true_values)

    # Flatten for metric calculation
    y_true = true_values.flatten()
    y_pred = forecasts.flatten()

    # Calculate metrics
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred)**0.5)
    mae = float(mean_absolute_error(y_true, y_pred))

    print(f"{model_name} :: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    results = {
        "ori_x": ori_x.tolist(),
        "disturb_x": disturb_x.tolist(),
        "config":config,
        "pred": forecasts.tolist(),
        "gt": true_values.tolist(),
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
    }

    if save_path and len(save_path)>0:
        with open(save_path, "w") as f:
            json.dump(results, f)
    return results


if __name__ == "__main__":

    MODEL_NAME = "chronos-t5-tiny"
    CONTEXT_LENGTH = 512
    PREDICTION_LENGTH = 96
    STRIDE = 24
    NUM_SAMPLES = 50
    TEMPERATURE = 0.9
    TOP_P = 0.9
    DEVICE = "cuda:3"
    TORCH_DTYPE = torch.bfloat16
    SAVE_PATH = "prediction"

    # for testing, load ETT dataset
    DATA_PATH = "dataset/ETT-small/ETTh1.csv"
    TARGET_COL = "OT"
    df = pd.read_csv(DATA_PATH)
    series = df[TARGET_COL].values.astype(np.float32)

    series_mean = series.mean()
    series_std = series.std()
    series = (series - series_mean) / series_std

    chronos_forecast(
        series=series,
        model_name=MODEL_NAME,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        stride=STRIDE,
        num_samples=NUM_SAMPLES,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        save_path=SAVE_PATH
    )
