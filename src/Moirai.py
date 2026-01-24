import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import warnings
warnings.filterwarnings("ignore")
from evaluate_models import generate_disturbed_data

def moirai_forecast(
    series,                      
    model_name="moirai-1.1-R-small",          
    context_length=512,
    prediction_length=96,
    patch_size=32,
    batch_size=8,
    device="cuda:0",
    num_samples=1,
    save_path="",
    target_col="OT",
    stride=96,
    config=None,
):
    """
    Run Moirai forecasting with a sliding-window protocol and optional input disturbance.

    This function mirrors the Chronos-style rolling inference pipeline to ensure
    consistent evaluation and aggregation across different TS foundation models.

    Args:
        series: 1D np.ndarray or pandas.Series containing the input time series.
        model_name: Pretrained Moirai model identifier.
        context_length: Length of the historical context window.
        prediction_length: Forecast horizon.
        patch_size: Patch size used by the Moirai backbone.
        batch_size: Batch size for predictor inference.
        device: Device identifier (e.g., "cuda:0").
        num_samples: Number of probabilistic forecast samples.
        save_path: Path to save prediction results as a JSON file.
        target_col: Target column name used for GluonTS datasets.
        stride: Sliding window stride.
        config: Optional disturbance configuration dictionary.

    Returns:
        results: dict containing predictions, ground truth, disturbance inputs,
                 and evaluation metrics.
    """
    if isinstance(series, pd.Series):
        values = series.values.astype(np.float32)
    else:
        values = np.asarray(series, dtype=np.float32)

    # Load pretrained Moirai module
    module = MoiraiModule.from_pretrained(f"Salesforce/{model_name}")

    new_context_length = context_length
    if config:
        for key in config:
            if "length" in key:
                new_context_length += config[key]
                break
                
    # Initialize Moirai forecasting model
    forecast_model = MoiraiForecast(
        module=module,
        prediction_length=prediction_length,
        context_length=new_context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0, 
        past_feat_dynamic_real_dim=0,
    )

    predictor = forecast_model.create_predictor(batch_size=batch_size, device=device)
    
    # Sliding-window forecasting with optional input disturbance
    if stride is None:
        stride = prediction_length

    forecasts_list = []  
    true_values_list = [] 
    ori_x = []
    disturb_x = []

    for start_idx in tqdm(range(0, len(values) - context_length - prediction_length + 1, stride), desc="Sliding Moirai Forecast"):
        context_window = values[start_idx : start_idx + context_length]
        ori_x.append(context_window)
        
        if config:
            context_window = generate_disturbed_data(values, context_window, config)
        disturb_x.append(context_window)

        target = values[start_idx + context_length : start_idx + context_length + prediction_length]

        context_index = pd.date_range(start="2000-01-01", periods=len(context_window))
        tmp_df = pd.DataFrame({target_col: context_window}, index=context_index)
        tmp_ds = PandasDataset({target_col: tmp_df[target_col]})

        forecasts = list(predictor.predict(tmp_ds))
        pred = forecasts[0].mean 
        
        forecasts_list.append(pred)
        true_values_list.append(target)

    forecasts_list = np.array(forecasts_list)
    true_values_list = np.array(true_values_list)
    ori_x = np.array(ori_x)
    disturb_x = np.array(disturb_x)

    # Flatten predictions and targets for metric computation
    y_pred = forecasts_list.flatten()
    y_true = true_values_list.flatten()

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    print(f"{model_name} :: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

    results = {
        "ori_x": ori_x.tolist(),
        "disturb_x": disturb_x.tolist(),
        "config": config,
        "pred": forecasts_list.tolist(), 
        "gt": true_values_list.tolist(),
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
    # ===== Parameters =====

    CONTEXT_LENGTH = 512
    PREDICTION_LENGTH = 96
    PATCH_SIZE = 32
    BATCH_SIZE = 8
    NUM_SAMPLES = 100
    MODEL_NAME = "moirai-1.1-R-small"
    DEVICE = "cuda:3"
    SAVE_PATH = "prediction"
    STRIDE = 24

    # ===== Load Data =====
    DATA_PATH = "dataset/ETT-small/ETTh1.csv"
    TARGET_COL = "OT"
    df = pd.read_csv(DATA_PATH)
    series = df[TARGET_COL].values.astype(np.float32)

    series_mean = series.mean()
    series_std = series.std()
    series = (series - series_mean) / series_std

    # ===== Run Moirai Forecast =====
    moirai_forecast(
        series=series,
        model_name=MODEL_NAME,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        patch_size=PATCH_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        num_samples=NUM_SAMPLES,
        save_path=SAVE_PATH,
        target_col=TARGET_COL,
        stride=STRIDE,
    )
