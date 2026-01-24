import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoModelForCausalLM
import json
import os
import warnings
warnings.filterwarnings("ignore")
from evaluate_models import generate_disturbed_data



def timemoe_forecast(
    series,
    model_name="TimeMoE-50M",
    context_length=512,
    prediction_length=96,
    stride=96,
    device="cuda:0",
    temperature=0.7,
    top_p=0.9,
    save_path="",
    config=None,
):
    if isinstance(series, pd.Series):
        values = series.values.astype(np.float32)
    else:
        values = np.asarray(series, dtype=np.float32)

    # Load TimeMoE Model
    model = AutoModelForCausalLM.from_pretrained(
        f"Maple728/{model_name}",
        device_map={"": torch.device(device)},
        trust_remote_code=True,
        top_p=top_p,
        temperature=temperature,
    )
    model.eval()

    # Sliding-window forecasting with optional input disturbance
    if stride is None:
        stride = prediction_length
    
    forecasts = []    
    ori_x = []
    disturb_x = []

    for start_idx in tqdm(range(0, len(values) - context_length - prediction_length + 1, stride), desc="Sliding TimeMoE Forecast"):
        context = values[start_idx : start_idx + context_length]
        ori_x.append(context)
        
        if config:
            context = generate_disturbed_data(values, context, config)
        disturb_x.append(context)

        target = values[start_idx + context_length : start_idx + context_length + prediction_length]

        seq = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        seq = seq.to(dtype=autocast_dtype)
        
        with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
            output = model(seq)
            if hasattr(output, "logits"):
                logits = output.logits
                if logits.ndim == 3:
                    predictions_batch = logits[:, -prediction_length:, 0]
                else:
                    predictions_batch = logits[:, -prediction_length:]
            else:
                predictions_batch = output[:, -prediction_length:]

        preds = predictions_batch.to(dtype=torch.float32).squeeze(0).cpu().numpy()
        
        forecasts.append(preds)
        true_values.append(target)

    forecasts = np.array(forecasts)
    true_values = np.array(true_values)
    ori_x = np.array(ori_x)
    disturb_x = np.array(disturb_x)

    # Flatten predictions and targets for metric computation
    y_pred = forecasts.flatten()
    y_true = true_values.flatten()

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
    MODEL_NAME = "TimeMoE-50M"
    CONTEXT_LENGTH = 512
    PREDICTION_LENGTH = 96
    STRIDE = 24
    TEMPERATURE = 0.9
    TOP_P = 0.9
    DEVICE = "cuda:3"
    SAVE_PATH = "prediction"
    
    DATA_PATH = "dataset/ETT-small/ETTh1.csv"
    TARGET_COL = "OT"
    df = pd.read_csv(DATA_PATH)
    series = df[TARGET_COL].values.astype(np.float32)
    series_mean = series.mean()
    series_std = series.std()
    series = (series - series_mean) / series_std

    timemoe_forecast(
        series=series,
        model_name=MODEL_NAME,
        context_length=CONTEXT_LENGTH,
        prediction_length=PREDICTION_LENGTH,
        stride=STRIDE,
        device=DEVICE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        save_path=SAVE_PATH,
    )

