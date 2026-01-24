import numpy as np
from statsmodels.tsa.seasonal import STL
import random
def task_dependent_structural_injection(
    series,
    period=24,       # STL decomposition period
    rho=0.1,         # Injection strength (the only hyperparameter)
    w=12             # Residual local variance window
):
    """
    Version 1: pure data-dependent (structure-driven).
    Now with sign consistency: f(x) = sign(x) * |f(x)|, and zero-mean.
    """

    series = np.asarray(series, dtype=np.float32)
    T_len = len(series)

    # 1. STL decomposition 
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    T = res.trend.astype(np.float32)
    S = res.seasonal.astype(np.float32)
    R = res.resid.astype(np.float32)

    # 2. Trend perturbation f_T(x)
    f_T = np.gradient(T)

    # 3. Seasonal perturbation f_S(x) 
    f_S = np.abs(S)

    # 4. Residual perturbation f_R(x) 
    pad_R = np.pad(R, (w, w), mode='reflect')
    local_std = np.array([
        pad_R[i:i+2*w+1].std()
        for i in range(T_len)
    ], dtype=np.float32)
    f_R = local_std

    # 5. total f(x)
    f = f_T + f_S + f_R

    # 5a. sign consistency with x 
    f = np.sign(series + 1e-6) * np.abs(f)

    # 5b. zero-mean 
    f = f - f.mean()

    # 6. SNR normalization 
    eps = 1e-8
    eta = rho * np.linalg.norm(series) / (np.linalg.norm(f) + eps)
    f_scaled = eta * f

    series_prime = series + f_scaled

    components = {
        "T": T, "S": S, "R": R, 
        "f_T": f_T, "f_S": f_S, "f_R": f_R
    }
    return series_prime.astype(np.float32), f_scaled.astype(np.float32), components


def task_sensitive_structural_injection(
    series,
    period=24,         # STL decomposition period
    rho=0.1,
    smooth_window=12
):
    """
    Version 2: task-sensitive (model-free).
    Now with sign consistency: f(x) = sign(x) * |f(x)|, and zero-mean.
    """

    series = np.asarray(series, dtype=np.float32)
    T_len = len(series)

    # 1. STL decomposition
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    T = res.trend.astype(np.float32)
    S = res.seasonal.astype(np.float32)
    R = res.resid.astype(np.float32)

    # 2. Trend difficulty 
    f_T = np.abs(np.gradient(T))

    # 3. Seasonal non-stationarity 
    S_shift = np.roll(S, period)
    f_S = np.abs(S - S_shift)

    # 4. Residual unpredictability 
    f_R = np.abs(R)

    # 5. Combine 
    f = f_T + f_S + f_R

    # 5a. sign consistency with x 
    f = np.sign(series + 1e-6) * np.abs(f)

    # 5b. zero-mean 
    f = f - f.mean()

    # 6. SNR normalization
    eps = 1e-8
    eta = rho * np.linalg.norm(series) / (np.linalg.norm(f) + eps)
    f_scaled = eta * f

    series_prime = series + f_scaled

    components = {
        "T": T, "S": S, "R": R,
        "f_T": f_T, "f_S": f_S, "f_R": f_R
    }
    return series_prime.astype(np.float32), f_scaled.astype(np.float32), components

def task_reconstruction(series, model_name,temp=0.7):
    """
    Version 3: Model-based task-specific reconstruction.
    This method simulates inference-time uncertainty induced by model sampling rather than direct signal manipulation.
    """
    # print("x", len(series))
    context_length=64
    pre=8
    replace_prob=0.8
    if random.uniform(0, 1)<replace_prob:
        return series, None, None

    x=len(series)
    # 1. Model-specific reconstruction
    if "chronos" in model_name: #decoder model
        from Chronos import chronos_forecast
        results = chronos_forecast(series, model_name, context_length=context_length, prediction_length=pre,stride=pre,temperature=temp)
    elif "moirai" in model_name: #encoder model
        from Moirai import moirai_forecast
        results = moirai_forecast(series, model_name, context_length=context_length, prediction_length=pre, stride=pre,temperature=temp)
    elif "timesfm" in model_name:
        from TimesFM import timesfm_forecast
        results = timesfm_forecast(series, model_name, context_length=context_length, prediction_length=pre, stride=pre)

    # 2. Reassemble reconstructed series
    series_prime = results["pred"]
    series_prime = np.concatenate(series_prime)
    if (x-context_length)%pre==0:
        temp=[]
    else:
        temp=series[-(x-context_length)%pre:]
    series_prime = np.concatenate([series[:context_length], series_prime, temp])
    return series_prime, None, None

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("dataset/ETT-small/ETTh1.csv")
    series = ((df["OT"] - df["OT"].mean()) / df["OT"].std()).values.astype(np.float32)[:800]

    # series_prime, f, comps = task_dependent_structural_injection(series, rho=0.1) # first data-dependent method
    # series_prime, f, comps = task_sensitive_structural_injection(series, rho=0.1) # second task-specific method
    series_prime, f, comps = task_reconstruction(series, model_name="chronos-t5-tiny", context_length=64) # model-based task-specific method

    print(len(series_prime), len(series))

    plt.figure(figsize=(10, 4))
    plt.plot(series[:500], label="Original series", linewidth=1.5)
    plt.plot(series_prime[:500], label="Perturbed series", linewidth=1.5)

    plt.xlabel("Time steps")
    plt.ylabel("Value")
    plt.title("Comparison of Original and Perturbed Series (First 500 Time Steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("series_vs_series_prime_500.png")
    plt.close()
