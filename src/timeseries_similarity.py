"""
Time Series Similarity Evaluation Module

This module provides similarity metrics specifically designed for 
comparing time series predictions with ground truth values.

These are NUMERICAL similarity metrics, not text-based metrics.
"""

import numpy as np
from typing import List, Dict, Optional
import json
import os
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dtaidistance import dtw


def pearson_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between prediction and ground truth.
    
    Measures linear correlation between two sequences.
    Range: [-1, 1], where 1 = perfect positive correlation
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        Pearson correlation coefficient
    """
    if len(pred) != len(gt):
        raise ValueError("Prediction and ground truth must have same length")
    
    if len(pred) < 2:
        return 1.0
    
    corr, _ = pearsonr(pred.flatten(), gt.flatten())
    return float(corr)


def spearman_correlation(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Spearman rank correlation coefficient.
    
    Measures monotonic relationship between two sequences.
    Range: [-1, 1], where 1 = perfect monotonic relationship
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        Spearman correlation coefficient
    """
    if len(pred) != len(gt):
        raise ValueError("Prediction and ground truth must have same length")
    
    if len(pred) < 2:
        return 1.0
    
    corr, _ = spearmanr(pred.flatten(), gt.flatten())
    return float(corr)


def cosine_similarity_numerical(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute cosine similarity between prediction and ground truth vectors.
    
    Measures angle between two vectors (direction similarity).
    Range: [-1, 1], where 1 = same direction
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        Cosine similarity
    """
    if len(pred) != len(gt):
        raise ValueError("Prediction and ground truth must have same length")
    
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    # Avoid division by zero
    pred_norm = np.linalg.norm(pred_flat)
    gt_norm = np.linalg.norm(gt_flat)
    
    if pred_norm == 0 or gt_norm == 0:
        return 0.0
    
    # scipy.spatial.distance.cosine returns distance, not similarity
    # similarity = 1 - distance
    similarity = 1 - cosine_distance(pred_flat, gt_flat)
    return float(similarity)


def dtw_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance.
    
    Measures similarity allowing for time shifts and warping.
    Range: [0, inf], where 0 = identical sequences
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        DTW distance (lower is better)
    """
    try:
        distance = dtw.distance(pred.flatten(), gt.flatten())
        return float(distance)
    except Exception as e:
        print(f"DTW computation failed: {e}")
        return float('inf')


def normalized_dtw_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute normalized DTW similarity score.
    
    Converts DTW distance to similarity in [0, 1] range.
    1 = identical, 0 = very different
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        Normalized DTW similarity
    """
    distance = dtw_distance(pred, gt)
    
    if distance == float('inf'):
        return 0.0
    
    # Normalize by sequence length and value range
    max_val = max(np.max(np.abs(pred)), np.max(np.abs(gt)))
    if max_val == 0:
        max_val = 1.0
    
    normalized_distance = distance / (len(pred) * max_val)
    
    # Convert distance to similarity: similarity = 1 / (1 + distance)
    similarity = 1.0 / (1.0 + normalized_distance)
    return float(similarity)


def mse_based_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute MSE-based similarity score.
    
    Converts MSE to similarity in [0, 1] range.
    1 = perfect prediction (MSE=0), approaches 0 for large MSE
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        MSE-based similarity
    """
    if len(pred) != len(gt):
        raise ValueError("Prediction and ground truth must have same length")
    
    mse = mean_squared_error(gt.flatten(), pred.flatten())
    
    # Convert MSE to similarity
    # Use negative exponential: similarity = exp(-mse)
    similarity = np.exp(-mse)
    return float(similarity)


def r2_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute R² score (coefficient of determination).
    
    Measures proportion of variance explained by predictions.
    Range: (-inf, 1], where 1 = perfect prediction
    Note: Can be negative if model is worse than mean baseline
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        R² score
    """
    if len(pred) != len(gt):
        raise ValueError("Prediction and ground truth must have same length")
    
    if len(pred) < 2:
        return 1.0
    
    r2 = r2_score(gt.flatten(), pred.flatten())
    return float(r2)


def smape(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Range: [0, 200], where 0 = perfect prediction
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        SMAPE value
    """
    if len(pred) != len(gt):
        raise ValueError("Prediction and ground truth must have same length")
    
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    
    denominator = (np.abs(pred_flat) + np.abs(gt_flat))
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape_value = np.mean(2.0 * np.abs(pred_flat - gt_flat) / denominator) * 100
    return float(smape_value)


def smape_based_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Convert SMAPE to similarity score in [0, 1] range.
    
    1 = perfect prediction (SMAPE=0), 0 = worst prediction (SMAPE=200)
    
    Args:
        pred: Prediction array
        gt: Ground truth array
    
    Returns:
        SMAPE-based similarity
    """
    smape_value = smape(pred, gt)
    similarity = 1.0 - (smape_value / 200.0)
    return float(max(0.0, similarity))


def compute_all_timeseries_similarities(
    pred: np.ndarray,
    gt: np.ndarray,
    include_dtw: bool = True
) -> Dict[str, float]:
    """
    Compute all available time series similarity metrics.
    
    Args:
        pred: Prediction array
        gt: Ground truth array
        include_dtw: Whether to include DTW (can be slow for long sequences)
    
    Returns:
        Dictionary of metric names to values
    """
    results = {}
    
    # Correlation-based metrics
    try:
        results['pearson_correlation'] = pearson_correlation(pred, gt)
    except Exception as e:
        print(f"Pearson correlation failed: {e}")
        results['pearson_correlation'] = None
    
    try:
        results['spearman_correlation'] = spearman_correlation(pred, gt)
    except Exception as e:
        print(f"Spearman correlation failed: {e}")
        results['spearman_correlation'] = None
    
    # Vector similarity
    try:
        results['cosine_similarity'] = cosine_similarity_numerical(pred, gt)
    except Exception as e:
        print(f"Cosine similarity failed: {e}")
        results['cosine_similarity'] = None
    
    # Distance-based metrics
    if include_dtw:
        try:
            results['dtw_similarity'] = normalized_dtw_similarity(pred, gt)
            results['dtw_distance'] = dtw_distance(pred, gt)
        except Exception as e:
            print(f"DTW failed: {e}")
            results['dtw_similarity'] = None
            results['dtw_distance'] = None
    
    # Error-based similarities
    try:
        results['mse_similarity'] = mse_based_similarity(pred, gt)
    except Exception as e:
        print(f"MSE similarity failed: {e}")
        results['mse_similarity'] = None
    
    try:
        results['smape_similarity'] = smape_based_similarity(pred, gt)
    except Exception as e:
        print(f"SMAPE similarity failed: {e}")
        results['smape_similarity'] = None
    
    # R² score
    try:
        results['r2_score'] = r2_similarity(pred, gt)
    except Exception as e:
        print(f"R² score failed: {e}")
        results['r2_score'] = None
    
    # Traditional error metrics (for reference)
    try:
        results['mse'] = float(mean_squared_error(gt.flatten(), pred.flatten()))
        results['rmse'] = float(np.sqrt(results['mse']))
        results['mae'] = float(mean_absolute_error(gt.flatten(), pred.flatten()))
        results['smape'] = smape(pred, gt)
    except Exception as e:
        print(f"Error metrics failed: {e}")
        results['mse'] = None
        results['rmse'] = None
        results['mae'] = None
        results['smape'] = None
    
    return results


def evaluate_predictions_vs_groundtruth(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    save_path: Optional[str] = None,
    is_print=False,
    include_dtw: bool = False  # DTW can be slow
) -> Dict[str, float]:
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match "
            f"number of ground truths ({len(ground_truths)})"
        )
    
    if len(predictions) == 0:
        print("Warning: No predictions provided")
        return {}
    
    all_metrics = {}
    for idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_arr = np.array(pred)
        gt_arr = np.array(gt)
        
        # Compute all metrics for this pair
        metrics = compute_all_timeseries_similarities(pred_arr, gt_arr, include_dtw)
        
        # Store individual results
        for metric_name, value in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)
    
    # Compute statistics
    results = {'num_samples': len(predictions)}
    
    for metric_name, values in all_metrics.items():
        valid_values = [v for v in values if v is not None and not np.isnan(v) and not np.isinf(v)]
        
        if valid_values:
            results[f'{metric_name}_mean'] = float(np.mean(valid_values))
            results[f'{metric_name}_std'] = float(np.std(valid_values))
            results[f'{metric_name}_min'] = float(np.min(valid_values))
            results[f'{metric_name}_max'] = float(np.max(valid_values))
        else:
            results[f'{metric_name}_mean'] = None
            results[f'{metric_name}_std'] = None
            results[f'{metric_name}_min'] = None
            results[f'{metric_name}_max'] = None
    # Store individual values
    results['individual_samples'] = all_metrics
    if is_print:
        # Group metrics by type
        correlation_metrics = ['pearson_correlation', 'spearman_correlation', 'cosine_similarity']
        similarity_metrics = ['dtw_similarity', 'mse_similarity', 'smape_similarity']
        score_metrics = ['r2_score']
        error_metrics = ['mse', 'rmse', 'mae']
        print("\nCorrelation Metrics (higher is better, range: [-1, 1]):")
        for metric in correlation_metrics:
            mean_val = results.get(f'{metric}_mean')
            std_val = results.get(f'{metric}_std')
            if mean_val is not None:
                print(f"  {metric:25s}: {mean_val:7.4f} ± {std_val:.4f}")
        
        print("\nSimilarity Metrics (higher is better, range: [0, 1]):")
        for metric in similarity_metrics:
            mean_val = results.get(f'{metric}_mean')
            std_val = results.get(f'{metric}_std')
            if mean_val is not None:
                print(f"  {metric:25s}: {mean_val:7.4f} ± {std_val:.4f}")
        
        print("\nScore Metrics:")
        for metric in score_metrics:
            mean_val = results.get(f'{metric}_mean')
            std_val = results.get(f'{metric}_std')
            if mean_val is not None:
                print(f"  {metric:25s}: {mean_val:7.4f} ± {std_val:.4f}")
        
        print("\nError Metrics (lower is better):")
        for metric in error_metrics:
            mean_val = results.get(f'{metric}_mean')
            std_val = results.get(f'{metric}_std')
            if mean_val is not None:
                print(f"  {metric:25s}: {mean_val:7.4f} ± {std_val:.4f}")
        
    # Save results
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {save_path}")
    
    return results


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("Time Series Similarity Evaluation Examples")
    print("="*70)
    
    np.random.seed(42)
    
    # Example 1: Perfect prediction
    print("\n### Example 1: Perfect Prediction ###")
    gt = np.sin(np.linspace(0, 4*np.pi, 50))
    perfect_pred = gt.copy()
    
    results = compute_all_timeseries_similarities(perfect_pred, gt, include_dtw=True)
    print("\nPerfect prediction results:")
    for metric, value in results.items():
        if value is not None and not np.isinf(value):
            print(f"  {metric:25s}: {value:.4f}")
    
    # Example 2: Good prediction (small noise)
    print("\n### Example 2: Good Prediction (Small Noise) ###")
    good_pred = gt + np.random.randn(50) * 0.1
    
    results = compute_all_timeseries_similarities(good_pred, gt, include_dtw=False)
    print("\nGood prediction results:")
    for metric, value in results.items():
        if value is not None and not np.isinf(value):
            print(f"  {metric:25s}: {value:.4f}")
    
    # Example 3: Poor prediction (large noise)
    print("\n### Example 3: Poor Prediction (Large Noise) ###")
    poor_pred = gt + np.random.randn(50) * 2.0
    
    results = compute_all_timeseries_similarities(poor_pred, gt, include_dtw=False)
    print("\nPoor prediction results:")
    for metric, value in results.items():
        if value is not None and not np.isinf(value):
            print(f"  {metric:25s}: {value:.4f}")
    
    # Example 4: Batch evaluation
    print("\n### Example 4: Batch Evaluation ###")
    predictions = [
        gt + np.random.randn(50) * 0.1 for _ in range(5)
    ]
    ground_truths = [gt for _ in range(5)]
    
    batch_results = evaluate_predictions_vs_groundtruth(
        predictions,
        ground_truths,
        save_path="timeseries_similarity_example.json",
        include_dtw=False
    )
