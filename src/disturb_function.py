import numpy as np
import pandas as pd
import argparse
import json
import os
import random
from typing import List, Tuple, Optional, Union
from datetime import datetime, timedelta
from task_specific_perturbation import task_reconstruction, task_sensitive_structural_injection, task_dependent_structural_injection

class TimeSeriesDisturbance:
    def __init__(self, data: Union[List, np.ndarray], all_series):
        self.original_data = np.array(data, dtype=np.float32)
        self.data_mean = np.mean(all_series)
        self.data_std = np.std(all_series)
    
    # 1. Prefix / Suffix Injection
    def add_prefix_constant(self, prefix_length: int, constant_value=0.0):
        prefix = np.full(prefix_length, constant_value, dtype=np.float32)
        return np.concatenate([prefix, self.original_data])
    
    def add_suffix_constant(self, suffix_length: int, constant_value= 0.0):
        suffix = np.full(suffix_length, constant_value, dtype=np.float32)
        return np.concatenate([self.original_data, suffix])
    
    def add_prefix_mean(self, prefix_length: int):
        prefix = np.full(prefix_length, self.data_mean, dtype=np.float32)
        return np.concatenate([prefix, self.original_data])
    
    def add_suffix_mean(self, suffix_length: int):
        suffix = np.full(suffix_length, self.data_mean, dtype=np.float32)
        return np.concatenate([self.original_data, suffix])
    
    # 2. Segment Insertion
    def insert_mean_segment(self, insert_position: int, segment_length: int):
        data_copy = self.original_data.copy()
        mean_segment = np.full(segment_length, self.data_mean, dtype=np.float32)
        
        insert_pos = min(insert_position, len(data_copy))
        
        return np.concatenate([
            data_copy[:insert_pos],
            mean_segment,
            data_copy[insert_pos:]
        ])
    
    def insert_same_segment(self, insert_position: int, segment_length: int):
        if insert_position <= 0:
            base_value = self.original_data[0]
        else:
            base_value = self.original_data[min(insert_position - 1, len(self.original_data) - 1)]

        data_copy = self.original_data.copy()
        same_segment = np.full(segment_length, base_value, dtype=np.float32)
        insert_pos = min(max(insert_position, 0), len(data_copy))
        return np.concatenate([
            data_copy[:insert_pos],
            same_segment,
            data_copy[insert_pos:]
        ])
    
    # 3. Stochastic Noise Injection
    def add_gaussian_noise(self, eta: float):
        noise_mean = self.data_mean
        noise_std = eta * self.data_std
        noise = np.random.normal(0, noise_std, size=self.original_data.shape)
        return self.original_data + noise.astype(np.float32)
    
    def add_random_offset_noise(self, eta: float):
        data_copy = self.original_data.copy()
        n_points = len(data_copy)
        n_disturb = int(eta * n_points)
        disturb_indices = np.random.choice(n_points, n_disturb, replace=False)
        for idx in disturb_indices:
            random_offset = np.random.uniform(-self.data_std, self.data_std)
            data_copy[idx] += random_offset
            
        return data_copy
    
    def add_missing_data(self, eta: float, constant_value):
        data_copy = self.original_data.copy()
        n_points = len(data_copy)
        n_disturb = int(eta * n_points)
        disturb_indices = np.random.choice(n_points, n_disturb, replace=False)
        for idx in disturb_indices:
            data_copy[idx] = self.data_mean#constant_value
        return data_copy

    # 4. Task-specific Structural Disturbances
    def task_sensitive(self, rho: float):
        data_copy = self.original_data.copy()
        series_prime, f, comps = task_sensitive_structural_injection(data_copy, rho=rho)
        return series_prime

    def task_dependent(self, rho: float):
        data_copy = self.original_data.copy()
        series_prime, f, comps = task_dependent_structural_injection(data_copy, rho=rho)
        return series_prime

    def task_reconstruct(self, temp: float, model_name='chronos-t5-tiny'):
        data_copy = self.original_data.copy()
        series_prime, f, comps = task_reconstruction(data_copy, model_name=model_name, temp=temp)
        return series_prime

    # 5. Unified Dispatch Interface
    def apply_disturbance(self, disturbance_type: str, **kwargs):
        disturbance_methods = {
            'prefix_constant': self.add_prefix_constant,
            'suffix_constant': self.add_suffix_constant,
            'prefix_mean': self.add_prefix_mean,
            'suffix_mean': self.add_suffix_mean,
            'insert_mean_segment': self.insert_mean_segment,
            'insert_same_segment': self.insert_same_segment,
            'gaussian_noise': self.add_gaussian_noise,
            'random_offset_noise': self.add_random_offset_noise,
            'missing_data': self.add_missing_data,
            'task_sensitive': self.task_sensitive,
            'task_dependent': self.task_dependent,
            'task_reconstruct': self.task_reconstruct,
        }
        
        if disturbance_type not in disturbance_methods:
            raise ValueError(f"NOT THIS DISTURB TYPE: {disturbance_type}")
        
        disturbed_data = disturbance_methods[disturbance_type](**kwargs)
    
        disturbed_data = np.asarray(disturbed_data, dtype=np.float32)
    
        if disturbed_data.ndim > 1:
            disturbed_data = disturbed_data.flatten()
    
        return disturbed_data

# 6. Disturbance Configuration Generators
def generate_disturbance_when(numbers=10, disturbance_type='gaussian_noise'):
    configs = []
    if 'gaussian_noise' == disturbance_type:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'gaussian_noise',
                'eta': random.choice(np.arange(0, 1.01, 0.05).tolist()),
            })
    elif 'task_sensitive' == disturbance_type:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'task_sensitive',
                'rho': random.choice(np.arange(0, 0.21, 0.01).tolist()),
            })
    else:
        raise ValueError("no imp")
    return configs

def generate_disturbance_configs(numbers=10, disturbance_types=None, eta_list = [0.05, 0.1, 0.2], max_length=8):
    DEFAULT_TYPES = [
        'prefix_constant', 'prefix_mean',
        'suffix_constant', 'suffix_mean',
        'insert_mean_segment', 'insert_same_segment',
        'gaussian_noise', 'random_offset_noise', 'missing_data',
        "task_dependent", 'task_sensitive', 'task_reconstruct'
    ]
    disturbance_types = disturbance_types or DEFAULT_TYPES
    configs = []

    if 'prefix' in disturbance_types:
        for i in range(numbers):
            if random.uniform(0, 1)<0.5:
                configs.append({
                    'disturbance_type': 'prefix_mean',
                    'prefix_length': random.randint(0, max_length)
                })
            else:
                configs.append({
                    'disturbance_type': 'prefix_constant',
                    'prefix_length': random.randint(0, max_length),
                    'constant_value': random.uniform(0, 1)
                })
    if 'suffix' in disturbance_types:
        for i in range(numbers):
            if random.uniform(0, 1)<0.5:
                configs.append({
                    'disturbance_type': 'suffix_mean',
                    'suffix_length': random.randint(0, max_length)
                })
            else:
                configs.append({
                    'disturbance_type': 'suffix_constant',
                    'suffix_length': random.randint(0, max_length),
                    'constant_value': random.uniform(0, 1)
                })
    if 'insert' in disturbance_types:
        for i in range(numbers):
            if random.uniform(0, 1)<0.5:
                configs.append({
                    'disturbance_type': 'insert_mean_segment',
                    'segment_length': random.randint(0, max_length),
                    'insert_ratio': random.uniform(0.3, 0.7)
                })
            else:
                configs.append({
                    'disturbance_type': 'insert_same_segment',
                    'segment_length': random.randint(0, max_length),
                    'insert_ratio': random.uniform(0.3, 0.7)
                })
    if 'gaussian_noise' in disturbance_types:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'gaussian_noise',
                'eta': random.choice(eta_list),
            })
    if 'random_offset_noise' in disturbance_types:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'random_offset_noise',
                'eta': random.choice(eta_list),
            })
    if 'missing_data' in disturbance_types:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'missing_data',
                'eta': random.choice(eta_list),
                'constant_value': 0.0,
            })
    if 'task_sensitive' in disturbance_types:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'task_sensitive',
                'rho': random.uniform(0.01, 0.05),
            })
    if 'task_dependent' in disturbance_types:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'task_dependent',
                'rho': random.uniform(0.01, 0.05),
            })
    if 'task_reconstruct' in disturbance_types:
        for i in range(numbers):
            configs.append({
                'disturbance_type': 'task_reconstruct',
                'temp': random.choice([0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
                "model_name": "chronos-t5-tiny"
            })
    return configs
