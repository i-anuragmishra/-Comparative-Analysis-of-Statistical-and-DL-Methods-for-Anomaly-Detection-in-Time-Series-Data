import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_nab_dataset(dataset_name, data_dir="data/nab"):
    
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

def load_iot_dataset(file_path):
    
    df = pd.read_csv(file_path)
    
    
    timestamp_col = df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    
    return df

def check_irregularities(df):
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
        
    
    time_diffs = df.index.to_series().diff()
    
    
    missing_values = df.isna().sum()
    
    
    unique_time_diffs = time_diffs.unique()
    is_regular = len(unique_time_diffs) <= 2  
    
    return {
        "missing_values": missing_values,
        "is_regular_sampling": is_regular,
        "unique_time_differences": unique_time_diffs,
        "min_diff": time_diffs.min(),
        "max_diff": time_diffs.max(),
        "median_diff": time_diffs.median()
    }

def handle_missing_values(df, method='interpolate', **kwargs):
    
    df_copy = df.copy()
    
    if method == 'interpolate':
        df_copy = df_copy.interpolate(method=kwargs.get('interp_method', 'linear'))
    elif method == 'ffill':
        df_copy = df_copy.ffill()
    elif method == 'bfill':
        df_copy = df_copy.bfill()
    elif method == 'drop':
        df_copy = df_copy.dropna()
    else:
        raise ValueError(f"Unsupported method: {method}")
        
    
    df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
        
    return df_copy

def resample_time_series(df, freq='1H', agg_func='mean'):
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
        
    return df.resample(freq).agg(agg_func)

def preprocess_data(df, scaler=None, return_scaler=False):
    
    df_copy = df.copy()
    
    
    if scaler is None:
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_copy),
            index=df_copy.index,
            columns=df_copy.columns
        )
    else:
        df_scaled = pd.DataFrame(
            scaler.transform(df_copy),
            index=df_copy.index,
            columns=df_copy.columns
        )
    
    if return_scaler:
        return df_scaled, scaler
    else:
        return df_scaled

def create_sequences(data, seq_length):
    
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences) 