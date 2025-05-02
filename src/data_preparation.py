import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_nab_dataset(dataset_name, data_dir="data/nab"):
    """
    Load a dataset from the Numenta Anomaly Benchmark (NAB).
    
    Parameters:
    -----------
    dataset_name: str
        Name of the dataset to load, e.g., 'realKnownCause/ambient_temperature_system_failure.csv'
    data_dir: str
        Directory where NAB data is stored
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamp index and value column
    """
    file_path = os.path.join(data_dir, dataset_name)
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    return df

def load_iot_dataset(file_path):
    """
    Load an IoT dataset from a CSV file.
    
    Parameters:
    -----------
    file_path: str
        Path to the IoT dataset CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with timestamp index and sensor values
    """
    df = pd.read_csv(file_path)
    
    # Assuming the first column is timestamp - adjust as needed
    timestamp_col = df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.set_index(timestamp_col)
    
    return df

def check_irregularities(df):
    """
    Check for irregularities in the time series data.
    
    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame with timestamp index
        
    Returns:
    --------
    dict
        Dictionary with statistics about data irregularities
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
        
    # Calculate time differences
    time_diffs = df.index.to_series().diff()
    
    # Check for missing values
    missing_values = df.isna().sum()
    
    # Check time sampling
    unique_time_diffs = time_diffs.unique()
    is_regular = len(unique_time_diffs) <= 2  # One value + possibly NaN for first entry
    
    return {
        "missing_values": missing_values,
        "is_regular_sampling": is_regular,
        "unique_time_differences": unique_time_diffs,
        "min_diff": time_diffs.min(),
        "max_diff": time_diffs.max(),
        "median_diff": time_diffs.median()
    }

def handle_missing_values(df, method='interpolate', **kwargs):
    """
    Handle missing values in the DataFrame.
    
    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame with missing values
    method: str
        Method to handle missing values: 'interpolate', 'ffill', 'bfill', or 'drop'
    kwargs:
        Additional arguments for the chosen method
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
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
        
    # Handle any remaining NaNs at the beginning or end
    df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
        
    return df_copy

def resample_time_series(df, freq='1H', agg_func='mean'):
    """
    Resample time series to a regular frequency.
    
    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame with DatetimeIndex
    freq: str
        Target frequency (e.g., '1H' for hourly, '1D' for daily)
    agg_func: str or dict
        Aggregation function(s) to use for resampling
        
    Returns:
    --------
    pd.DataFrame
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
        
    return df.resample(freq).agg(agg_func)

def preprocess_data(df, scaler=None, return_scaler=False):
    """
    Preprocess data for anomaly detection models.
    
    Parameters:
    -----------
    df: pd.DataFrame
        Input data
    scaler: sklearn.preprocessing.Scaler, optional
        Scaler to use for normalization
    return_scaler: bool
        Whether to return the scaler along with the scaled data
        
    Returns:
    --------
    pd.DataFrame or tuple
        Scaled DataFrame, and optionally the scaler
    """
    df_copy = df.copy()
    
    # Create or use scaler
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
    """
    Create sequences for sequence-based models like LSTM.
    
    Parameters:
    -----------
    data: np.ndarray
        Input data array
    seq_length: int
        Length of each sequence
        
    Returns:
    --------
    np.ndarray
        Array of sequences
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences) 