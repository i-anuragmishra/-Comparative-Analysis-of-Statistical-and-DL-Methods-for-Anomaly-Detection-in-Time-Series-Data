import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true, y_pred, verbose=True):
    """
    Calculate evaluation metrics for anomaly detection.
    
    Parameters:
    -----------
    y_true: array-like
        Ground truth anomaly labels
    y_pred: array-like
        Predicted anomaly labels
    verbose: bool
        Whether to print the metrics
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Calculate basic metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate NAB-like metrics (detection delay isn't directly captured)
    fpr = fp / (fp + tn)  # False positive rate
    fnr = fn / (fn + tp)  # False negative rate
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr
    }
    
    if verbose:
        print("-" * 40)
        print("Performance Metrics:")
        print("-" * 40)
        print(f"Precision:            {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"F1 Score:             {f1:.4f}")
        print("-" * 40)
        print("Confusion Matrix:")
        print("-" * 40)
        print(f"True Positives:       {tp}")
        print(f"False Positives:      {fp}")
        print(f"True Negatives:       {tn}")
        print(f"False Negatives:      {fn}")
        print("-" * 40)
        print("Error Rates:")
        print("-" * 40)
        print(f"False Positive Rate:  {fpr:.4f}")
        print(f"False Negative Rate:  {fnr:.4f}")
        print("-" * 40)
    
    return metrics

def calculate_detection_delay(anomaly_timestamps, detected_timestamps, max_delay=None):
    """
    Calculate detection delay for anomalies.
    
    Parameters:
    -----------
    anomaly_timestamps: list or array-like
        Timestamps of true anomalies
    detected_timestamps: list or array-like
        Timestamps of detected anomalies
    max_delay: float or timedelta, optional
        Maximum allowed delay for detection
        
    Returns:
    --------
    float
        Average detection delay
    float
        Detection rate (percentage of anomalies detected within max_delay)
    """
    if len(anomaly_timestamps) == 0:
        return 0, 0
    
    detected_anomalies = 0
    total_delay = 0
    
    for anomaly_time in anomaly_timestamps:
        if len(detected_timestamps) == 0:
            continue
            
        # Find closest detection after the anomaly
        detection_times = [t for t in detected_timestamps if t >= anomaly_time]
        
        if detection_times:
            closest_detection = min(detection_times)
            delay = closest_detection - anomaly_time
            
            # Check if detection is within max_delay
            if max_delay is None or delay <= max_delay:
                detected_anomalies += 1
                total_delay += delay
    
    # Calculate average delay and detection rate
    avg_delay = total_delay / detected_anomalies if detected_anomalies > 0 else float('inf')
    detection_rate = detected_anomalies / len(anomaly_timestamps)
    
    return avg_delay, detection_rate

def plot_anomalies(series, anomalies, title="Anomaly Detection Results", 
                   threshold=None, scores=None, ground_truth=None):
    """
    Plot time series with detected anomalies using matplotlib.
    
    Parameters:
    -----------
    series: pd.Series
        Time series data
    anomalies: pd.Series or np.ndarray
        Boolean array/series indicating anomalies
    title: str
        Plot title
    threshold: float, optional
        Anomaly threshold for plotting
    scores: pd.Series or np.ndarray, optional
        Anomaly scores
    ground_truth: pd.Series or np.ndarray, optional
        Ground truth anomaly labels
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original series
    ax.plot(series.index, series.values, label='Time Series', color='blue')
    
    # Plot anomalies
    if isinstance(anomalies, pd.Series):
        # Make sure indices are aligned
        if not anomalies.index.equals(series.index):
            # Reindex to match the series index
            aligned_anomalies = pd.Series(False, index=series.index)
            aligned_anomalies.loc[anomalies.index.intersection(series.index)] = anomalies.loc[anomalies.index.intersection(series.index)]
            anomaly_points = series[aligned_anomalies]
        else:
            anomaly_points = series[anomalies]
    else:
        anomaly_points = series.iloc[anomalies]
    
    ax.scatter(anomaly_points.index, anomaly_points.values, color='red', 
               label='Detected Anomalies', s=50)
    
    # Plot ground truth if available
    if ground_truth is not None:
        if isinstance(ground_truth, pd.Series):
            # Ensure ground truth aligns with series
            aligned_ground_truth = pd.Series(False, index=series.index)
            aligned_ground_truth.loc[ground_truth.index.intersection(series.index)] = ground_truth.loc[ground_truth.index.intersection(series.index)]
            truth_points = series[aligned_ground_truth]
        else:
            truth_points = series.iloc[ground_truth]
            
        ax.scatter(truth_points.index, truth_points.values, 
                    color='green', marker='x', label='True Anomalies', s=50)
    
    # Plot anomaly scores if available
    if scores is not None:
        ax2 = ax.twinx()
        if isinstance(scores, pd.Series):
            ax2.plot(scores.index, scores.values, label='Anomaly Score', 
                     color='purple', alpha=0.5)
        else:
            ax2.plot(series.index, scores, label='Anomaly Score', 
                     color='purple', alpha=0.5)
            
        if threshold is not None:
            ax2.axhline(y=threshold, color='orange', linestyle='--', 
                        label=f'Threshold ({threshold:.4f})')
        
        ax2.set_ylabel('Anomaly Score')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')
    else:
        ax.legend(loc='best')
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    plt.tight_layout()
    return fig

def plot_anomalies_plotly(series, anomalies, title="Anomaly Detection Results", 
                         threshold=None, scores=None, ground_truth=None):
    """
    Plot time series with detected anomalies using Plotly for interactive visualization.
    
    Parameters:
    -----------
    series: pd.Series
        Time series data
    anomalies: pd.Series or np.ndarray
        Boolean array/series indicating anomalies
    title: str
        Plot title
    threshold: float, optional
        Anomaly threshold for plotting
    scores: pd.Series or np.ndarray, optional
        Anomaly scores
    ground_truth: pd.Series or np.ndarray, optional
        Ground truth anomaly labels
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create figure with secondary y-axis for scores
    if scores is not None:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
    
    # Add time series
    fig.add_trace(
        go.Scatter(
            x=series.index, 
            y=series.values,
            mode='lines',
            name='Time Series',
            line=dict(color='blue')
        )
    )
    
    # Add detected anomalies
    if isinstance(anomalies, pd.Series):
        # Make sure indices are aligned
        if not anomalies.index.equals(series.index):
            # Reindex to match the series index
            aligned_anomalies = pd.Series(False, index=series.index)
            aligned_anomalies.loc[anomalies.index.intersection(series.index)] = anomalies.loc[anomalies.index.intersection(series.index)]
            anomaly_points = series[aligned_anomalies]
        else:
            anomaly_points = series[anomalies]
    else:
        anomaly_points = series.iloc[anomalies]
        
    fig.add_trace(
        go.Scatter(
            x=anomaly_points.index,
            y=anomaly_points.values,
            mode='markers',
            name='Detected Anomalies',
            marker=dict(color='red', size=10)
        )
    )
    
    # Add ground truth if available
    if ground_truth is not None:
        if isinstance(ground_truth, pd.Series):
            # Ensure ground truth aligns with series
            aligned_ground_truth = pd.Series(False, index=series.index)
            aligned_ground_truth.loc[ground_truth.index.intersection(series.index)] = ground_truth.loc[ground_truth.index.intersection(series.index)]
            truth_points = series[aligned_ground_truth]
        else:
            truth_points = series.iloc[ground_truth]
            
        fig.add_trace(
            go.Scatter(
                x=truth_points.index,
                y=truth_points.values,
                mode='markers',
                name='True Anomalies',
                marker=dict(color='green', size=10, symbol='x')
            )
        )
    
    # Add anomaly scores if available
    if scores is not None:
        if isinstance(scores, pd.Series):
            score_x = scores.index
            score_y = scores.values
        else:
            score_x = series.index
            score_y = scores
            
        fig.add_trace(
            go.Scatter(
                x=score_x,
                y=score_y,
                mode='lines',
                name='Anomaly Score',
                line=dict(color='purple', width=1),
                opacity=0.7
            ),
            secondary_y=True
        )
        
        # Add threshold line if available
        if threshold is not None:
            fig.add_trace(
                go.Scatter(
                    x=[score_x[0], score_x[-1]],
                    y=[threshold, threshold],
                    mode='lines',
                    name=f'Threshold ({threshold:.4f})',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                secondary_y=True
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest'
    )
    
    if scores is not None:
        fig.update_yaxes(title_text="Anomaly Score", secondary_y=True)
    
    return fig

def compare_models(series, model_results, ground_truth=None, title="Model Comparison"):
    """
    Compare multiple anomaly detection models.
    
    Parameters:
    -----------
    series: pd.Series
        Time series data
    model_results: dict
        Dictionary of model results {model_name: anomalies}
    ground_truth: pd.Series or np.ndarray, optional
        Ground truth anomaly labels
    title: str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    dict
        Dictionary of evaluation metrics by model
    """
    # Create figure
    fig = go.Figure()
    
    # Add time series
    fig.add_trace(
        go.Scatter(
            x=series.index, 
            y=series.values,
            mode='lines',
            name='Time Series',
            line=dict(color='black', width=1)
        )
    )
    
    # Add results for each model
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    metrics = {}
    
    for i, (model_name, anomalies) in enumerate(model_results.items()):
        color = colors[i % len(colors)]
        
        # Extract anomaly points - fix for index alignment issue
        if isinstance(anomalies, pd.Series):
            # Make sure indices are aligned
            if not anomalies.index.equals(series.index):
                # Reindex to match the series index
                aligned_anomalies = pd.Series(False, index=series.index)
                aligned_anomalies.loc[anomalies.index.intersection(series.index)] = anomalies.loc[anomalies.index.intersection(series.index)]
                anomaly_points = series[aligned_anomalies]
            else:
                anomaly_points = series[anomalies]
        else:
            # For numpy arrays
            anomaly_points = series.iloc[anomalies]
            
        # Add to plot
        fig.add_trace(
            go.Scatter(
                x=anomaly_points.index,
                y=anomaly_points.values,
                mode='markers',
                name=f'{model_name} Anomalies',
                marker=dict(color=color, size=8, symbol='circle')
            )
        )
        
        # Calculate metrics if ground truth available
        if ground_truth is not None:
            if isinstance(anomalies, pd.Series) and isinstance(ground_truth, pd.Series):
                # Align indices
                aligned_data = pd.concat([anomalies, ground_truth], axis=1).fillna(False)
                metrics[model_name] = calculate_metrics(
                    aligned_data.iloc[:, 1], 
                    aligned_data.iloc[:, 0],
                    verbose=False
                )
            else:
                metrics[model_name] = calculate_metrics(
                    ground_truth, 
                    anomalies,
                    verbose=False
                )
    
    # Add ground truth if available
    if ground_truth is not None:
        if isinstance(ground_truth, pd.Series):
            # Ensure ground truth aligns with series
            aligned_ground_truth = pd.Series(False, index=series.index)
            aligned_ground_truth.loc[ground_truth.index.intersection(series.index)] = ground_truth.loc[ground_truth.index.intersection(series.index)]
            truth_points = series[aligned_ground_truth]
        else:
            truth_points = series.iloc[ground_truth]
            
        fig.add_trace(
            go.Scatter(
                x=truth_points.index,
                y=truth_points.values,
                mode='markers',
                name='True Anomalies',
                marker=dict(color='black', size=10, symbol='x')
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest'
    )
    
    return fig, metrics 