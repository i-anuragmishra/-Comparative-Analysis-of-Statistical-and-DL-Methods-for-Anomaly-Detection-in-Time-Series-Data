import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from IPython.display import display


from src.data_preparation import (
    load_nab_dataset, 
    load_iot_dataset,
    check_irregularities,
    handle_missing_values,
    resample_time_series,
    preprocess_data
)
from src.models.classical_models import (
    ARIMADetector,
    SeasonalDecompositionDetector,
    IsolationForestDetector
)
from src.models.deep_learning_models import DeepAnomalyDetector
from src.evaluation import (
    calculate_metrics,
    calculate_detection_delay,
    plot_anomalies,
    plot_anomalies_plotly,
    compare_models
)

def parse_args():
    
    parser = argparse.ArgumentParser(description='Time Series Anomaly Detection')
    
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset to use (e.g., nab, iot)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset file')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to anomaly labels file (if available)')
    
    
    parser.add_argument('--impute_method', type=str, default='interpolate',
                       choices=['interpolate', 'ffill', 'bfill', 'drop'],
                       help='Method for handling missing values')
    parser.add_argument('--resample', type=str, default=None,
                       help='Resampling frequency (e.g., "1H" for hourly)')
    
    
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['arima', 'seasonal_decomp', 'isolation_forest', 'lstm_ae'],
                       help='Models to use')
    parser.add_argument('--seq_length', type=int, default=60,
                       help='Sequence length for deep learning models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs for deep learning models')
    
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory for saving results')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to output directory')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive Plotly visualizations')
    
    
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Fraction of data to use for testing (0.0-1.0)')
    parser.add_argument('--cv_folds', type=int, default=0,
                       help='Number of cross-validation folds (0 for no CV)')
    parser.add_argument('--time_split', action='store_true',
                       help='Use time-based train/test split instead of random')
    parser.add_argument('--split_date', type=str, default=None,
                       help='Specific date to split train/test data (YYYY-MM-DD)')
    
    return parser.parse_args()

def load_data(args):
    
    print(f"Loading dataset: {args.dataset} from {args.data_path}")
    
    
    if args.dataset.lower() == 'nab':
        df = load_nab_dataset(args.data_path)
    elif args.dataset.lower() == 'iot':
        df = load_iot_dataset(args.data_path)
    else:
        
        df = pd.read_csv(args.data_path)
        time_col = df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    
    
    labels = None
    if args.labels_path:
        
        if args.dataset.lower() == 'nab' and args.labels_path.endswith('.json'):
            import json
            
            
            with open(args.labels_path, 'r') as f:
                all_labels = json.load(f)
                
            
            import os
            if 'data/nab/data/' in args.data_path:
                
                rel_path = args.data_path.split('data/nab/data/')[-1]
            else:
                
                rel_path = os.path.basename(args.data_path)
                
            
            possible_paths = [
                rel_path,
                'data/' + rel_path,
                'data/nab/data/' + rel_path
            ]
            
            
            file_key = None
            for path in possible_paths:
                if path in all_labels:
                    file_key = path
                    break
            
            if file_key:
                print(f"Found labels for: {file_key}")
                
                windows = all_labels[file_key]
                
                
                labels = pd.Series(False, index=df.index)
                
                
                for window in windows:
                    start = pd.Timestamp(window[0])
                    end = pd.Timestamp(window[1])
                    labels.loc[start:end] = True
                
                print(f"Loaded {labels.sum()} labeled anomaly points")
            else:
                print(f"Warning: No labels found for the current file in {args.labels_path}")
                print(f"Looked for keys: {possible_paths}")
        else:
            
            try:
                labels_df = pd.read_csv(args.labels_path)
                if 'timestamp' in labels_df.columns:
                    labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
                    labels_df = labels_df.set_index('timestamp')
                    labels = labels_df['anomaly'] if 'anomaly' in labels_df.columns else labels_df.iloc[:, 0]
            except Exception as e:
                print(f"Error loading labels: {e}")
    
    
    irregularities = check_irregularities(df)
    print("\nData Irregularities:")
    print(f"Missing Values: {irregularities['missing_values'].sum()}")
    print(f"Regular Sampling: {irregularities['is_regular_sampling']}")
    print(f"Median Time Difference: {irregularities['median_diff']}")
    
    
    if irregularities['missing_values'].sum() > 0:
        print(f"\nHandling missing values with method: {args.impute_method}")
        df = handle_missing_values(df, method=args.impute_method)
    
    
    if args.resample:
        print(f"Resampling data to frequency: {args.resample}")
        df = resample_time_series(df, freq=args.resample)
        
        
        if labels is not None:
            
            all_timestamps = df.index.union(labels.index)
            
            
            df_reindexed = df.reindex(all_timestamps)
            labels_reindexed = labels.reindex(all_timestamps).fillna(False)
            
            
            df = df_reindexed.resample(args.resample).mean()
            labels = labels_reindexed.resample(args.resample).max().astype(bool)  
    
    return df, labels

def train_test_split_data(df, labels, args):
    
    if args.time_split or args.split_date:
        
        if args.split_date:
            
            split_date = pd.Timestamp(args.split_date)
            print(f"Splitting data at date: {split_date}")
        else:
            
            total_time_range = df.index[-1] - df.index[0]
            split_offset = pd.Timedelta(total_time_range * (1 - args.test_split))
            split_date = df.index[0] + split_offset
            print(f"Splitting data at date: {split_date} (time-based {(1-args.test_split)*100:.1f}%/{args.test_split*100:.1f}% split)")
        
        
        train_df = df[df.index < split_date]
        test_df = df[df.index >= split_date]
        
        
        train_labels = None
        test_labels = None
        if labels is not None:
            train_labels = labels[labels.index < split_date]
            test_labels = labels[labels.index >= split_date]
            
            
            if train_labels.sum() == 0:
                print(f"Warning: No anomalies in training set. Consider adjusting split date.")
            if test_labels.sum() == 0:
                print(f"Warning: No anomalies in test set. Consider adjusting split date.")
            
            print(f"Train set: {len(train_df)} points with {train_labels.sum()} anomalies ({train_labels.sum()/len(train_df)*100:.2f}%)")
            print(f"Test set: {len(test_df)} points with {test_labels.sum()} anomalies ({test_labels.sum()/len(test_df)*100:.2f}%)")
    else:
        
        from sklearn.model_selection import train_test_split
        
        
        indices = np.arange(len(df))
        train_indices, test_indices = train_test_split(indices, test_size=args.test_split, random_state=42)
        
        
        train_indices.sort()
        test_indices.sort()
        
        
        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]
        
        
        train_labels = None
        test_labels = None
        if labels is not None:
            train_labels = labels.iloc[train_indices]
            test_labels = labels.iloc[test_indices]
            
            print(f"Train set: {len(train_df)} points with {train_labels.sum()} anomalies ({train_labels.sum()/len(train_df)*100:.2f}%)")
            print(f"Test set: {len(test_df)} points with {test_labels.sum()} anomalies ({test_labels.sum()/len(test_df)*100:.2f}%)")
    
    return train_df, test_df, train_labels, test_labels

def create_time_series_cv_splits(df, labels, n_splits=5):
    
    total_rows = len(df)
    test_size = total_rows // n_splits
    
    cv_splits = []
    
    for i in range(n_splits):
        
        test_start = i * test_size
        test_end = (i + 1) * test_size if i < n_splits - 1 else total_rows
        
        
        test_indices = list(range(test_start, test_end))
        train_indices = [j for j in range(total_rows) if j not in test_indices]
        
        
        train_df = df.iloc[train_indices]
        test_df = df.iloc[test_indices]
        
        
        train_labels = None
        test_labels = None
        if labels is not None:
            train_labels = labels.iloc[train_indices]
            test_labels = labels.iloc[test_indices]
        
        cv_splits.append((train_df, test_df, train_labels, test_labels))
    
    return cv_splits

def train_model(model_name, series, args):
    
    if model_name == 'arima':
        detector = ARIMADetector(order=(5, 1, 0))
        detector.fit(series)
    elif model_name == 'seasonal_decomp':
        detector = SeasonalDecompositionDetector()
        detector.fit(series)
    elif model_name == 'isolation_forest':
        detector = IsolationForestDetector()
        detector.fit(series)
    elif model_name == 'lstm_ae':
        detector = DeepAnomalyDetector(
            model_type='lstm_ae',
            seq_length=args.seq_length,
            num_epochs=args.epochs
        )
        detector.fit(series)
    elif model_name == 'tcn':
        detector = DeepAnomalyDetector(
            model_type='tcn',
            seq_length=args.seq_length,
            num_epochs=args.epochs
        )
        detector.fit(series)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return detector

def train_and_evaluate_models(df, labels, args):
    
    results = {}
    model_objects = {}
    all_metrics = {}
    
    
    series = df.iloc[:, 0] if df.shape[1] > 1 else df.iloc[:, 0]
    
    
    if args.test_split > 0 or args.split_date:
        print("\n===== Using Train/Test Split =====")
        train_df, test_df, train_labels, test_labels = train_test_split_data(df, labels, args)
        
        
        train_series = train_df.iloc[:, 0] if train_df.shape[1] > 1 else train_df.iloc[:, 0]
        test_series = test_df.iloc[:, 0] if test_df.shape[1] > 1 else test_df.iloc[:, 0]
        
        
        for model_name in args.models:
            print(f"\nTraining {model_name.replace('_', ' ').title()} model...")
            
            
            detector = train_model(model_name, train_series, args)
            model_objects[model_name] = detector
            
            
            formatted_name = model_name.replace('_', ' ').title()
            print(f"Evaluating {formatted_name} on test data...")
            anomalies = detector.detect_anomalies(test_series)
            results[formatted_name] = anomalies
            
            
            if test_labels is not None:
                print(f"\n=== {formatted_name} (Test Set) ===")
                
                
                if isinstance(anomalies, pd.Series) and isinstance(test_labels, pd.Series):
                    aligned = pd.concat([anomalies, test_labels], axis=1).fillna(False)
                    metrics = calculate_metrics(aligned.iloc[:, 1], aligned.iloc[:, 0])
                else:
                    metrics = calculate_metrics(test_labels, anomalies)
                
                
                all_metrics[formatted_name] = metrics
    
    
    elif args.cv_folds > 1:
        print(f"\n===== Using {args.cv_folds}-Fold Cross-Validation =====")
        cv_splits = create_time_series_cv_splits(df, labels, n_splits=args.cv_folds)
        
        
        cv_metrics = {model: [] for model in args.models}
        
        
        for fold, (train_df, test_df, train_labels, test_labels) in enumerate(cv_splits):
            print(f"\n----- Fold {fold+1}/{args.cv_folds} -----")
            train_series = train_df.iloc[:, 0] if train_df.shape[1] > 1 else train_df.iloc[:, 0]
            test_series = test_df.iloc[:, 0] if test_df.shape[1] > 1 else test_df.iloc[:, 0]
            
            
            for model_name in args.models:
                print(f"Training {model_name.replace('_', ' ').title()} model...")
                
                
                detector = train_model(model_name, train_series, args)
                anomalies = detector.detect_anomalies(test_series)
                
                
                if test_labels is not None:
                    formatted_name = model_name.replace('_', ' ').title()
                    
                    
                    if isinstance(anomalies, pd.Series) and isinstance(test_labels, pd.Series):
                        aligned = pd.concat([anomalies, test_labels], axis=1).fillna(False)
                        metrics = calculate_metrics(aligned.iloc[:, 1], aligned.iloc[:, 0], verbose=False)
                    else:
                        metrics = calculate_metrics(test_labels, anomalies, verbose=False)
                    
                    
                    cv_metrics[model_name].append(metrics)
        
        
        print("\n===== Cross-Validation Results =====")
        for model_name in args.models:
            formatted_name = model_name.replace('_', ' ').title()
            print(f"\n=== {formatted_name} (Average across {args.cv_folds} folds) ===")
            
            
            if len(cv_metrics[model_name]) > 0:
                
                avg_metrics = {}
                for metric in cv_metrics[model_name][0].keys():
                    values = [fold_metrics[metric] for fold_metrics in cv_metrics[model_name]]
                    avg_metrics[metric] = sum(values) / len(values)
                
                
                print("-" * 40)
                print("Performance Metrics:")
                print("-" * 40)
                print(f"Precision:            {avg_metrics['precision']:.4f}")
                print(f"Recall (Sensitivity): {avg_metrics['recall']:.4f}")
                print(f"F1 Score:             {avg_metrics['f1_score']:.4f}")
                print("-" * 40)
                print("Error Rates:")
                print("-" * 40)
                print(f"False Positive Rate:  {avg_metrics['false_positive_rate']:.4f}")
                print(f"False Negative Rate:  {avg_metrics['false_negative_rate']:.4f}")
                print("-" * 40)
                
                
                all_metrics[formatted_name] = avg_metrics
    
    
    else:
        print("\n===== Using Full Dataset (No train/test split) =====")
        
        for model_name in args.models:
            formatted_name = model_name.replace('_', ' ').title()
            print(f"\nTraining {formatted_name} model...")
            
            
            detector = train_model(model_name, series, args)
            model_objects[formatted_name] = detector
            anomalies = detector.detect_anomalies(series)
            results[formatted_name] = anomalies
            
            
            if labels is not None:
                print(f"\n=== {formatted_name} ===")
                
                
                if isinstance(anomalies, pd.Series) and isinstance(labels, pd.Series):
                    aligned = pd.concat([anomalies, labels], axis=1).fillna(False)
                    metrics = calculate_metrics(aligned.iloc[:, 1], aligned.iloc[:, 0])
                else:
                    metrics = calculate_metrics(labels, anomalies)
                
                
                all_metrics[formatted_name] = metrics
    
    
    if args.test_split > 0 or args.cv_folds > 1:
        print("\n======== SUMMARY OF EVALUATION METRICS ========")
        for model_name, metrics in all_metrics.items():
            print(f"\n{model_name} - F1 Score: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    
    
    if labels is not None and all_metrics and args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        
        
        eval_method = "full"
        if args.test_split > 0:
            eval_method = "test_split"
        elif args.cv_folds > 1:
            eval_method = f"cv{args.cv_folds}"
            
        metrics_file = os.path.join(args.output_dir, f"{args.dataset}_{eval_method}_metrics.json")
        
        
        serializable_metrics = {}
        for model, metrics_dict in all_metrics.items():
            serializable_metrics[model] = {k: float(v) for k, v in metrics_dict.items()}
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        print(f"\nMetrics saved to: {metrics_file}")
    elif labels is None:
        print("\nNo labels provided for evaluation. Skipping metrics calculation.")
    
    
    if args.interactive and results:
        fig, _ = compare_models(
            test_series if args.test_split > 0 else series, 
            results, 
            ground_truth=test_labels if args.test_split > 0 else labels, 
            title=f"Anomaly Detection Results - {args.dataset}"
        )
        if args.save_plots:
            os.makedirs(args.output_dir, exist_ok=True)
            fig.write_html(os.path.join(args.output_dir, f"{args.dataset}_comparison.html"))
        fig.show()
    elif results:
        
        for model_name, anomalies in results.items():
            plot_series = test_series if args.test_split > 0 else series
            plot_labels = test_labels if args.test_split > 0 else labels
            
            fig = plot_anomalies(
                plot_series, 
                anomalies, 
                title=f"{model_name} - {args.dataset}",
                ground_truth=plot_labels
            )
            if args.save_plots:
                os.makedirs(args.output_dir, exist_ok=True)
                fig.savefig(os.path.join(args.output_dir, f"{args.dataset}_{model_name.replace(' ', '_')}.png"))
            plt.show()
    
    return results, model_objects

def main():
    
    args = parse_args()
    
    
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    
    df, labels = load_data(args)
    
    
    print("\nData Sample:")
    print(df.head())
    
    if labels is not None:
        print("\nLabels Sample:")
        print(labels.head())
        print(f"Total anomalies in ground truth: {labels.sum()}")
    
    
    results, models = train_and_evaluate_models(df, labels, args)
    
    print("\nAnomaly Detection Pipeline Complete!")

if __name__ == "__main__":
    main() 