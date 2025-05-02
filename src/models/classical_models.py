import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

class ARIMADetector:
    """
    Anomaly detector based on ARIMA model.
    """
    
    def __init__(self, order=(5, 1, 0), seasonal_order=None, alpha=0.05):
        """
        Initialize ARIMA detector.
        
        Parameters:
        -----------
        order: tuple
            ARIMA order (p, d, q)
        seasonal_order: tuple, optional
            Seasonal order (P, D, Q, s)
        alpha: float
            Significance level for anomaly threshold
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.alpha = alpha
        self.model = None
        self.residuals_mean = None
        self.residuals_std = None
        
    def fit(self, series):
        """
        Fit ARIMA model to time series.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
        """
        if self.seasonal_order:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order
            ).fit(disp=False)
        else:
            self.model = ARIMA(
                series,
                order=self.order
            ).fit()
            
        # Calculate residuals statistics for anomaly detection
        residuals = self.model.resid
        self.residuals_mean = residuals.mean()
        self.residuals_std = residuals.std()
        
    def predict(self, series):
        """
        Generate predictions and confidence intervals.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions and confidence intervals
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        # Make predictions
        # For train-test split compatibility, we need to handle different indices
        try:
            # Try predicting directly on the given index
            predictions = self.model.predict(start=0, end=len(series)-1)
            
            # Realign with original series index
            if not predictions.index.equals(series.index):
                # Create a new index-aligned Series
                predictions = pd.Series(
                    predictions.values,
                    index=series.index[:len(predictions)]
                )
                
                # If there are still length mismatches, handle them 
                if len(predictions) < len(series):
                    # Extend predictions to match series length
                    missing_idx = series.index[len(predictions):]
                    extension = pd.Series(
                        [predictions.iloc[-1]] * len(missing_idx),
                        index=missing_idx
                    )
                    predictions = pd.concat([predictions, extension])
                    
        except Exception as e:
            # Fallback: create predictions with the series index
            # Train a simple AR model on the first few points of the test data
            from statsmodels.tsa.ar_model import AutoReg
            test_model = AutoReg(series[:min(10, len(series))], lags=1).fit()
            predictions = test_model.predict(start=0, end=len(series)-1)
            predictions = pd.Series(predictions.values, index=series.index)
            
        # Compute confidence intervals (mean ± z*std)
        z_value = 1.96  # 95% confidence interval
        std_err = np.sqrt(self.model.mse)
        lower_bound = predictions - z_value * std_err
        upper_bound = predictions + z_value * std_err
        
        return pd.DataFrame({
            'actual': series,
            'predicted': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    
    def detect_anomalies(self, series, return_scores=False):
        """
        Detect anomalies based on prediction intervals.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
        return_scores: bool
            Whether to return anomaly scores
            
        Returns:
        --------
        pd.Series or tuple
            Boolean series indicating anomalies, and optionally anomaly scores
        """
        predictions_df = self.predict(series)
        
        # Ensure all indices are aligned
        predictions_df = predictions_df.loc[series.index]
        
        # Points outside prediction intervals are anomalies
        anomalies = (
            (series < predictions_df['lower_bound']) | 
            (series > predictions_df['upper_bound'])
        )
        
        if return_scores:
            # Calculate anomaly scores as normalized deviations
            scores = np.abs(series - predictions_df['predicted']) / (
                predictions_df['upper_bound'] - predictions_df['lower_bound']
            ) * 2
            return anomalies, scores
        
        return anomalies


class SeasonalDecompositionDetector:
    """
    Anomaly detector based on seasonal decomposition.
    """
    
    def __init__(self, period=None, model='additive', threshold=2.5):
        """
        Initialize seasonal decomposition detector.
        
        Parameters:
        -----------
        period: int, optional
            Period of seasonality
        model: str
            Type of seasonal model ('additive' or 'multiplicative')
        threshold: float
            Threshold for anomaly detection (in standard deviations)
        """
        self.period = period
        self.model = model
        self.threshold = threshold
        self.decomposition = None
        self.residuals_mean = None
        self.residuals_std = None
        
    def fit(self, series):
        """
        Fit seasonal decomposition to time series.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
        """
        # Store training data statistics for fallback detection
        self.training_mean = series.mean()
        self.training_std = series.std()
        
        # Auto-detect period if not provided
        if self.period is None:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(series, nlags=min(len(series)//2, 365))
            # Find first peak after lag 0
            peaks = [i for i in range(1, len(acf_values)-1) 
                   if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]]
            if peaks:
                self.period = peaks[0]
            else:
                self.period = 7  # Default to weekly
        
        # Perform decomposition
        self.decomposition = seasonal_decompose(
            series,
            model=self.model,
            period=self.period
        )
        
        # Calculate residuals statistics
        residuals = self.decomposition.resid.dropna()
        self.residuals_mean = residuals.mean()
        self.residuals_std = residuals.std()
        
    def detect_anomalies(self, series, return_scores=False):
        """
        Detect anomalies based on residuals.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
        return_scores: bool
            Whether to return anomaly scores
            
        Returns:
        --------
        pd.Series or tuple
            Boolean series indicating anomalies, and optionally anomaly scores
        """
        if self.decomposition is None:
            raise ValueError("Model not fitted yet")
            
        # For test data, we need to decompose it separately
        # since seasonal decomposition is not a parametric model
        try:
            # Decompose the new data
            new_decomposition = seasonal_decompose(
                series,
                model=self.model,
                period=self.period
            )
            
            # Calculate residuals based on the new decomposition
            residuals = new_decomposition.resid
            
            # Points with residuals beyond threshold*std are anomalies
            threshold_value = self.threshold * self.residuals_std
            
            # Use the statistics from the training data for consistency
            anomalies = np.abs(residuals - self.residuals_mean) > threshold_value
            
            # Handle NaNs in residuals (at edges)
            anomalies = anomalies.fillna(False)
            
            if return_scores:
                # Calculate anomaly scores as normalized residuals
                scores = np.abs(residuals - self.residuals_mean) / self.residuals_std
                return anomalies, scores.fillna(0)
            
            return anomalies
        
        except Exception as e:
            # Fallback for when decomposition fails (e.g., test series too short)
            print(f"Warning: Error in seasonal decomposition: {e}")
            print(f"Falling back to simple threshold-based detection.")
            
            # Create empty anomaly series
            anomalies = pd.Series(False, index=series.index)
            
            # Use simple thresholding on the raw values 
            # based on training data statistics
            if hasattr(self, 'training_mean') and hasattr(self, 'training_std'):
                anomalies = np.abs(series - self.training_mean) > (self.threshold * self.training_std)
            else:
                # Calculate mean and std on this series as a last resort
                series_mean = series.mean()
                series_std = series.std()
                anomalies = np.abs(series - series_mean) > (self.threshold * series_std)
            
            if return_scores:
                # Simple scores based on deviation from mean
                if hasattr(self, 'training_mean') and hasattr(self, 'training_std'):
                    scores = np.abs(series - self.training_mean) / self.training_std
                else:
                    scores = np.abs(series - series_mean) / series_std
                return anomalies, scores
            
            return anomalies


class IsolationForestDetector:
    """
    Anomaly detector based on Isolation Forest algorithm.
    """
    
    def __init__(self, n_estimators=100, contamination=0.05, random_state=42, window_size=10):
        """
        Initialize Isolation Forest detector.
        
        Parameters:
        -----------
        n_estimators: int
            Number of trees in the forest
        contamination: float
            Expected proportion of anomalies
        random_state: int
            Random seed for reproducibility
        window_size: int
            Size of rolling window for feature creation
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.window_size = window_size
        self.model = None
        
    def _create_features(self, series):
        """
        Create features from time series for Isolation Forest.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with engineered features
        """
        df = pd.DataFrame()
        
        # Original value
        df['value'] = series
        
        # Add rolling statistics
        df['rolling_mean'] = series.rolling(window=self.window_size).mean()
        df['rolling_std'] = series.rolling(window=self.window_size).std()
        
        # Add lagged values
        for i in range(1, min(self.window_size + 1, 6)):  # Limit to 5 lags max
            df[f'lag_{i}'] = series.shift(i)
            
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
        
    def fit(self, series):
        """
        Fit Isolation Forest to time series.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
        """
        # Create features
        features_df = self._create_features(series)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(features_df)
        
    def detect_anomalies(self, series, return_scores=False):
        """
        Detect anomalies using Isolation Forest.
        
        Parameters:
        -----------
        series: pd.Series
            Time series data
        return_scores: bool
            Whether to return anomaly scores
            
        Returns:
        --------
        pd.Series or tuple
            Boolean series indicating anomalies, and optionally anomaly scores
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        # Create features
        features_df = self._create_features(series)
        
        # Detect anomalies
        predictions = self.model.predict(features_df)
        anomalies = pd.Series(predictions == -1, index=features_df.index)
        
        if return_scores:
            # Calculate anomaly scores
            scores = pd.Series(
                -self.model.decision_function(features_df),
                index=features_df.index
            )
            return anomalies, scores
        
        return anomalies 