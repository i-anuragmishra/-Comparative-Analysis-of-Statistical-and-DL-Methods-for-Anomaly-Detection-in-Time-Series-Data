import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

class ARIMADetector:
    
    
    def __init__(self, order=(5, 1, 0), seasonal_order=None, alpha=0.05):
       
        self.order = order
        self.seasonal_order = seasonal_order
        self.alpha = alpha
        self.model = None
        self.residuals_mean = None
        self.residuals_std = None
        
    def fit(self, series):
        
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
            
       
        residuals = self.model.resid
        self.residuals_mean = residuals.mean()
        self.residuals_std = residuals.std()
        
    def predict(self, series):
        
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        
        try:
           
            predictions = self.model.predict(start=0, end=len(series)-1)
            
            
            if not predictions.index.equals(series.index):
                
                predictions = pd.Series(
                    predictions.values,
                    index=series.index[:len(predictions)]
                )
                
                
                if len(predictions) < len(series):
                    
                    missing_idx = series.index[len(predictions):]
                    extension = pd.Series(
                        [predictions.iloc[-1]] * len(missing_idx),
                        index=missing_idx
                    )
                    predictions = pd.concat([predictions, extension])
                    
        except Exception as e:
            
            
            from statsmodels.tsa.ar_model import AutoReg
            test_model = AutoReg(series[:min(10, len(series))], lags=1).fit()
            predictions = test_model.predict(start=0, end=len(series)-1)
            predictions = pd.Series(predictions.values, index=series.index)
            
        
        z_value = 1.96  
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
       
        predictions_df = self.predict(series)
        
        
        predictions_df = predictions_df.loc[series.index]
        
        
        anomalies = (
            (series < predictions_df['lower_bound']) | 
            (series > predictions_df['upper_bound'])
        )
        
        if return_scores:
            
            scores = np.abs(series - predictions_df['predicted']) / (
                predictions_df['upper_bound'] - predictions_df['lower_bound']
            ) * 2
            return anomalies, scores
        
        return anomalies


class SeasonalDecompositionDetector:
   
    
    def __init__(self, period=None, model='additive', threshold=2.5):
        
        self.period = period
        self.model = model
        self.threshold = threshold
        self.decomposition = None
        self.residuals_mean = None
        self.residuals_std = None
        
    def fit(self, series):
        
        
        self.training_mean = series.mean()
        self.training_std = series.std()
        
        
        if self.period is None:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(series, nlags=min(len(series)//2, 365))
            
            peaks = [i for i in range(1, len(acf_values)-1) 
                   if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]]
            if peaks:
                self.period = peaks[0]
            else:
                self.period = 7  
        
        
        self.decomposition = seasonal_decompose(
            series,
            model=self.model,
            period=self.period
        )
        
        
        residuals = self.decomposition.resid.dropna()
        self.residuals_mean = residuals.mean()
        self.residuals_std = residuals.std()
        
    def detect_anomalies(self, series, return_scores=False):
       
        if self.decomposition is None:
            raise ValueError("Model not fitted yet")
            
        
        
        try:
            
            new_decomposition = seasonal_decompose(
                series,
                model=self.model,
                period=self.period
            )
            
            
            residuals = new_decomposition.resid
            
            
            threshold_value = self.threshold * self.residuals_std
            
            
            anomalies = np.abs(residuals - self.residuals_mean) > threshold_value
            
            
            anomalies = anomalies.fillna(False)
            
            if return_scores:
                
                scores = np.abs(residuals - self.residuals_mean) / self.residuals_std
                return anomalies, scores.fillna(0)
            
            return anomalies
        
        except Exception as e:
            
            print(f"Warning: Error in seasonal decomposition: {e}")
            print(f"Falling back to simple threshold-based detection.")
            
            
            anomalies = pd.Series(False, index=series.index)
            
            
            
            if hasattr(self, 'training_mean') and hasattr(self, 'training_std'):
                anomalies = np.abs(series - self.training_mean) > (self.threshold * self.training_std)
            else:
                
                series_mean = series.mean()
                series_std = series.std()
                anomalies = np.abs(series - series_mean) > (self.threshold * series_std)
            
            if return_scores:
                
                if hasattr(self, 'training_mean') and hasattr(self, 'training_std'):
                    scores = np.abs(series - self.training_mean) / self.training_std
                else:
                    scores = np.abs(series - series_mean) / series_std
                return anomalies, scores
            
            return anomalies


class IsolationForestDetector:
    
    
    def __init__(self, n_estimators=100, contamination=0.05, random_state=42, window_size=10):
       
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.window_size = window_size
        self.model = None
        
    def _create_features(self, series):
       
        df = pd.DataFrame()
        
        
        df['value'] = series
        
        
        df['rolling_mean'] = series.rolling(window=self.window_size).mean()
        df['rolling_std'] = series.rolling(window=self.window_size).std()
        
        
        for i in range(1, min(self.window_size + 1, 6)):  
            df[f'lag_{i}'] = series.shift(i)
            
        
        df = df.dropna()
        
        return df
        
    def fit(self, series):
        
        
        features_df = self._create_features(series)
        
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model.fit(features_df)
        
    def detect_anomalies(self, series, return_scores=False):
        
        if self.model is None:
            raise ValueError("Model not fitted yet")
            
        
        features_df = self._create_features(series)
        
        
        predictions = self.model.predict(features_df)
        anomalies = pd.Series(predictions == -1, index=features_df.index)
        
        if return_scores:
            
            scores = pd.Series(
                -self.model.decision_function(features_df),
                index=features_df.index
            )
            return anomalies, scores
        
        return anomalies 