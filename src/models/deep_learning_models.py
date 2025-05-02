import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time series anomaly detection.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        """
        Initialize LSTM Autoencoder.
        
        Parameters:
        -----------
        input_dim: int
            Dimensionality of input features
        hidden_dim: int
            Dimensionality of hidden layers
        num_layers: int
            Number of LSTM layers
        dropout: float
            Dropout rate
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Reconstructed input of shape (batch_size, seq_len, input_dim)
        """
        # Encode
        _, (hidden, _) = self.encoder(x)
        
        # Repeat hidden state for each time step
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Decode
        decoder_output, _ = self.decoder(hidden_repeated)
        
        # Project to original dimension
        output = self.output_layer(decoder_output)
        
        return output


class TCN(nn.Module):
    """
    Temporal Convolutional Network for time series anomaly detection.
    """
    
    class CausalConv1d(nn.Module):
        """
        1D causal convolution with dilation.
        """
        
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super(TCN.CausalConv1d, self).__init__()
            
            # Compute padding needed for causal convolution
            self.padding = (kernel_size - 1) * dilation
            
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                dilation=dilation
            )
            
        def forward(self, x):
            """Forward pass with causal padding removal."""
            x = self.conv(x)
            if self.padding != 0:
                return x[:, :, :-self.padding]
            return x
    
    class ResidualBlock(nn.Module):
        """
        Residual block with dilated causal convolutions.
        """
        
        def __init__(self, channels, kernel_size, dilation):
            super(TCN.ResidualBlock, self).__init__()
            
            self.causal_conv1 = TCN.CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation
            )
            self.causal_conv2 = TCN.CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation
            )
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            """Forward pass with residual connection."""
            residual = x
            
            out = self.relu(self.causal_conv1(x))
            out = self.dropout(out)
            out = self.relu(self.causal_conv2(out))
            out = self.dropout(out)
            
            return out + residual
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, kernel_size=3):
        """
        Initialize TCN.
        
        Parameters:
        -----------
        input_dim: int
            Dimensionality of input features
        hidden_dim: int
            Dimensionality of hidden layers
        output_dim: int
            Dimensionality of output
        num_layers: int
            Number of residual blocks
        kernel_size: int
            Kernel size for convolutions
        """
        super(TCN, self).__init__()
        
        # Input layer
        self.input_layer = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )
        
        # Residual blocks with exponentially increasing dilations
        self.residual_blocks = nn.ModuleList([
            self.ResidualBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2**i
            )
            for i in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=1
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        # Transpose to (batch_size, input_dim, seq_len) for 1D convolution
        x = x.transpose(1, 2)
        
        # Input layer
        x = self.input_layer(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Transpose back to (batch_size, seq_len, output_dim)
        x = x.transpose(1, 2)
        
        return x


class DeepAnomalyDetector:
    """
    Base class for deep learning-based anomaly detectors.
    """
    
    def __init__(
        self, 
        model_type='lstm_ae',
        seq_length=60,
        hidden_dim=64,
        num_epochs=50,
        batch_size=32,
        learning_rate=0.001,
        device=None,
        threshold_multiplier=3.0
    ):
        """
        Initialize deep anomaly detector.
        
        Parameters:
        -----------
        model_type: str
            Type of model ('lstm_ae' or 'tcn')
        seq_length: int
            Length of input sequences
        hidden_dim: int
            Dimensionality of hidden layers
        num_epochs: int
            Number of training epochs
        batch_size: int
            Batch size for training
        learning_rate: float
            Learning rate for optimizer
        device: str or torch.device
            Device for computation ('cpu' or 'cuda')
        threshold_multiplier: float
            Multiplier for anomaly threshold
        """
        self.model_type = model_type
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_multiplier = threshold_multiplier
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Initialize model, loss function, optimizer
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = None
        
    def _prepare_sequences(self, data):
        """
        Prepare sequences for the model.
        
        Parameters:
        -----------
        data: np.ndarray
            Input data
            
        Returns:
        --------
        np.ndarray
            Array of sequences
        """
        sequences = []
        for i in range(len(data) - self.seq_length + 1):
            sequences.append(data[i:i+self.seq_length])
        return np.array(sequences)
    
    def _prepare_data_loader(self, sequences, shuffle=True):
        """
        Prepare data loader for training or evaluation.
        
        Parameters:
        -----------
        sequences: np.ndarray
            Array of sequences
        shuffle: bool
            Whether to shuffle the data
            
        Returns:
        --------
        torch.utils.data.DataLoader
            Data loader
        """
        tensor_x = torch.FloatTensor(sequences)
        dataset = TensorDataset(tensor_x, tensor_x)  # Autoencoder: input = target
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
    
    def fit(self, series):
        """
        Fit model to time series.
        
        Parameters:
        -----------
        series: pd.Series or np.ndarray
            Time series data
        """
        # Convert to numpy array if needed
        if isinstance(series, pd.Series):
            data = series.values.reshape(-1, 1)
        else:
            data = series.reshape(-1, 1)
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Prepare sequences
        sequences = self._prepare_sequences(data_scaled)
        
        # Create data loader
        train_loader = self._prepare_data_loader(sequences)
        
        # Initialize model
        input_dim = data_scaled.shape[1]
        if self.model_type == 'lstm_ae':
            self.model = LSTMAutoencoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim
            ).to(self.device)
        elif self.model_type == 'tcn':
            self.model = TCN(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=input_dim
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Train model
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.8f}')
        
        # Calculate reconstruction errors to set threshold
        self.model.eval()
        all_errors = []
        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                errors = torch.mean(torch.pow(outputs - batch_x, 2), dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())
        
        # Set threshold as mean + k*std of reconstruction errors
        error_mean = np.mean(all_errors)
        error_std = np.std(all_errors)
        self.threshold = error_mean + self.threshold_multiplier * error_std
        
    def detect_anomalies(self, series, return_scores=False):
        """
        Detect anomalies in time series.
        
        Parameters:
        -----------
        series: pd.Series or np.ndarray
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
        
        # Convert to numpy array if needed
        if isinstance(series, pd.Series):
            data = series.values.reshape(-1, 1)
            index = series.index
        else:
            data = series.reshape(-1, 1)
            index = None
        
        # Scale data
        data_scaled = self.scaler.transform(data)
        
        # Prepare sequences
        sequences = self._prepare_sequences(data_scaled)
        
        # Create data loader
        data_loader = self._prepare_data_loader(sequences, shuffle=False)
        
        # Predict
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                errors = torch.mean(torch.pow(outputs - batch_x, 2), dim=(1, 2))
                reconstruction_errors.extend(errors.cpu().numpy())
        
        # Pad beginning with zeros
        padded_errors = np.zeros(len(series))
        padded_errors[self.seq_length-1:] = reconstruction_errors
        
        # Create anomaly flags
        anomalies = padded_errors > self.threshold
        
        # Convert to Series if index provided
        if index is not None:
            anomalies = pd.Series(anomalies, index=index)
            if return_scores:
                scores = pd.Series(padded_errors, index=index)
                return anomalies, scores
        elif return_scores:
            return anomalies, padded_errors
        
        return anomalies 