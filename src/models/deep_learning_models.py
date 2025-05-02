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
    
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        
        
        _, (hidden, _) = self.encoder(x)
        
        
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        
        
        decoder_output, _ = self.decoder(hidden_repeated)
        
        
        output = self.output_layer(decoder_output)
        
        return output


class TCN(nn.Module):
    
    
    class CausalConv1d(nn.Module):
        
        
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super(TCN.CausalConv1d, self).__init__()
            
            
            self.padding = (kernel_size - 1) * dilation
            
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                dilation=dilation
            )
            
        def forward(self, x):
            
            x = self.conv(x)
            if self.padding != 0:
                return x[:, :, :-self.padding]
            return x
    
    class ResidualBlock(nn.Module):
        
        
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
            
            residual = x
            
            out = self.relu(self.causal_conv1(x))
            out = self.dropout(out)
            out = self.relu(self.causal_conv2(out))
            out = self.dropout(out)
            
            return out + residual
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, kernel_size=3):
        
        super(TCN, self).__init__()
        
        
        self.input_layer = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )
        
        
        self.residual_blocks = nn.ModuleList([
            self.ResidualBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2**i
            )
            for i in range(num_layers)
        ])
        
        
        self.output_layer = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=output_dim,
            kernel_size=1
        )
        
    def forward(self, x):
        
        
        x = x.transpose(1, 2)
        
        
        x = self.input_layer(x)
        
        
        for block in self.residual_blocks:
            x = block(x)
        
        
        x = self.output_layer(x)
        
        
        x = x.transpose(1, 2)
        
        return x


class DeepAnomalyDetector:
    
    
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
        
        self.model_type = model_type
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_multiplier = threshold_multiplier
        
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.threshold = None
        
    def _prepare_sequences(self, data):
        
        sequences = []
        for i in range(len(data) - self.seq_length + 1):
            sequences.append(data[i:i+self.seq_length])
        return np.array(sequences)
    
    def _prepare_data_loader(self, sequences, shuffle=True):
        
        tensor_x = torch.FloatTensor(sequences)
        dataset = TensorDataset(tensor_x, tensor_x)  
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
    
    def fit(self, series):
        
        
        if isinstance(series, pd.Series):
            data = series.values.reshape(-1, 1)
        else:
            data = series.reshape(-1, 1)
        
        
        data_scaled = self.scaler.fit_transform(data)
        
        
        sequences = self._prepare_sequences(data_scaled)
        
        
        train_loader = self._prepare_data_loader(sequences)
        
        
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
        
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_x)
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.8f}')
        
        
        self.model.eval()
        all_errors = []
        with torch.no_grad():
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                errors = torch.mean(torch.pow(outputs - batch_x, 2), dim=(1, 2))
                all_errors.extend(errors.cpu().numpy())
        
        
        error_mean = np.mean(all_errors)
        error_std = np.std(all_errors)
        self.threshold = error_mean + self.threshold_multiplier * error_std
        
    def detect_anomalies(self, series, return_scores=False):
        
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        
        if isinstance(series, pd.Series):
            data = series.values.reshape(-1, 1)
            index = series.index
        else:
            data = series.reshape(-1, 1)
            index = None
        
        
        data_scaled = self.scaler.transform(data)
        
        
        sequences = self._prepare_sequences(data_scaled)
        
        
        data_loader = self._prepare_data_loader(sequences, shuffle=False)
        
        
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                errors = torch.mean(torch.pow(outputs - batch_x, 2), dim=(1, 2))
                reconstruction_errors.extend(errors.cpu().numpy())
        
        
        padded_errors = np.zeros(len(series))
        padded_errors[self.seq_length-1:] = reconstruction_errors
        
        
        anomalies = padded_errors > self.threshold
        
        
        if index is not None:
            anomalies = pd.Series(anomalies, index=index)
            if return_scores:
                scores = pd.Series(padded_errors, index=index)
                return anomalies, scores
        elif return_scores:
            return anomalies, padded_errors
        
        return anomalies 