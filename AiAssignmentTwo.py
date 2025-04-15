import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

WINDOW_SIZE = 30
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_TEST_SPLIT = 0.8
FUTURE_PREDICTION_DAYS = 180  

class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for time series data."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """Standard LSTM model for time series forecasting."""
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_RATE):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM model for time series forecasting."""
    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT_RATE):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def load_and_preprocess_data(filepath):
    """Load and preprocess time series data."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.dropna(inplace=True)
    df.set_index('date', inplace=True)
    
    data = df['temp'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_sequences(data, window_size):
    """Create input-output pairs for time series prediction."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def prepare_data_loaders(X, y, train_split=TRAIN_TEST_SPLIT, batch_size=BATCH_SIZE):
    """Split data and create PyTorch dataloaders."""
    train_size = int(len(X) * train_split)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def train_model(model, train_loader, optimizer, criterion, num_epochs=NUM_EPOCHS, device="cpu"):
    """Train the model."""
    model.train()
    history = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            
        avg_loss = total_loss / len(train_loader.dataset)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return history

def evaluate_model(model, X_test_tensor, y_test_tensor, scaler, device="cpu"):
    """Evaluate the model on test data."""
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        
        predictions = model(X_test_tensor)
        predictions = predictions.cpu().numpy()
        
        y_true = y_test_tensor.numpy()
        predictions_inv = scaler.inverse_transform(predictions)
        y_true_inv = scaler.inverse_transform(y_true)
        
        mse = mean_squared_error(y_true_inv, predictions_inv)
        r2 = r2_score(y_true_inv, predictions_inv)
        
    return predictions_inv, y_true_inv, mse, r2

def predict_future(model, scaled_data, window_size, future_steps, scaler, device="cpu"):
    """Predict future values based on the trained model."""
    model.eval()
    last_window = scaled_data[-window_size:].copy()
    future_preds = []
    
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(last_window).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(input_tensor)
        
        pred_val = pred.cpu().numpy()[0, 0]
        future_preds.append(pred_val)
        
        last_window = np.append(last_window[1:], [[pred_val]], axis=0)
    
    future_preds = np.array(future_preds).reshape(-1, 1)
    future_preds_inv = scaler.inverse_transform(future_preds)
    
    return future_preds_inv

def plot_test_predictions(true_values, predictions, model_name):
    """Plot test set predictions against true values."""
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Temperature')
    plt.plot(predictions, label=f'{model_name} Predicted Temperature')
    plt.title(f"Test Set Temperature Prediction - {model_name}")
    plt.xlabel("Sample")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_future_predictions(predictions, days=FUTURE_PREDICTION_DAYS):
    """Plot future temperature predictions."""
    plt.figure(figsize=(12, 6))
    plt.plot(predictions, label=f'Predicted Temperature for Future {days} Days')
    plt.title(f"Future {days} Days Temperature Prediction")
    plt.xlabel("Day")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    file_paths = [
        "/mnt/data/psspredict10.csv",  
        "C:\\Users\\aboud\\Downloads\\psspredict10.csv"  
    ]
    
    filepath = None
    for path in file_paths:
        if os.path.exists(path):
            filepath = path
            print(f"Using file: {filepath}")
            break
    
    if filepath is None:
        raise FileNotFoundError("Could not find the temperature data file in any of the specified paths.")
    
    scaled_data, scaler = load_and_preprocess_data(filepath)
    
    X, y = create_sequences(scaled_data, WINDOW_SIZE)
    
    train_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data_loaders(X, y)
    
    model_lstm = LSTMModel().to(device)
    model_bilstm = BiLSTMModel().to(device)
    
    criterion = nn.MSELoss()
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)
    optimizer_bilstm = torch.optim.Adam(model_bilstm.parameters(), lr=LEARNING_RATE)
    
    print("\nTraining LSTM model...")
    lstm_history = train_model(model_lstm, train_loader, optimizer_lstm, criterion, device=device)
    
    print("\nTraining BiLSTM model...")
    bilstm_history = train_model(model_bilstm, train_loader, optimizer_bilstm, criterion, device=device)
    
    print("\nEvaluating models on test data...")
    pred_lstm, y_true_inv, mse_lstm, r2_lstm = evaluate_model(model_lstm, X_test_tensor, y_test_tensor, scaler, device)
    print(f"LSTM Test MSE: {mse_lstm:.4f}, R²: {r2_lstm:.4f}")
    
    pred_bilstm, y_true_inv, mse_bilstm, r2_bilstm = evaluate_model(model_bilstm, X_test_tensor, y_test_tensor, scaler, device)
    print(f"BiLSTM Test MSE: {mse_bilstm:.4f}, R²: {r2_bilstm:.4f}")
    
    plot_test_predictions(y_true_inv, pred_lstm, "LSTM")
    plot_test_predictions(y_true_inv, pred_bilstm, "BiLSTM")
    
    print("\nPredicting future temperatures...")
    best_model = model_bilstm if mse_bilstm < mse_lstm else model_lstm
    best_model_name = "BiLSTM" if mse_bilstm < mse_lstm else "LSTM"
    print(f"Using {best_model_name} model for future predictions")
    
    future_predictions = predict_future(best_model, scaled_data, WINDOW_SIZE, FUTURE_PREDICTION_DAYS, scaler, device)
    plot_future_predictions(future_predictions)
    
    return best_model, future_predictions

if __name__ == "__main__":
    main()