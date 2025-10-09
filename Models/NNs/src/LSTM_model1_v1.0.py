# -*- coding: utf-8 -*-
"""
This script trains an LSTM model to predict and extrapolate a time-series signal.
The process includes:
1. Loading and preprocessing a signal from a text file.
2. Creating sequential data for training the LSTM.
3. Defining and training the LSTM model without teacher forcing.
4. Extrapolating the signal beyond the training data.
5. Plotting the results for analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# 1. Configuration & Hyperparameters
# =============================================================================
# -- Data & Preprocessing --
FILE_PATH = "path/to/your/o11_model1.txt" # <--- CHANGE THIS TO YOUR FILE PATH
TRAIN_VAL_PERCENTAGE = 0.3
INITIAL_SAMPLE_OFFSET = 5

# -- Downsampling Parameters --
MAX_FREQ_HZ = 180e9
SAFETY_FACTOR = 2.2

# -- LSTM Model Hyperparameters --
HIDDEN_SIZE = 250
NUM_LAYERS = 1
DROPOUT_PROB = 0.4

# -- Training Hyperparameters --
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01
BATCH_SIZE = 1


# =============================================================================
# 2. Model Definition
# =============================================================================
class LSTMModel(nn.Module):
    """A simple LSTM model for time-series prediction."""
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1, dropout_prob=0.4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(self.dropout(out[:, -1, :]))
        return out, hidden

# =============================================================================
# 3. Data Handling Functions
# =============================================================================
# (These functions are copied here for completeness)
def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    raw_data = np.loadtxt(file_path).T
    time_steps_ns, signal_data = raw_data[0], raw_data[1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(signal_data.reshape(-1, 1)).flatten()
    nyquist_freq = MAX_FREQ_HZ * SAFETY_FACTOR
    max_time_step = 1 / nyquist_freq
    simulation_time_step = np.mean(np.diff(time_steps_ns)) * 1e-9
    downsampling_factor = max(1, int(max_time_step / simulation_time_step))
    print(f"Calculated downsampling factor: {downsampling_factor}")
    signal_downsampled = data_scaled[::downsampling_factor]
    return signal_downsampled, scaler

def create_sequences(full_signal, seq_length):
    inputs, targets = [], []
    for i in range(len(full_signal) - seq_length):
        input_seq = full_signal[i : i + seq_length]
        target_seq = full_signal[i + 1 : i + seq_length + 1]
        inputs.append(input_seq)
        targets.append(target_seq)
    return np.array(inputs), np.array(targets)

# =============================================================================
# 4. Training and Extrapolation Functions
# =============================================================================
def train_model(model, train_loader, loss_fn, optimizer, num_epochs, device):
    print("\nStarting LSTM model training...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        hidden_state = None

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            seq_length = inputs.shape[1]

            if hidden_state:
                h, c = hidden_state
                hidden_state = (h.detach(), c.detach())

            optimizer.zero_grad()
            current_input = inputs[:, 0, :].unsqueeze(1)
            predictions = []

            for t in range(seq_length):
                output, hidden_state = model(current_input, hidden_state)
                predictions.append(output)
                current_input = output.unsqueeze(1)

            predictions_tensor = torch.cat(predictions, dim=1)
            loss = loss_fn(predictions_tensor, targets.squeeze(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.12f}')

def extrapolate_signal(model, start_input, num_steps, device):
    print("\nExtrapolating signal with LSTM...")
    model.eval()
    predictions = []
    hidden = None
    current_input = start_input.to(device)
    
    with torch.no_grad():
        for _ in range(num_steps):
            output, hidden = model(current_input, hidden)
            predicted_value = output.cpu().item()
            predictions.append(predicted_value)
            current_input = output.unsqueeze(1)
            
    return np.array(predictions)

# =============================================================================
# 5. Plotting Function
# =============================================================================
def plot_results(true_signal, predictions, train_end_idx):
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(true_signal, 'o-', label='True Signal', markersize=4)
    prediction_range = range(INITIAL_SAMPLE_OFFSET + 1, INITIAL_SAMPLE_OFFSET + 1 + len(predictions))
    ax.plot(prediction_range, predictions, 'x-', label='LSTM Prediction')
    ax.axvline(x=train_end_idx, color='r', linestyle='--', label='End of Training Data')
    ax.set_title('LSTM Time-Series Extrapolation', fontsize=16)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Scaled Amplitude', fontsize=12)
    ax.legend()
    plt.show()

# =============================================================================
# 6. Main Execution
# =============================================================================
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    signal, _ = load_and_preprocess_data(FILE_PATH)
    
    train_end_index = int(len(signal) * TRAIN_VAL_PERCENTAGE)
    signal_training = signal[INITIAL_SAMPLE_OFFSET : train_end_index]
    
    seq_length = len(signal_training) - 1
    inputs, targets = create_sequences(signal_training, seq_length)
    
    inputs_tensor = torch.tensor(inputs).unsqueeze(-1)
    targets_tensor = torch.tensor(targets).unsqueeze(-1)
    train_dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LSTMModel(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT_PROB
    ).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
    
    train_model(model, train_loader, loss_fn, optimizer, NUM_EPOCHS, device)

    start_input = torch.tensor([[signal_training[0]]])
    extrapolated_values = extrapolate_signal(model, start_input, len(signal), device)
    
    plot_results(signal, extrapolated_values, train_end_index)