# -*- coding: utf-8 -*-
"""
This script trains a Transformer model to predict and extrapolate a time-series signal.
The process includes:
1. Loading and preprocessing a signal from a text file.
2. Creating sequential data for training the Transformer.
3. Defining and training the Transformer model.
4. Extrapolating the signal beyond the training data.
5. Plotting the results for analysis.
"""
import math
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

# -- Transformer Model Hyperparameters --
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT_PROB = 0.1

# -- Training Hyperparameters --
NUM_EPOCHS = 1000
LEARNING_RATE = 0.005
BATCH_SIZE = 1


# =============================================================================
# 2. Model Definition
# =============================================================================
class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """A Transformer model for time-series prediction."""
    def __init__(self, input_size=1, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, input_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(D_MODEL)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

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
        input_seq = full_signal[i: i + seq_length]
        # For Transformer, we might predict the final point
        target = full_signal[i + seq_length]
        inputs.append(input_seq)
        targets.append(target)
    return np.array(inputs), np.array(targets)

# =============================================================================
# 4. Training and Extrapolation Functions
# =============================================================================
def train_model(model, train_loader, loss_fn, optimizer, num_epochs, device):
    print("\nStarting Transformer model training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.12f}')

def extrapolate_signal(model, start_sequence, num_steps, device):
    print("\nExtrapolating signal with Transformer...")
    model.eval()
    predictions = []
    
    current_sequence = torch.tensor(start_sequence, device=device).unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        for _ in range(num_steps):
            output = model(current_sequence)
            predicted_value = output.cpu().item()
            predictions.append(predicted_value)
            
            # Append the prediction and slide the window for the next step
            next_step_input = output.unsqueeze(-1)
            current_sequence = torch.cat((current_sequence[:, 1:, :], next_step_input), dim=1)
            
    return np.array(predictions)

# =============================================================================
# 5. Plotting Function
# =============================================================================
def plot_results(true_signal, predictions, train_end_idx, seq_length):
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(true_signal, 'o-', label='True Signal', markersize=4)
    prediction_range = range(train_end_idx, train_end_idx + len(predictions))
    ax.plot(prediction_range, predictions, 'x-', label='Transformer Prediction')
    ax.axvline(x=train_end_idx, color='r', linestyle='--', label='Start of Extrapolation')
    ax.set_title('Transformer Time-Series Extrapolation', fontsize=16)
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
    
    # Transformer is often trained on shorter, fixed-length sequences
    seq_length = 50 
    
    signal_training = signal[:train_end_index]
    inputs, targets = create_sequences(signal_training, seq_length)
    
    inputs_tensor = torch.tensor(inputs).unsqueeze(-1)
    targets_tensor = torch.tensor(targets)
    train_dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = TransformerModel(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT_PROB
    ).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)
    
    train_model(model, train_loader, loss_fn, optimizer, NUM_EPOCHS, device)

    start_sequence = signal[train_end_index - seq_length : train_end_index]
    extrapolate_steps = len(signal) - train_end_index
    extrapolated_values = extrapolate_signal(model, start_sequence, extrapolate_steps, device)
    
    plot_results(signal, extrapolated_values, train_end_index, seq_length)