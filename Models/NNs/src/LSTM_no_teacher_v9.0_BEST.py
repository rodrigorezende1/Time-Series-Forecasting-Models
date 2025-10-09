import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Full implementation of the seq to seq 
#Plotting the error in each time step
# Step 1: Generate a full decaying sinusoidal signal
def generate_full_decaying_sinusoidal_signal(total_length):
    x = np.linspace(0, 40, total_length)
    decay1 = np.exp(-0.05 * x)  # Exponential decay
    decay2 = np.exp(-0.07 * x)
    full_signal = decay1 * np.sin(1.5*x)+ decay2 * np.sin(1.7*x)  # Sinusoidal pattern with decay
    return full_signal

# Step 2: Extract sub-inputs for training
def extract_inputs(full_signal, seq_length):
    #max_start_index = len(full_signal) - seq_length
    inputs = []
    targets = []
    for start_idx in range(len(full_signal)-seq_length):
        # = np.random.randint(0, max_start_index)
        input = full_signal[start_idx:start_idx + seq_length]
        target = full_signal[start_idx+1:start_idx + seq_length+1]
        inputs.append(input)
        targets.append(target)
    return np.array(inputs), np.array(targets)

# LSTM Model Definition
torch.set_default_dtype(torch.float64)
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(self.dropout(out[:, -1, :]))  # Only the last output is used
        #out = self.fc(out[:, -1, :])
        return out, hidden

# Hyperparameters
total_length = 75  # Total length of the full signal
seq_length = total_length-1#(sequential training)#40      # Length of each training sequence
hidden_size = 150#100   # LSTM hidden state size
batch_size = 1 #no batch training

# Generate data
full_signal = generate_full_decaying_sinusoidal_signal(total_length)

inputs,targets = extract_inputs(full_signal, seq_length)

# Prepare data for PyTorch
inputs = torch.tensor(inputs, dtype=torch.float64).unsqueeze(-1)  # Shape (num_inputs, seq_length, 1)
targets = torch.tensor(targets, dtype=torch.float64).unsqueeze(-1)  # Shape (num_inputs, seq_length, 1)
dataset = torch.utils.data.TensorDataset(inputs, targets)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=1)

num_epochs = 5000     # Number of training epochs
learning_rate = 0.0001  # Learning rate
loss_fn = nn.MSELoss()
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
#optimizer = optim.Adamax(model.parameters(), lr=learning_rate,weight_decay=1e-5)

# Training loop without teacher forcing
for epoch in range(num_epochs):
    total_loss = 0
    #hidden=None
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Initialize hidden state
        
        batch_size = inputs.size(0)
        hidden = None
        #Clear gradients
        optimizer.zero_grad()
        
        # Start with the first input timestep
        input_t = inputs[:, 0, :].unsqueeze(1)  # Shape (batch_size, 1, input_size)
        predictions = torch.zeros(batch_size, seq_length, 1)  # Placeholder for storing outputs
        #break
        # Recursive prediction loop across sequence length
        for t in range(seq_length):
            output, hidden = model(input_t, hidden)  # Forward pass with hidden state
            predictions[:, t, :] = output  # Store output without adding an extra dimension
            input_t = output[:,:,None]  # Shape (batch_size, 1, output_size)

        # Compute loss
        loss = loss_fn(predictions, targets)
        loss.backward()  # Backpropagation
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # Update model parameters
        
        # Accumulate loss
        total_loss += loss.item()
        
    #break
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.12f}')

inputs,targets = extract_inputs(full_signal, seq_length)
inputs = torch.tensor(inputs, dtype=torch.float64).unsqueeze(-1)  # Shape (num_inputs, seq_length, 1)
targets = torch.tensor(targets, dtype=torch.float64).unsqueeze(-1)  # Shape (num_inputs, seq_length, 1)

# model.eval()
# with torch.no_grad():
#     # Start with the first timestep of the original signal
#     test_input = inputs[0, :, :,None]  # Start with first sequence's first timestep
#     hidden = None
#     predictions = []

#     # Predict within the sequence length
#     for t in range(len(full_signal)-seq_length):
#         output, hidden = model(test_input, hidden)
#         predictions.append(output[-1].item())
#         test_input = output[:,:,None]# inputs[0:1, t+1:t+2, :]  # Use actual data up to sequence length

#     # Extrapolate beyond the sequence length
#     for _ in range(extrapolate_steps):
#         output, hidden = model(test_input, hidden)
#         predictions.append(output[-1].item())
#         test_input = output[:,:,None]#.unsqueeze(0).unsqueeze(1)  # Use model's output as input

# # Plotting
# true_signal = full_signal#[:seq_length + extrapolate_steps]
# plt.plot(range(len(full_signal)), full_signal, label='True Signal',marker='x')
# plt.plot(range(seq_length,len(full_signal) + extrapolate_steps), predictions, label='LSTM Prediction (Extrapolated)',marker='x')
# plt.axvline(x=total_length, color='r', linestyle='--', label="Start of Extrapolation")
# plt.legend()
# plt.xlabel('Timestep')
# plt.ylabel('Amplitude')
# plt.title('Decaying Sinusoidal Signal with Extrapolation')
# plt.show()


# ##
# #### Ploting the results if we give just the first point
# # Recursive Forecasting and Extrapolation
n=10
extrapolate_steps = n*(total_length)  # Number of steps to extrapolate beyond training data
def generate_decaying_sinusoidal_signal2(total_length,n):
    x= np.linspace(0, (40*(total_length*(n+1)-1))/(total_length-1), total_length*(n+1))
    decay1 = np.exp(-0.05 * x)  # Exponential decay
    decay2 = np.exp(-0.07 * x)
    full_signal = decay1 * np.sin(1.5*x)+ decay2 * np.sin(1.7*x)  # Sinusoidal pattern with decay
    return full_signal

signal_extrapol = generate_decaying_sinusoidal_signal2(total_length,n)

model.eval()
with torch.no_grad():
    # Start with the first timestep of the original signal
    test_input = inputs[0, 0, :,None,None]  # Start with first sequence's first timestep
    hidden = None
    predictions = []

    # Predict within the sequence length
    for t in range(len(full_signal)):
        output, hidden = model(test_input, hidden)
        predictions.append(output[-1].item())
        test_input = output[:,:,None]# inputs[0:1, t+1:t+2, :]  # Use actual data up to sequence length

    # Extrapolate beyond the sequence length
    for _ in range(extrapolate_steps):
        output, hidden = model(test_input, hidden)
        predictions.append(output[-1].item())
        test_input = output[:,:,None]#.unsqueeze(0).unsqueeze(1)  # Use model's output as input

# Plotting
error_per_ts = np.sqrt((signal_extrapol[1:]-predictions[:-1])**2)#[:seq_length + extrapolate_steps]
plt.plot(range(0,len(signal_extrapol)),signal_extrapol, label='True Signal',marker='x')
plt.plot(range(1,total_length*(n+1)+1), predictions, label='LSTM Prediction (Extrapolated)',marker='x')
plt.plot(range(1,total_length*(n+1)), error_per_ts, label='Predictions errors',marker='x')
plt.axvline(x=total_length, color='r', linestyle='--', label="Start of Extrapolation")
plt.legend()
plt.xlabel('Timestep')
plt.ylabel('Amplitude')
plt.title('Decaying Sinusoidal Signal with Extrapolation')
plt.show()



