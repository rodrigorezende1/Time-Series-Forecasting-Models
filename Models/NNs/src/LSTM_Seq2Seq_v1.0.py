import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 10, 1000)
y = np.sin(x**2)

# plt.plot(x,y)
# plt.show()
# Define sequence length
input_seq_length = 5
output_seq_length = 3

# Create sequences
def create_sequences(data, input_seq_length, output_seq_length):
    sequences = []
    for i in range(len(data) - input_seq_length - output_seq_length):
        input_seq = data[i:i+input_seq_length]
        output_seq = data[i+input_seq_length:i+input_seq_length+output_seq_length]
        sequences.append((input_seq, output_seq))
    return sequences

from sklearn.preprocessing import MinMaxScaler
# Normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

sequences = create_sequences(y, input_seq_length, output_seq_length)

class SineDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, output_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

dataset = SineDataset(sequences)
dataloader = DataLoader(dataset, batch_size=50, shuffle=True)


import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.9):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = trg.shape[2]

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)

        #input = trg[:, 0, None]
        input = src[:,-1,None]

        for t in range(1, trg_len):
            output, hidden, cell = decoder(input, hidden, cell)
            outputs[:, t, :] = output[:,0]
            teacher_force = np.random.random() < teacher_forcing_ratio
            input = trg[:, t, None] if teacher_force else output
        
        return outputs


import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 1
hidden_size = 50
output_size = 1
num_layers = 1

encoder = Encoder(input_size, hidden_size, num_layers).to(device)
decoder = Decoder(input_size, hidden_size, output_size, num_layers).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adamax(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    for input_seq, output_seq in dataloader:
        input_seq = input_seq.unsqueeze(-1).to(device)  # Add feature dimension and move to device
        output_seq = output_seq.unsqueeze(-1).to(device)
        
        optimizer.zero_grad()
        
        output = model(input_seq, output_seq)
        
        loss = criterion(output, output_seq)
        
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


#Eval
import matplotlib.pyplot as plt

# Prepare data for prediction
test_x = np.linspace(10, 20, 1000)
test_y = np.sin(test_x**2)

# Create test sequences
test_sequences = create_sequences(test_y, input_seq_length, output_seq_length)

# Predict using the model
model.eval()
predicted = []
with torch.no_grad():
    for input_seq, _ in test_sequences:
        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        output_seq = torch.zeros((1, output_seq_length, 1)).to(device)
        
        # Generate predictions using the model
        pred = model(input_seq, output_seq, teacher_forcing_ratio=0.9).squeeze().cpu().numpy()
        
        # Collect the predictions
        predicted.extend(pred)

# Adjust the length of the predicted list to match the length of test_x[input_seq_length:]
predicted = predicted[:len(test_x) - input_seq_length]

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(test_x[input_seq_length:], predicted,"x" ,label='Predicted')
plt.plot(test_x, test_y, label='Actual')
plt.xlabel('x')
plt.ylabel('sin(x^2)')
plt.title('Prediction of sin(x^2) using Seq2Seq LSTM')
plt.legend()
plt.show()
