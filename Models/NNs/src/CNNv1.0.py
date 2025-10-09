import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
from torch.autograd import Variable
from torch import nn
from sklearn import preprocessing
import math
from scipy.fft import fft, fftfreq,fftshift
warnings.filterwarnings('ignore')
import time
from numpy import array
#matplotlib inline

# Supplemetary packages
import gc

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset,DataLoader


#RNN usage LSTM
#data = np.loadtxt("RNN_Antenne_lossy.txt") ## import data

data = np.loadtxt("5G_cellphone_o2_1.txt")

#Data
downsampling_factor = 10
train_percentage = 0.4
initial_sample = 500

# #LSTM
seq_length = 6
# input_size = 1
# hidden_size = 10
# num_layers = 1
# output_size = 1

#Training Loop
# training_loop = 5
# num_epochs_init = 10000

#Initializing
data=data.T
tt=data[0]
tt = tt[0:-1:downsampling_factor]
dt = tt[1]-tt[0]  
yy= data[1]
yy = yy[initial_sample:-1:downsampling_factor]
# data = preprocessing.normalize([yy]) ## normalize the data
# data = data[0]
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(yy.reshape(-1, 1))[:,0]

# data = np.diff(data, axis=0)

#Making regressive data
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Preprocess the historical data
X, y = create_sequences(data, seq_length)

# Split the data into training and testing sets
train_size = int(len(y)*train_percentage)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()


#Unifying input and target class
class ElecDataset(Dataset):
    def __init__(self,feature,target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self,idx):
        item = self.feature[idx]
        label = self.target[idx]
        
        return item,label
    

# Model definition
class CNN_ForecastNet(nn.Module):
    def __init__(self):
        super(CNN_ForecastNet,self).__init__()
        self.conv1d = nn.Conv1d(seq_length,256,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(256*batch,50)#50 best value
        self.fc2 = nn.Linear(50,1)
        
    def forward(self,x):
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.view(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
# class CNN_ForecastNet(nn.Module):
#     def __init__(self):
#         super(CNN_ForecastNet,self).__init__()
#         self.conv1d = nn.Conv1d(seq_length,128,kernel_size=1)
#         self.conv2d = nn.Conv1d(128,128,kernel_size=1)
#         self.conv3d = nn.Conv1d(128,128,kernel_size=1)
#         self.conv4d = nn.Conv1d(128,128,kernel_size=1)
#         self.conv5d = nn.Conv1d(128,128,kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc1 = nn.Linear(128*batch,100)
#         self.fc2 = nn.Linear(100,1)
        
#     def forward(self,x):
#         x = self.conv1d(x)
#         x = self.conv2d(x)
#         x = self.conv3d(x)
#         x = self.conv4d(x)
#         x = self.conv5d(x)
#         x = self.relu(x)
#         x = x.view(-1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
        
#         return x
    

batch = 1
device = torch.device("cpu")
model = CNN_ForecastNet().to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()


train = ElecDataset(X_train.reshape(X_train.shape[0],X_train.shape[1],1),y_train)
valid = ElecDataset(X_test.reshape(X_test.shape[0],X_test.shape[1],1),y_test)
train_loader = torch.utils.data.DataLoader(train,batch_size=batch,shuffle=False)
valid_loader = torch.utils.data.DataLoader(train,batch_size=batch,shuffle=False)

train_losses = []
valid_losses = []

def Train():
    
    running_loss = .0
    
    model.train()

    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss
        
        train_loss = running_loss/len(train_loader)
        train_losses.append(train_loss.detach().numpy())
  
        print(f'train_loss {train_loss}')

def Valid():
    running_loss = .0
    
    model.eval()
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds,labels)
            running_loss += loss
            
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')


start = time.time()
print("starting time measurement")
epochs = 1000
for epoch in range(epochs):
    print('epochs {}/{}'.format(epoch+1,epochs))
    Train()
    Valid()
    gc.collect()

end = time.time()
print("Total time"+ " "+ str(end - start))


plt.plot(train_losses,label='train_loss')
plt.plot(valid_losses,label='valid_loss')
plt.title('MSE Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


#prediciting the fitting 
target_x , target_y = split_sequence(y[:train_size],seq_length)
inputs = target_x.reshape(target_x.shape[0],target_x.shape[1],1)

model.eval()
prediction = []
batch_size = batch
iterations =  int(inputs.shape[0]/batch)

for i in range(iterations):
    preds = model(torch.tensor(inputs[batch_size*i:batch_size*(i+1)]).float())
    prediction.append(preds.detach().numpy())


#prediciting the extrapolation 
target_x , target_y = split_sequence(y[:train_size],seq_length)
inputs = target_x.reshape(target_x.shape[0],target_x.shape[1],1)

new_x_batch = target_x[-batch_size+1:]
new_x = new_x_batch[-1,1:]
new_x = np.concatenate((new_x[:,],target_y[-1]),axis=None)
new_x_batch = np.concatenate((new_x_batch, new_x[None,:]),axis=0)
model.eval()
prediction_ext = []
batch_size = batch
iterations =  int((data.size-train_size)/batch)#this 2 is for the batch size
for i in range(iterations):
    preds = model(torch.tensor(new_x_batch[i,:][None,:,None]).float())
    prediction_ext.append(preds.detach().numpy())
    new_x_batch = new_x_batch[-batch_size+1:]
    new_x = new_x_batch[-1,1:]
    new_x = np.concatenate((new_x[:,],preds.detach().numpy()),axis=None)
    new_x_batch = np.concatenate((new_x_batch, new_x[None,:]),axis=0)


len(prediction_ext)
len(prediction)

#Plotting
test_start_index = (len(data) - len(y_test) - seq_length)/batch

plt.plot(target_y[:-1:batch],  label="True Values") #why the two here?
plt.plot(prediction, 'x', label="Predictions")
plt.axvline(x=test_start_index, color='gray', linestyle='--', label="Test set start")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("CNN Predictions vs True Values")
plt.show()


plt.plot(y[train_size+1::batch],  label="True Values") #why the two here?
plt.plot(prediction_ext, 'x', label="Predictions")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("CNN Predictions vs True Values")
plt.show()


plt.plot(target_y[:-1:batch],  label="True Values") #why the two here?
plt.plot(prediction, 'x', label="Predictions")

plt.plot(y[train_size+1::batch],  label="True Values") #why the two here?
plt.plot(prediction_ext, 'x', label="Predictions")

plt.plot(np.concatenate((target_y[:-1:batch],y[train_size+1::batch])),  label="True Values") #why the two here?
plt.plot(prediction+prediction_ext, 'x', label="Extrapolation CNN")
plt.plot(prediction, 'x', label="Training CNN")
plt.axvline(x=len(prediction), color='red', linestyle='--', label="Test set start")
plt.xlabel("Time steps(k)",fontsize=20)
plt.ylabel("Amplitude [a.u.]",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.grid()
plt.legend(fontsize=20)
plt.title("CNN Predictions",fontsize=20)
plt.show()