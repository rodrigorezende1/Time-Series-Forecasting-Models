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

#LSTM
seq_length = 100
input_size = 1
hidden_size = 10
num_layers = 1
output_size = 1

#Training Loop
training_loop = 5
num_epochs_init = 15000

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

# data = np.sin(10*np.pi*tt)
# plt.plot(data)
# plt.show()
#Making regressive data

# Split the data into training and testing sets
train_size = int(len(yy)*train_percentage)

seq_length = 10
def create_dataset(dataset, look_back=seq_length):  ## create the seq_length signal
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

x, y = create_dataset(data)  ## seq_length dataset

x_model = x[:train_size,:]
y_model = y[:train_size]

#I should not use x and y anymore

from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(x_model,y_model,test_size=0.2,random_state=42)
#X_train,X_test,y_train,y_test=train_test_split(x_model,y_model,test_size=0.01)
X_train = x_model[:]
y_train = y_model[:]

print(type(X_train))
X_train = torch.tensor(X_train, dtype=torch.float32)
#X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)

# since data is ready we can develop the model:
# class linearRegression(nn.Module): # all the dependencies from torch will be given to this class [parent class] # nn.Module contains all the building block of neural networks:
#   def __init__(self,input_dim):
#     super(linearRegression,self).__init__()  # building connection with parent and child classes
#     self.fc1=nn.Linear(input_dim,10)          # hidden layer 1
#     self.fc2=nn.Linear(10,5)                  # hidden layer 2
#     self.fc3=nn.Linear(5,3)                   # hidden layer 3
#     self.fc4=nn.Linear(3,1)                   # last layer

#   def forward(self,d):
#     out=torch.relu(self.fc1(d))              # seq_length * weights + bias for layer 1
#     out=torch.relu(self.fc2(out))            # seq_length * weights + bias for layer 2
#     out=torch.relu(self.fc3(out))            # seq_length * weights + bias for layer 3
#     out=self.fc4(out)                        # seq_length * weights + bias for last layer
#     return out                               # final outcome

class linearRegression(nn.Module): # all the dependencies from torch will be given to this class [parent class] # nn.Module contains all the building block of neural networks:
  def __init__(self,input_dim):
    super(linearRegression,self).__init__()  # building connection with parent and child classes
    self.fc1=nn.Linear(input_dim,20)          # hidden layer 1
    self.fc2=nn.Linear(20,20)                  # hidden layer 2
    self.fc3=nn.Linear(20,20)                   # hidden layer 3
    self.fc4=nn.Linear(20,20)                   # hidden layer 4
    self.fc5=nn.Linear(20,20)                   # hidden layer 5
    self.fc6=nn.Linear(20,20)                   # hidden layer 6
    self.fc7=nn.Linear(20,20)                   # hidden layer 7
    self.fc8=nn.Linear(20,20)                   # hidden layer 8
    self.fc9=nn.Linear(20,1)                   # last layer

  def forward(self,d):
    out=torch.tanh(self.fc1(d))              # seq_length * weights + bias for layer 1
    out=torch.tanh(self.fc2(out))            # seq_length * weights + bias for layer 2
    out=torch.tanh(self.fc3(out))            # seq_length * weights + bias for layer 3
    out=torch.tanh(self.fc4(out))            # seq_length * weights + bias for layer 4
    out=torch.tanh(self.fc5(out))            # seq_length * weights + bias for layer 5
    out=torch.tanh(self.fc6(out))            # seq_length * weights + bias for layer 5
    out=torch.tanh(self.fc7(out))            # seq_length * weights + bias for layer 5
    out=torch.tanh(self.fc8(out))            # seq_length * weights + bias for layer 5
    out=self.fc9(out)                        # seq_length * weights + bias for last layer
    return out                               # final outcome

input_dim = X_train.shape[1]
torch.manual_seed(42)  # to make initilized weights stable:
model = linearRegression(input_dim)

start = time.time()
print("starting time measurement")
loss = nn.MSELoss() # loss function
optimizers=optim.Adamax(params=model.parameters(),lr=0.001)

# training the model:
num_of_epochs = 150000
for i in range(num_of_epochs):
  # give the seq_length data to the architecure
  y_train_prediction=model(X_train)  # model initilizing
  loss_value=loss(y_train_prediction.squeeze(),y_train)   # find the loss function:
  optimizers.zero_grad() # make gradients zero for every iteration so next iteration it will be clear
  loss_value.backward()  # back propagation
  optimizers.step()  # update weights in NN

  # print the loss in training part:
  if i % 10 == 0:
    print(f'[epoch:{i}]: The loss value for training part={loss_value}')

end = time.time()
print("The downsampling is" + " " + str(Downsampling_factor))
print("Total time"+ " "+ str(end - start))

# y_train_prediction = np.zeros(y_train.shape[0])
# for i in range(y_train.shape[0]):
#     y_train_prediction[i]=model(X_train[i,:])

# plt.plot(y_train,'-x', label="True Values")
# plt.plot(y_train_prediction,'-x', label="Fitted")
# plt.show()

# with torch.no_grad():
#   model.eval()   # make model in evaluation stage
#   y_test_prediction=model(X_test)
#   test_loss=loss(y_test_prediction.squeeze(),y_test)
#   print(f'Test loss value : {test_loss.item():e}')
# # Inference with own data:
# pr = torch.tensor(torch.arange(1, 101).unsqueeze(dim=0), dtype=torch.float32).clone().detach()
# print(pr)

#The fitting
y_train_prediction = np.zeros(y_train.shape[0])
for i in range(y_train.shape[0]):
    y_train_prediction[i]=model(X_train[i,:])



#The extrapolation
model.eval()
x_ext = x_model[-1,:][None,:]
x_ext = torch.tensor(x_ext, dtype=torch.float32)
iterations = data.size-train_size
prediction = np.zeros(iterations)
for i in range(iterations):
    preds = model(x_ext[i,:])[:,None]
    x_new = x_ext[i,1:seq_length][None,:]
    x_new = torch.cat((x_new,preds), 1)
    x_ext = torch.cat((x_ext, x_new), 0)
    prediction[i] = preds.detach().numpy()[0]


#Concatenating fitting and prediction
full_prediction = np.concatenate((y_train_prediction,prediction),axis=None)
 

#Plotting
test_start_index =int(len(yy)*train_percentage/5)
plt.plot(data, label="True Values") #why the two here?
plt.plot(full_prediction, 'x', label="Extrapolation NN")
plt.plot(full_prediction[:train_size], 'x', label="Training NN")
plt.axvline(x=full_prediction[:train_size].shape[0], color='red', linestyle='--', label="Extrapolation start")
plt.xlabel("Time steps(k)",fontsize=20)
plt.ylabel("Amplitude [a.u.]",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.grid()
plt.legend(fontsize=20)
plt.title("NN Predictions",fontsize=20)
plt.show()


