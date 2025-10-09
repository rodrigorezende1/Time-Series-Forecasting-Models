#We will be using the Stasmodels library
import numpy as np
import matplotlib.pyplot as plt

# AR example
from statsmodels.tsa.ar_model import AutoReg

# contrived dataset
data = np.sin(np.linspace(0,10,250)*2*np.pi)*np.exp(-0.5*np.linspace(0,10,250))
# fit model
data_training = np.zeros(len(data))
train_percentage = int(0.3*len(data))
data_training[0:train_percentage] = data[0:train_percentage]
for i in range(train_percentage,len(data)):
    model = AutoReg(data_training[0:i], lags=1)
    model_fit = model.fit()
    # make prediction
    data_training[i] = model_fit.predict(i, i)
    
plt.plot(data,"x",label="True Values")
plt.plot(data_training,"o",label="Model")
plt.plot([train_percentage, train_percentage], [-1, 1], "r")
plt.show()

#Moving Average MA
# contrived dataset
from statsmodels.tsa.arima.model import ARIMA

data = np.sin(np.linspace(0,10,250)*2*np.pi)*np.exp(-0.5*np.linspace(0,10,250))
# fit model
data_training = np.zeros(len(data))
train_percentage = int(0.3*len(data))
data_training[0:train_percentage] = data[0:train_percentage]
for i in range(train_percentage,len(data)):
    model = ARIMA(data_training[0:i], order=(0, 0, 1))
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([train_percentage, train_percentage], [-1, 1], "r")
plt.show()

# ARMA example
from statsmodels.tsa.arima.model import ARIMA
# contrived dataset
data = np.sin(np.linspace(0,10,250)*2*np.pi)*np.exp(-0.5*np.linspace(0,10,250))
# fit model
data_training = np.zeros(len(data))
train_percentage = int(0.3*len(data))
data_training[0:train_percentage] = data[0:train_percentage]
for i in range(train_percentage,len(data)):
    model = ARIMA(data_training[0:i], order=(1, 0, 1))
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([train_percentage, train_percentage], [-1, 1], "r")
plt.show()

# ARIMA example # I should try this one
from statsmodels.tsa.arima.model import ARIMA
# contrived dataset
data = np.sin(np.linspace(0,10,250)*2*np.pi)*np.exp(-0.5*np.linspace(0,10,250))

# fit model
data_training = np.zeros(len(data))
train_percentage = int(0.3*len(data))
data_training[0:train_percentage] = data[0:train_percentage]
for i in range(train_percentage,len(data)):
    model = ARIMA(data_training[0:i], order=(12, 0, 4))#order=(12, 0, 4)
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i, typ='levels')

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([train_percentage, train_percentage], [-1, 1], "r")
plt.show()


# SARIMA example # The best until now
from statsmodels.tsa.statespace.sarimax import SARIMAX
# contrived dataset
data = np.sin(np.linspace(0,10,250)*2*np.pi)*np.exp(-0.5*np.linspace(0,10,250))
# fit model
data_training = np.zeros(len(data))
train_percentage = int(0.3*len(data))
data_training[0:train_percentage] = data[0:train_percentage]
for i in range(train_percentage,len(data)):
    model = SARIMAX(data_training[0:i], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([train_percentage, train_percentage], [-1, 1], "r")
plt.show()

# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# contrived dataset
data = np.sin(np.linspace(0,10,250)*2*np.pi)*np.exp(-0.5*np.linspace(0,10,250))
# fit model
data_training = np.zeros(len(data))
train_percentage = int(0.3*len(data))
data_training[0:train_percentage] = data[0:train_percentage]
for i in range(train_percentage,len(data)):
    model = SimpleExpSmoothing(data_training[0:i])
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([train_percentage, train_percentage], [-1, 1], "r")
plt.show()


# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# contrived dataset
data = np.sin(np.linspace(0,10,250)*2*np.pi)*np.exp(-0.5*np.linspace(0,10,250))
# fit model
data_training = np.zeros(len(data))
train_percentage = int(0.3*len(data))
data_training[0:train_percentage] = data[0:train_percentage]
for i in range(train_percentage,len(data)):
    model = ExponentialSmoothing(data_training[0:i])
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([train_percentage, train_percentage], [-1, 1], "r")
plt.show()

