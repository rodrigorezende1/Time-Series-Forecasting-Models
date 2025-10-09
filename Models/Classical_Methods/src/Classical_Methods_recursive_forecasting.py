#We will be using the Stasmodels library
import numpy as np
import matplotlib.pyplot as plt

# AR example
from statsmodels.tsa.ar_model import AutoReg
from random import random


# contrived dataset
data = [x + random() for x in range(1, 20)]
# fit model
data_training = np.zeros(len(data))
data_training[0:int(np.floor(len(data)/2))] = data[0:int(np.floor(len(data)/2))]
y_predictions = np.zeros(int(np.floor(len(data)/2)))
for i in range(int(np.floor(len(data)/2)),len(data)):
    model = AutoReg(data_training[0:i], lags=1)
    model_fit = model.fit()
    # make prediction
    data_training[i] = model_fit.predict(i, i)
    #yhat = model_fit.predict(i, i)
    #data_training[i] = yhat
    #y_predictions[i] = yhat
    #print(yhat)

plt.plot(data,"x",label="True Values")
plt.plot(data_training,"o",label="Model")
plt.plot([int(np.floor(len(data)/2)), int(np.floor(len(data)/2))], [1, 20], "r")
plt.show()

#Moving Average MA
# contrived dataset
from statsmodels.tsa.arima.model import ARIMA

data = [x + random() for x in range(1, 20)]
# fit model
data_training = np.zeros(len(data))
data_training[0:int(np.floor(len(data)/2))] = data[0:int(np.floor(len(data)/2))]
for i in range(int(np.floor(len(data)/2)),len(data)):
    model = ARIMA(data_training[0:i], order=(0, 0, 1))
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([int(np.floor(len(data)/2)), int(np.floor(len(data)/2))], [1, 20], "r")
plt.show()

# ARMA example
from statsmodels.tsa.arima.model import ARIMA
# contrived dataset
data = [x + random() for x in range(1, 20)]
# fit model
data_training = np.zeros(len(data))
data_training[0:int(np.floor(len(data)/2))] = data[0:int(np.floor(len(data)/2))]
for i in range(int(np.floor(len(data)/2)),len(data)):
    model = ARIMA(data_training[0:i], order=(1, 0, 1))
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([int(np.floor(len(data)/2)), int(np.floor(len(data)/2))], [1, 20], "r")
plt.show()

# ARIMA example # I should try this one # The best until now
from statsmodels.tsa.arima.model import ARIMA
# contrived dataset
data = [x + random() for x in range(1, 20)]
# fit model
data_training = np.zeros(len(data))
data_training[0:int(np.floor(len(data)/2))] = data[0:int(np.floor(len(data)/2))]
for i in range(int(np.floor(len(data)/2)),len(data)):
    model = ARIMA(data_training[0:i], order=(1, 1, 1))
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i, typ='levels')

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([int(np.floor(len(data)/2)), int(np.floor(len(data)/2))], [1, 20], "r")
plt.show()

# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
# contrived dataset
data = [x + random() for x in range(1, 20)]
# fit model
data_training = np.zeros(len(data))
data_training[0:int(np.floor(len(data)/2))] = data[0:int(np.floor(len(data)/2))]
for i in range(int(np.floor(len(data)/2)),len(data)):
    model = SARIMAX(data_training[0:i], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_fit = model.fit(disp=False)
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([int(np.floor(len(data)/2)), int(np.floor(len(data)/2))], [1, 20], "r")
plt.show()

# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# contrived dataset
data = [x + random() for x in range(1, 20)]
# fit model
data_training = np.zeros(len(data))
data_training[0:int(np.floor(len(data)/2))] = data[0:int(np.floor(len(data)/2))]
for i in range(int(np.floor(len(data)/2)),len(data)):
    model = SimpleExpSmoothing(data_training[0:i])
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([int(np.floor(len(data)/2)), int(np.floor(len(data)/2))], [1, 20], "r")
plt.show()


# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# contrived dataset
data = [x + random() for x in range(1, 20)]
# fit model
data_training = np.zeros(len(data))
data_training[0:int(np.floor(len(data)/2))] = data[0:int(np.floor(len(data)/2))]
for i in range(int(np.floor(len(data)/2)),len(data)):
    model = ExponentialSmoothing(data_training[0:i])
    model_fit = model.fit()
    data_training[i] = model_fit.predict(i, i)

plt.plot(data,"x")
plt.plot(data_training,"o")
plt.plot([int(np.floor(len(data)/2)), int(np.floor(len(data)/2))], [1, 20], "r")
plt.show()

