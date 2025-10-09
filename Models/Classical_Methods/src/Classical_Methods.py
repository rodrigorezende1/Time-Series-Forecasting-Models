#We will be using the Stasmodels library
import numpy as np
import matplotlib.pyplot as plt

# AR example
from statsmodels.tsa.ar_model import AutoReg
from random import random


# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = AutoReg(data, lags=1)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


#Moving Average MA
# contrived dataset
from statsmodels.tsa.arima.model import ARIMA

data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(0, 0, 1)) #the order should be high here =10 for good results
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()


# ARMA example
from statsmodels.tsa.arima.model import ARIMA
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(1, 0, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()


# ARIMA example # I should try this one
from statsmodels.tsa.arima.model import ARIMA
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data), typ='levels')
print(yhat)

plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()


# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)

plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()

# SARIMAX example
from statsmodels.tsa.statespace.sarimax import SARIMAX
# contrived dataset
data1 = [x + random() for x in range(1, 100)]
data2 = [x + random() for x in range(101, 200)]
# fit model
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
# make prediction
exog2 = [200 + random()]
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
print(yhat)

plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()

# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VAR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()

# VARMA example
from statsmodels.tsa.statespace.varmax import VARMAX
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
# fit model
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)


plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()


# VARMAX example
from statsmodels.tsa.statespace.varmax import VARMAX
# contrived dataset with dependency
data = list()
for i in range(100):
    v1 = i + random()
    v2 = v1 + random()
    row = [v1, v2]
    data.append(row)
data_exog = [x + random() for x in range(100)]
# fit model
model = VARMAX(data, exog=data_exog, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
data_exog2 = [[100]]
yhat = model_fit.forecast(exog=data_exog2)
print(yhat)

plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()


# SES example
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SimpleExpSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()


# HWES example
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = ExponentialSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(data))
print(yhat)


plt.plot(data,"x")
plt.plot(len(data),yhat,"o")
plt.show()
