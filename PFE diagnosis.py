import numpy as np
import pandas as pd
import warnings
from pandas_datareader import data as wb
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

import os
import matplotlib.pyplot as plt
import pmdarima
from pmdarima import auto_arima
from math import sqrt
tickers = ['^GSPC', 'PFE']

data = pd.DataFrame()
for t in tickers:
    data[t] = wb.DataReader(t, data_source='yahoo', start='2017-01-02')['Adj Close']

#(data / data.iloc[0] * 100).plot(figsize=(15, 6));
#plt.ylabel('Value ($)')
#plt.xlabel('Time')


log_returns = np.log(data/data.shift(1))
#Drop nan values
log_returns = data.dropna()
log_returns.to_csv(r'C:\Users\Ege\Dropbox\WiSo 21-22\Python Work\PFE\log_returns.csv')
log_returns = pd.read_csv(r'C:\Users\Ege\Dropbox\WiSo 21-22\Python Work\PFE\log_returns.csv',index_col= 'Date',parse_dates=True)
#Augmented Dicky Fuller Test for Stationary
warnings.filterwarnings("ignore")
for i in range(len(log_returns.columns)):
    result = adfuller(log_returns[log_returns.columns[i]])
    if result[1] > 0.05:
        print('{} - Series is not Stationary - Statistical properties of a system do not change over time- '.format(log_returns.columns[i]))
    else:
        print('{} - Series is  Stationary'.format(log_returns.columns[i]))

step_fit = auto_arima(log_returns['PFE'], trace=True, suppress_warnings=True)
step_fit.summary()

train = log_returns.iloc[:-300]
test = log_returns.iloc[-300:]
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train.PFE, order=(1, 1, 3))
model = model.fit()

test_mean = test['PFE'].mean()
print(test_mean)
print('efe')

start = len(train)

end = len(train)+len(test)-1
pred = model.predict(start=start, end=end, typ='levels')
pred.index = log_returns.index[start:end+1]
#pred.plot(legend=True)

#log_returns['PFE'].plot(legend=True)
#test['PFE'].plot(legend=True)

#plt.show()


start = len(train)
end = len(train)+len(test)-1
pred = model.predict(start=start, end=end, typ='levels')
pred.index = log_returns.index[start:end+1]
#pred.plot(legend=True)
pred.to_csv(r'C:\Users\Ege\Dropbox\WiSo 21-22\Python Work\PFE\pred2.csv')
test.to_csv(r'C:\Users\Ege\Dropbox\WiSo 21-22\Python Work\PFE\test.csv'),
train.to_csv(r'C:\Users\Ege\Dropbox\WiSo 21-22\Python Work\PFE\train.csv')

print(pred)
#log_returns['PFE'].plot(legend=True)
#test['PFE'].plot(legend=True)
#plt.show()
test_mean = test['PFE'].mean()
print(test_mean)


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(pred, test['PFE']))
print(rmse)

model2 = ARIMA(log_returns.PFE, order=(1, 1, 3))
model2 = model2.fit()
print(log_returns.tail())
index_future_dates = pd.date_range(start='2021-12-13', end='2022-10-12')
pred = model2.predict(start=len(log_returns), end=len(log_returns)+303, typ='levels').rename('ARIMA Predictions')
pred.index = index_future_dates
pred.plot(figsize=(15, 5), legend=True)
print('eye2')
plt.show()
print('fin')
