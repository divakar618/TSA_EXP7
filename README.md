# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 
## Name : Divakar R
## Register number : 212222240026


### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM

```python

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

df=pd.read_csv('powerconsumption.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(df['Total'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Plot ACF and PACF
plt.figure(figsize=(10,5))
plt.subplot(121)
plot_acf(df['Total'], lags=13, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(df['Total'], lags=13, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(df) * 0.8)
train, test = df['Total'][:train_size], df['Total'][train_size:]

# Fit AutoRegressive model with 13 lags
model = AutoReg(train, lags=13)
model_fit = model.fit()

# Make predictions on the test data
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the test data and the predictions
plt.plot(test.index, test, label='Actual', marker='o')
plt.plot(test.index, predictions, label='Predicted', marker='x')
plt.title('Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Total')
plt.legend()
plt.show()
```

### OUTPUT:

GIVEN DATA
![376885857-8c1127a2-8bc2-4b37-ba97-966af271afba](https://github.com/user-attachments/assets/6f7b56da-3c89-497f-bd4d-fb3212a69cee)

PACF - ACF
![376885919-fc6c89d7-c791-45f6-af18-3f29492fe52a](https://github.com/user-attachments/assets/98a10a39-5127-4d5e-bb97-40f5db5fc095)


PREDICTION
![376885983-8902bf01-88cd-42dc-b933-b0539bf9b270](https://github.com/user-attachments/assets/b84bd17e-49f6-4dd8-81de-0d7732f70f0f)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
