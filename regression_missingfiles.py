import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("C:/Users/shakgane/Downloads/real-estate-transaction-master/realestatetransactions.csv", parse_dates=["sale_date"])

data.replace(0, np.nan, inplace=True)

new_data = data.fillna({
        'sq__ft' : '0'
        })

new_data = data['sq__ft'].interpolate()

data['sq__ft'] =new_data

linreg = LinearRegression()



x_train, x_test, y_train, y_test = train_test_split(data['sq__ft'].values.reshape(-1,1), data['price'].values.reshape(-1,1),random_state=1)
linreg.fit(x_train, y_train)
ypred = linreg.predict(x_test)
print(r2_score(y_test, ypred))


