import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv("C:/Users/shakgane/Downloads/Advertising.csv", index_col=0)


print(data.head())  

linreg = LinearRegression()

#sns.pairplot(data, x_vars = ['TV', 'radio', 'newspaper'], y_vars = ['sales'], kind='reg')

x_train, x_test, y_train, y_test = train_test_split(data['TV'].values.reshape(-1,1), data['sales'].values.reshape(-1,1),random_state=1)
linreg.fit(x_train, y_train)
ypred = linreg.predict(x_test)
print(r2_score(y_test, ypred))









  