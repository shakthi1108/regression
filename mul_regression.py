import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("C:/Users/shakgane/Downloads/Advertising.csv", index_col=0)
print(data.head())  


#extracting the features
feature_col = ['TV','radio','newspaper']
x = data[feature_col]
#x = [['TV', 'radio', 'newspaper']]

y = data['sales']


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1)
print(x_train.shape)
print(x_test.shape)


linreg.fit(x_train, y_train)

print(linreg.intercept_)
print(linreg.coef_)

ypred = linreg.predict(x_test)

print(r2_score(y_test, ypred))