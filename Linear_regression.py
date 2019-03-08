import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from matplotlib import style
from scipy.interpolate import *

data = pd.read_csv("C:/Users/shakgane/Downloads/Advertising.csv", index_col=0)


print(data.head())  

#data['TV'] = data['TV'].apply(lambda v:(v-data['TV'].min())/(data['TV'].max()-data['TV'].min()))

#print(data['TV'].head())

#linreg = LinearRegression()

#data.plot.box()
#plt.xticks(list(range(len(data.columns))), data.columns, rotation='vertical')

#sns.pairplot(data, x_vars = ['TV', 'radio', 'newspaper'], y_vars = ['sales'], kind='reg')
p1 = polyfit(data['TV'], data['sales'], 1)
plt.plot(data['TV'], data['sales'], 'o', color='r')
plt.plot(data['TV'], polyval(p1,data['TV']), 'k-')
plt.title('TV vs Sales')
plt.xlabel('TV')
plt.ylabel('sales')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data['TV'].values.reshape(-1,1), data['sales'].values.reshape(-1,1),random_state=1)
linreg.fit(x_train, y_train)
ypred = linreg.predict(x_test)
print(r2_score(y_test, ypred))









  