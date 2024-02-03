import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('salaries.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(sparse_output=False),[1,2,3,4,6,8,9])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train[:,[0,5,7]] = sc_x.fit_transform(x_train[:,[0,5,7]])
x_test[:,[0,5,7]] = sc_x.transform(x_test[:,[0,5,7]])
y_train = y_train.reshape(len(y_train),1)
y_test = y_test.reshape(len(y_test),1)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(10,random_state=0)
regressor.fit(x_train,y_train)

#from xgboost import XGBRegressor
#regressor = XGBRegressor()
#regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

<!---
Bhagya-MP/Bhagya-MP is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
