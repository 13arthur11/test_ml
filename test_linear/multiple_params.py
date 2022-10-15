import pandas as pd
import numpy as np
!pip install scikit-learn
!pip install matplotlib
import sys
sys.path.append("./venv/lib/python3.10/site-packages")
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model
import math


data = pd.read_csv("data.csv")

# plt.scatter(data.area, data.Price, marker="+")
# plt.xlabel('area(sqrt ft)')
# plt.ylabel("price")
median_bedroms = math.floor(data.bedroms.median()) 
data.bedroms = data.bedroms.fillna(median_bedroms)
data


reg = linear_model.LinearRegression()
reg.fit(data[['area', 'bedroms', 'age']], data.price)
# rg.predict([[3300]])
m = reg.coef_
b = reg.intercept_
m

x = reg.predict(data[['area', 'bedroms', 'age']])

# plt.scatter(data.area, data.Price, marker="+", color='red')
plt.plot(data.age, x)
