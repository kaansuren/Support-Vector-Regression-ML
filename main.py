# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading Csv
data = pd.read_csv("rank_salary.csv")
print(data)

# Slicing
x = data.iloc[:,1:2]
y = data.iloc[:,2:]

# Converting to Numpy arrays
X = x.values
Y = y.values

# Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
Y_scaled = sc.fit_transform(Y)

# Support Vector Regression with Radial Basis Function
from sklearn.svm import SVR
svr_reg = SVR(kernel = "rbf")
svr_reg.fit(X_scaled, Y_scaled)

# Visualization - 1
plt.scatter(X_scaled, Y_scaled)
plt.plot(X_scaled, svr_reg.predict(X_scaled))
plt.show()

print(svr_reg.predict([[5.5]]))

# Support Vector Regression with Polynomial
from sklearn.svm import SVR
svr_reg2 = SVR(kernel = "poly")
svr_reg2.fit(X_scaled, Y_scaled)

# Visualization - 2
plt.scatter(X_scaled, Y_scaled)
plt.plot(X_scaled, svr_reg2.predict(X_scaled))
plt.show()

print(svr_reg2.predict([[5.5]]))


# Support Vector Regression with Sigmoid
from sklearn.svm import SVR
svr_reg3 = SVR(kernel = "sigmoid")
svr_reg3.fit(X_scaled, Y_scaled)

# Visualization - 3
plt.scatter(X_scaled, Y_scaled)
plt.plot(X_scaled, svr_reg3.predict(X_scaled))
plt.show()

print(svr_reg3.predict([[5.5]]))

