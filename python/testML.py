# -*- coding: utf-8 -*-
"""
/////////////////////////////////////////////////////////////////////////////
//
//   Project/Path:      %M%
//   Last Change Set:   %L% (%G% %U%)
//
/////////////////////////////////////////////////////////////////////////////
//
//   COPYRIGHT Vusion Technologies, all rights reserved.
//
//   No part of this software may be reproduced or modified in any
//   form or by any means - electronic, mechanical, photocopying,
//   recording, or otherwise - without the prior written consent of
//   Vusion Technologies.
//

@author: jacky.chow
"""

# importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import neighbors
from sklearn import metrics
# For least squares fit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, K1, K2, K3):
    return x[1]*(K1 * np.power(x[0],2) + K2 * np.power(x[0],4) + K3 * np.power(x[0],6))

D = 10;
N = 1000; 
K = 3; # nearest neighbour in KNN
noise = 0.5;

# Create the dataset
rng = np.random.RandomState(1)
#X = np.linspace(0, 6, 100)[:, np.newaxis]
#y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

X = np.linspace(-2000, 2000, 100)[:, np.newaxis]
#x_bar = X/1000.0;
#y_bar = X/1000.0;
x_bar = X;
y_bar = X;
r = np.sqrt( np.power(x_bar,2) + np.power(y_bar,2) );
K1 = -1.1e-08; K2 = -1.3e-15; K3 = 0.0;
#K1 = -1.3e+02; K2 = 5.5e+00; K3 = -3.0e+00;
y = x_bar*(K1 * np.power(r,2) + K2 * np.power(r,4) + K3 * np.power(r,6))
y = y.ravel()
yTest = y;
y = y + rng.normal(0, noise, X.shape[0]);
yTest = yTest + rng.normal(0, noise, X.shape[0]);
np.disp("Residual noise: " + str(noise))

# Least squares polynomial fit
x_bar*(K1 * np.power(r,2) + K2 * np.power(r,4) + K3 * np.power(r,6))

func(np.concatenate((r.transpose(),r.transpose()),axis=0), K1, K2, K3)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=D)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=D),
                          n_estimators=N, random_state=rng)

regr_3 = neighbors.KNeighborsRegressor(n_neighbors=K, weights='uniform', n_jobs=1)

regr_4 =  AdaBoostRegressor(neighbors.KNeighborsRegressor(n_neighbors=K),
                          n_estimators=N, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)
regr_4.fit(X, y)

# Predict
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)
y_3 = regr_3.predict(X)
y_4 = regr_4.predict(X)


np.disp("  Decision Tree Error: Depth=" + str(D))
np.disp("    L2-Norm Error: " + str(np.linalg.norm(y_1-y)) )
np.disp("    Average Error: " + str(np.mean(y_1-y)))
np.disp("    First Error: " + str(np.linalg.norm(y_1[0]-y[0])) )
np.disp("    Last Error: " + str(np.linalg.norm(y_1[len(y)-1]-y[len(y)-1])) )

np.disp("  Adaboost Decision Tree: Depth=" + str(D) + ", nEstimator=" + str(N))
np.disp("    L2-Norm Error: " + str(np.linalg.norm(y_2-y)) )
np.disp("    Average Error: " + str(np.mean(y_2-y)) )
np.disp("    First Error: " + str(np.linalg.norm(y_2[0]-y[0])) )
np.disp("    Last Error: " + str(np.linalg.norm(y_2[len(y)-1]-y[len(y)-1])) )

np.disp("  KNN: K=" + str(K) )
np.disp("    L2-Norm Error: " + str(np.linalg.norm(y_3-y)) )
np.disp("    Average Error: " + str(np.mean(y_3-y)) )
np.disp("    First Error: " + str(np.linalg.norm(y_3[0]-y[0])) )
np.disp("    Last Error: " + str(np.linalg.norm(y_3[len(y)-1]-y[len(y)-1])) )

np.disp("  Adaboost KNN: K=" + str(K)  + ", nEstimator=" + str(N))
np.disp("    L2-Norm Error: " + str(np.linalg.norm(y_4-y)) )
np.disp("    Average Error: " + str(np.mean(y_4-y)) )
np.disp("    First Error: " + str(np.linalg.norm(y_4[0]-y[0])) )
np.disp("    Last Error: " + str(np.linalg.norm(y_4[len(y)-1]-y[len(y)-1])) )

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.plot(X, y_2, c="r", label="AdaboostTree="+ str(N), linewidth=2)
plt.plot(X, y_3, c="b", label="kNN="+ str(K), linewidth=2)
plt.plot(X, y_4, c="m", label="AdaboostKNN="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Comparison of ML Methods")
plt.legend()

plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X, y-y_1, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("y - yTrue")
fig = plt.tight_layout()

plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_2, c="r", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Adaboost Decision Tree Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X, y-y_2, c="r", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("y - yTrue")
fig = plt.tight_layout()

plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_3, c="b", label="kNN="+ str(K), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("kNN Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X, y-y_3, c="b", label="kNN="+ str(K), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("y - yTrue")
fig = plt.tight_layout()

plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_4, c="m", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Adaboost KNN Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X, y-y_4, c="m", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("y - yTrue")
fig = plt.tight_layout()

plt.show()


##############################################################
### Testing decision tree hyperparamters
maxDepth = 50;
scoreTraining = np.zeros((maxDepth))
scoreTesting = np.zeros((maxDepth))
for n in range(0,maxDepth):
    regr = DecisionTreeRegressor(max_depth=n+1)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    scoreTraining[n] = np.sqrt( metrics.mean_squared_error(y, y_pred) )
    

plt.figure()
plt.plot(range(1,maxDepth+1), scoreTraining, linewidth=2)
plt.xlabel("Depth")
plt.ylabel("RMSE")
plt.title("Decision Tree Regression")


##############################################################
### Testing adaboost decision tree hyperparamters
maxN = 1000;
increment = 100;
scoreTraining = np.zeros(len(range(0,maxN,increment)))
scoreTesting = np.zeros(len(range(0,maxN,increment)))
D = 90;
i = 0;
for n in range(0,maxN,increment):
    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=D),
                          n_estimators=n+1, random_state=rng)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    scoreTraining[i] = np.sqrt( metrics.mean_squared_error(y, y_pred) )
    i = i + 1;
    
plt.figure()
plt.plot(range(0,maxN,increment), scoreTraining, label='Depth='+str(D), linewidth=2)

D = 150;
i = 0;
for n in range(0,maxN,increment):
    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=D),
                          n_estimators=n+1, random_state=rng)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    scoreTraining[i] = np.sqrt( metrics.mean_squared_error(y, y_pred) )
    i = i + 1;
   
plt.plot(range(0,maxN,increment), scoreTraining, label='Depth='+str(D), linewidth=2)

D = 25;
i = 0;
for n in range(0,maxN,increment):
    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=D),
                          n_estimators=n+1, random_state=rng)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    scoreTraining[i] = np.sqrt( metrics.mean_squared_error(y, y_pred) )
    i = i + 1;
   
plt.plot(range(0,maxN,increment), scoreTraining, label='Depth='+str(D), linewidth=2)

plt.xlabel("Number Estimator")
plt.ylabel("RMSE")
plt.title("Adaboost Decision Tree Regression")
plt.legend()

##############################################################
### Testing KNN hyperparamters
maxK = 10;
scoreTraining = np.zeros((maxK))
scoreTesting = np.zeros((maxK))
for n in range(0,maxK):
    regr = neighbors.KNeighborsRegressor(n_neighbors=n+1, weights='uniform', n_jobs=1)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    scoreTraining[n] = np.sqrt( metrics.mean_squared_error(y, y_pred) )
    

plt.figure()
plt.plot(range(1,maxK+1), scoreTraining, linewidth=2)
plt.xlabel("K")
plt.ylabel("RMSE")
plt.title("KNN Regression")
