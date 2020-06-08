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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# For least squares fit
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, K1, K2, K3):
    return x[1]*(K1 * np.power(x[0],2) + K2 * np.power(x[0],4) + K3 * np.power(x[0],6))


# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(X, y, tree_depths, cv=5, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeRegressor(max_depth=depth)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores
  
# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-1.96*cv_scores_std, cv_scores_mean+1.96*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()



##########################################################################
### User parameter
##########################################################################
# Decision Tree
D = 10;
minDepth = 1;
maxDepth = 50;
minLeafSample = 2;
maxLeafSample = 4; # max leaf size is recommended by Rafael Gomes Mantovani, Tomáš Horváth, Ricardo Cerri, Sylvio Barbon Junior, Joaquin Vanschoren, André Carlos Ponce de Leon Ferreira de Carvalho, “An empirical study on hyperparameter tuning of decision trees” arXiv:1812.02207
minLeaf = 0; # the CV tuned best minLeafSize, don't touch or change this

# Adaboost
N = 1000; 

# K-Nearest Neighbour
K = 3; # nearest neighbour in KNN
minK = 2;
maxK = 50;
noise = 0.5;


###########################################################################
### Create the dataset
###########################################################################
rng = np.random.RandomState(1)
#X = np.linspace(0, 6, 100)[:, np.newaxis]
#y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

X = np.linspace(-2000, 2000, 1000)[:, np.newaxis]

#x_bar = X/1000.0;
#y_bar = X/1000.0;
x_bar = X;
y_bar = X;
r = np.sqrt( np.power(x_bar,2) + np.power(y_bar,2) );
K1 = -1.1e-08; K2 = -1.3e-15; K3 = -2.0e-24;
#K1 = -1.3e+02; K2 = 5.5e+00; K3 = -3.0e+00;
y = x_bar*(K1 * np.power(r,2) + K2 * np.power(r,4) + K3 * np.power(r,6))
y = y.ravel()

#
#yTest = y;
#y = y + rng.normal(0, noise, X.shape[0]);
#yTest = yTest + rng.normal(0, noise, X.shape[0]);
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

I = np.argsort(X_train, axis=0).ravel();
X_train = X_train[I]
y_train = y_train[I]
J = np.argsort(X_test, axis=0).ravel();
X_test = X_test[J]
y_test = y_test[J]


np.disp("Residual noise: " + str(noise))

######################################################################
# Least squares polynomial fit
x_bar*(K1 * np.power(r,2) + K2 * np.power(r,4) + K3 * np.power(r,6))
func(np.concatenate((r.transpose(),r.transpose()),axis=0), K1, K2, K3)


######################################################################
# Fit regression model

# get the optimal depth for decision tree
print ("  Decision Tree Cross-Validation of depth")
t0 = time()
param_grid = [ {'max_depth' : range(minDepth,maxDepth,1)} ]
rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=2652124)
#rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=2652124)
regCV = GridSearchCV(DecisionTreeRegressor(random_state=1,min_samples_leaf=3), param_grid, cv=rkf, verbose = 0,n_jobs=1,refit=True, scoring='neg_mean_squared_error')
#regCV = GridSearchCV(DecisionTreeRegressor(random_state=1,min_samples_leaf=2), param_grid, cv=10, verbose = 0,n_jobs=1,refit=True, scoring='neg_mean_squared_error')
#regCV = GridSearchCV(DecisionTreeRegressor(random_state=1), param_grid, cv=10, verbose = 0,n_jobs=1,refit=True, scoring=None)
regCV.fit(X_train, y_train)
print ("    Best in sample score: ", regCV.best_score_)
print ("    CV value for maxDepth ( between ", minDepth, " and", maxDepth-1,"): ", regCV.best_estimator_.max_depth)
print ("    Training Decision Tree Regressor + CV time:", round(time()-t0, 3), "s")
D = regCV.best_estimator_.max_depth; 
regr_0 = regCV.best_estimator_;

print ("  Decision Tree Cross-Validation of leaf size")
t0 = time()
param_grid = [ {'min_samples_leaf' : range(minLeafSample,maxLeafSample,1)} ]
rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=2652124)
regCV = GridSearchCV(DecisionTreeRegressor(random_state=1,), param_grid, cv=rkf, verbose = 0,n_jobs=1,refit=True, scoring='neg_mean_squared_error')
regCV.fit(X_train, y_train)
print ("    Best in sample score: ", regCV.best_score_)
print ("    CV value for minLeafSize ( between ", minLeafSample, " and", maxLeafSample-1,"): ", regCV.best_estimator_.min_samples_leaf)
print ("    Training Decision Tree Regressor + CV time:", round(time()-t0, 3), "s")
minLeaf = regCV.best_estimator_.min_samples_leaf;
regr_1 = regCV.best_estimator_;

#print ("  Decision Tree Cross-Validation of depth + leaf size")
#t0 = time()
#param_grid = [ {'max_depth' : range(minDepth,maxDepth,1), 'min_samples_leaf' : range(minLeafSample,maxLeafSample,1)} ] 
#regCV = GridSearchCV(DecisionTreeRegressor(random_state=1), param_grid, cv=10, verbose = 0,n_jobs=1,refit=True, scoring='neg_mean_squared_error')
#regCV.fit(X_train, y_train)
#print ("    Best in sample score: ", regCV.best_score_)
#print ("    CV value for maxDepth ( between ", minDepth, " and", maxDepth-1,"): ", regCV.best_estimator_.max_depth)
#print ("    CV value for minLeafSize ( between ", minLeafSample, " and", maxLeafSample-1,"): ", regCV.best_estimator_.min_samples_leaf)
#print ("    Training Decision Tree Regressor + CV time:", round(time()-t0, 3), "s")
#minLeaf = regCV.best_estimator_.min_samples_leaf;
#regr_1 = regCV.best_estimator_;


# get the optimal k for KNN
print ("  KNN Cross-Validation")
t0 = time()
param_grid = [ {'n_neighbors' : range(minK,maxK,1)} ] # test only up to 50 neighbours
regCV = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0, n_jobs=1, scoring='neg_mean_squared_error')
#regCV = GridSearchCV(neighbors.KNeighborsRegressor(weights='uniform'), param_grid, cv=10, verbose = 0,n_jobs=1)
regCV.fit(X_train, y_train)
print ("    Best in sample score: ", regCV.best_score_)
print ("    CV value for K ( between", minK, " and", maxK-1,"): ", regCV.best_estimator_.n_neighbors)
print ("    Training NN-Regressor + CV time:", round(time()-t0, 3), "s")
K = regCV.best_estimator_.n_neighbors;
regr_3 = regCV.best_estimator_;

#regr_1 = DecisionTreeRegressor(max_depth=D)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(min_samples_leaf=minLeaf),
                          n_estimators=N, random_state=1)

#regr_3 = neighbors.KNeighborsRegressor(n_neighbors=K, weights='uniform', n_jobs=1)

regr_4 =  AdaBoostRegressor(neighbors.KNeighborsRegressor(n_neighbors=K),
                          n_estimators=N, random_state=1)

#regr_1.fit(X, y)
regr_2.fit(X_train, y_train)
#regr_3.fit(X, y)
regr_4.fit(X_train, y_train)

# Predict
y_0_train = regr_0.predict(X_train)
y_1_train = regr_1.predict(X_train)
y_2_train = regr_2.predict(X_train)
y_3_train = regr_3.predict(X_train)
y_4_train = regr_4.predict(X_train)

y_0_test = regr_0.predict(X_test)
y_1_test = regr_1.predict(X_test)
y_2_test = regr_2.predict(X_test)
y_3_test = regr_3.predict(X_test)
y_4_test = regr_4.predict(X_test)

np.disp("  Decision Tree Error: depth=" + str(D))
np.disp("    Training Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_0_train-y_train)) )
np.disp("      Average Error: " + str(np.mean(y_0_train-y_train)) + " (" + str(np.min(y_0_train-y_train)) + " to " + str(np.max(y_0_train-y_train)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_0_train[0]-y_train[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_0_train[len(y_0_train)-1]-y_train[len(y_train)-1])) )
np.disp("    Testing Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_0_test-y_test)) )
np.disp("      Average Error: " + str(np.mean(y_0_test-y_test)) + " (" + str(np.min(y_0_test-y_test)) + " to " + str(np.max(y_0_test-y_test)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_0_test[0]-y_test[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_0_test[len(y_0_test)-1]-y_test[len(y_test)-1])) )

np.disp("  Decision Tree Error: minLeafSize=" + str(minLeaf))
np.disp("    Training Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_1_train-y_train)) )
np.disp("      Average Error: " + str(np.mean(y_1_train-y_train)) + " (" + str(np.min(y_1_train-y_train)) + " to " + str(np.max(y_1_train-y_train)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_1_train[0]-y_train[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_1_train[len(y_1_train)-1]-y_train[len(y_train)-1])) )
np.disp("    Testing Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_1_test-y_test)) )
np.disp("      Average Error: " + str(np.mean(y_1_test-y_test)) + " (" + str(np.min(y_1_test-y_test)) + " to " + str(np.max(y_1_test-y_test)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_1_test[0]-y_test[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_1_test[len(y_1_test)-1]-y_test[len(y_test)-1])) )

np.disp("  Adaboost Decision Tree: minLeafSize=" + str(minLeaf) + ", nEstimator=" + str(N))
np.disp("    Training Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_2_train-y_train)) )
np.disp("      Average Error: " + str(np.mean(y_2_train-y_train)) + " (" + str(np.min(y_2_train-y_train)) + " to " + str(np.max(y_2_train-y_train)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_2_train[0]-y_train[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_2_train[len(y_2_train)-1]-y_train[len(y_train)-1])) )
np.disp("    Testing Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_2_test-y_test)) )
np.disp("      Average Error: " + str(np.mean(y_2_test-y_test)) + " (" + str(np.min(y_2_test-y_test)) + " to " + str(np.max(y_2_test-y_test)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_2_test[0]-y_test[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_2_test[len(y_2_test)-1]-y_test[len(y_test)-1])) )

np.disp("  KNN: K=" + str(K) )
np.disp("    Training Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_3_train-y_train)) )
np.disp("      Average Error: " + str(np.mean(y_3_train-y_train)) + " (" + str(np.min(y_3_train-y_train)) + " to " + str(np.max(y_3_train-y_train)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_3_train[0]-y_train[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_3_train[len(y_3_train)-1]-y_train[len(y_train)-1])) )
np.disp("    Testing Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_3_test-y_test)) )
np.disp("      Average Error: " + str(np.mean(y_3_test-y_test)) + " (" + str(np.min(y_3_test-y_test)) + " to " + str(np.max(y_3_test-y_test)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_3_test[0]-y_test[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_3_test[len(y_3_test)-1]-y_test[len(y_test)-1])) )

np.disp("  Adaboost KNN: K=" + str(K)  + ", nEstimator=" + str(N))
np.disp("    Training Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_4_train-y_train)) )
np.disp("      Average Error: " + str(np.mean(y_4_train-y_train)) + " (" + str(np.min(y_4_train-y_train)) + " to " + str(np.max(y_4_train-y_train)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_4_train[0]-y_train[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_4_train[len(y_4_train)-1]-y_train[len(y_train)-1])) )
np.disp("    Testing Error" )
np.disp("      L2-Norm Error: " + str(np.linalg.norm(y_4_test-y_test)) )
np.disp("      Average Error: " + str(np.mean(y_4_test-y_test)) + " (" + str(np.min(y_4_test-y_test)) + " to " + str(np.max(y_4_test-y_test)) + ")")
np.disp("      First Error: " + str(np.linalg.norm(y_4_test[0]-y_test[0])) )
np.disp("      Last Error: " + str(np.linalg.norm(y_4_test[len(y_4_test)-1]-y_test[len(y_test)-1])) )

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X_train, y_1_train, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.plot(X_train, y_2_train, c="r", label="AdaboostTree="+ str(N), linewidth=2)
plt.plot(X_train, y_3_train, c="b", label="kNN="+ str(K), linewidth=2)
plt.plot(X_train, y_4_train, c="m", label="AdaboostKNN="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Training: Comparison of ML Methods")
plt.legend()
##################################################################
plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X_train, y_1_train, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X_train, y_train-y_1_train, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("Training: y - yTrue")
fig = plt.tight_layout()

plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X_test, y_1_test, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X_test, y_test-y_1_test, c="g", label="Decision Tree="+ str(D), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("Testing: y - yTrue")
fig = plt.tight_layout()

#######################################################################
plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X_train, y_2_train, c="r", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Adaboost Decision Tree Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X_train, y_train-y_2_train, c="r", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("Train: y - yTrue")
fig = plt.tight_layout()

plt.figure()
fig = plt.subplot(121)
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X_test, y_2_test, c="r", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Adaboost Decision Tree Regression")
plt.legend()
fig = plt.subplot(122)
plt.plot(X_test, y_test-y_2_test, c="r", label="Adaboost="+ str(N), linewidth=2)
plt.xlabel("data")
plt.ylabel("difference")
plt.title("Test: y - yTrue")
fig = plt.tight_layout()

#################################################################
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
#maxDepth = 50;
#scoreTraining = np.zeros((maxDepth))
#scoreTesting = np.zeros((maxDepth))
#for n in range(0,maxDepth):
#    regr = DecisionTreeRegressor(max_depth=n+1)
#    regr.fit(X, y)
#    y_pred = regr.predict(X)
#    scoreTraining[n] = np.sqrt( metrics.mean_squared_error(y, y_pred) )
#    
#
#plt.figure()
#plt.plot(range(1,maxDepth+1), scoreTraining, linewidth=2)
#plt.xlabel("Depth")
#plt.ylabel("RMSE")
#plt.title("Decision Tree Regression")

maxLeaf = 20;
scoreTraining = np.zeros((maxLeaf))
for n in range(0,maxLeaf):
    regr = DecisionTreeRegressor(min_samples_leaf=n+1)
    regr.fit(X, y)
    y_pred = regr.predict(X)
    scoreTraining[n] = np.sqrt( metrics.mean_squared_error(y, y_pred) )
    

plt.figure()
plt.plot(range(1,maxLeaf+1), scoreTraining, linewidth=2)
plt.xlabel("Leaf Size")
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
maxK = 20;
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

##############################################################

# fitting trees of depth 1 to 24
sm_tree_depths = range(1,25)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train, sm_tree_depths)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
                               'Accuracy per decision tree depth on training data')

