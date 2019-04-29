# -*- coding: utf-8 -*-
"""
@author: Lam Hin Tai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# %matplotlib inline

# set dir
path = r'C:\Users\Tai\Documents\MStat 2018\8017 project\stage'

listing_119_file = os.path.join(path, 'full_from_proposal_119_with_id.csv')
listing_119 = pd.read_csv(listing_119_file)

# dependent variable
Y = np.log(listing_119['total_price'])

# we do not fit host_id as a regressor
X = listing_119.drop(['host_id', 'total_price', 'availability_365'], axis=1)

# Load data and data preparation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                    random_state=40)

import multiprocessing
n_cores = multiprocessing.cpu_count()-1

# Load the regressor models and obtain residuals
import pickle
in_file_name = os.path.join(path, 'rf_gs_regressor.pickle')
with open(in_file_name, 'rb') as f:
    rf_price_model = pickle.load(f)

in_file_name = os.path.join(path, 'xgb_gs_regressor.pickle')
with open(in_file_name, 'rb') as f:
    xgb_price_model = pickle.load(f)

# Training set
Y_train_res_rf_rgr = Y_train - rf_price_model.predict(X_train)
Y_train_res_xgb_rgr = Y_train - xgb_price_model.predict(X_train)

# Testing set - hold out for testing
Y_test_res_rf_rgr = Y_test - rf_price_model.predict(X_test)
Y_test_res_xgb_rgr = Y_test - xgb_price_model.predict(X_test)

# check the 2 price model's residuals' sign agreement
Y_sign_train_rf_rgr = np.sign(Y_train_res_rf_rgr)
Y_sign_train_xgb_rgr = np.sign(Y_train_res_xgb_rgr)
matching = np.sum(Y_sign_train_rf_rgr==Y_sign_train_xgb_rgr)
matching

matching / np.len(Y_sign_train_rf_rgr)

Y_sign_test_rf_rgr = np.sign(Y_test_res_rf_rgr)
Y_sign_test_xgb_rgr = np.sign(Y_test_res_xgb_rgr)
matching = np.sum(Y_sign_test_rf_rgr==Y_sign_test_xgb_rgr)
matching

matching / np.len(Y_sign_test_xgb_rgr)


# Try to predict the sign of the residual to see if we have can learn any
#  pattern to see if a host is under- or over-priced

# Decide which columns we want to fit the residuals on
sign_model_cols = []
X_sign_train = X_train.loc(sign_model_cols, axis=1)
X_sign_test = X_test.loc(sign_model_cols, axis=1)

# Decide which price model to use, or averaging
Y_train_res = np.averaage(Y_train_res_rf_rgr, Y_train_res_xgb_rgr)
Y_test_res = np.averaage(Y_test_res_rf_rgr, Y_test_res_xgb_rgr)
Y_sign_train = np.sign(Y_train_res)
Y_sign_test = np.sign(Y_test_res)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classfr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=200))])
lr_params = {'lrc__penalty': ['l1', 'l2'],  # default='l2'
             'lrc__C': [0.1, 0.3, 1, 2]  # default=1
            }
lr_res_sign_classfr = GridSearchCV(lr_classfr, lr_params, cv=10,
                                   scoring='accuracy',
                                   return_train_score=True,
                                   n_jobs=n_cores)

lr_res_sign_classfr.fit(X_sign_train, Y_sign_train)

# Save models to serialized files
out_file_name = os.path.join(path, 'lr_res_sign_classfr.pickle')
with open(out_file_name, 'wb') as f:
    pickle.dump(lr_res_sign_classfr, f)

# Load models
in_file_name = os.path.join(path, 'lr_res_sign_classfr.pickle')
with open(in_file_name, 'rb') as f:
    lr_res_sign_classfr = pickle.load(f)

# Accuracies on training and testing datasets
lr_res_sign_classfr.score(X_sign_train)

lr_res_sign_classfr.score(X_sign_test)

# Prediction on price residual sign and then access its performance (test set)
Y_pred_sign_lr = lr_res_sign_classfr.predict(X_sign_test)


# Support Vector Machine
from sklearn.svm import SVC
svc_classfr = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(gamma='scale', max_iter=500))])
svc_params = {'svc__C': [0.1, 0.3, 1, 2],  # default=1
              'svc__kernel': ['linear', 'rbf']  # default='rbf'
             }
svc_res_sign_classfr = GridSearchCV(svc_classfr, svc_params, cv=10,
                                    scoring='accuracy',
                                    return_train_score=True,
                                    n_jobs=n_cores)

svc_res_sign_classfr.fit(X_sign_train, Y_sign_train)

# Save models to serialized files
out_file_name = os.path.join(path, 'svc_res_sign_classfr.pickle')
with open(out_file_name, 'wb') as f:
    pickle.dump(svc_res_sign_classfr, f)

# Load models
in_file_name = os.path.join(path, 'svc_res_sign_classfr.pickle')
with open(in_file_name, 'rb') as f:
    svc_res_sign_classfr = pickle.load(f)

# Accuracies on training and testing datasets
svc_res_sign_classfr.score(X_sign_train)

svc_res_sign_classfr.score(X_sign_test)

# Prediction on price residual sign and then access its performance (test set)
Y_pred_sign_svc = svc_res_sign_classfr.predict(X_sign_test)


# Gradient Boosting
from xgboost import XGBClassifier
xgb_classfr = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(gamma='scale', max_iter=500))])
xgbc_params = {'xgb__max_depth': [4, 5, 6, 7],  # default=6
               'xgb__learning_rate': [0.1, 0.3, 0.5]  # default=0.3
              }
xgb_res_sign_classfr = GridSearchCV(xgb_classfr, xgbc_params, cv=10,
                                    scoring='accuracy',
                                    return_train_score=True,
                                    n_jobs=n_cores)

xgb_res_sign_classfr.fit(X_sign_train, Y_sign_train)

# Save models to serialized files
out_file_name = os.path.join(path, 'xgb_res_sign_classfr.pickle')
with open(out_file_name, 'wb') as f:
    pickle.dump(xgb_res_sign_classfr, f)

# Load models
in_file_name = os.path.join(path, 'xgb_res_sign_classfr.pickle')
with open(in_file_name, 'rb') as f:
    xgb_res_sign_classfr = pickle.load(f)

# Accuracies on training and testing datasets
xgb_res_sign_classfr.score(X_sign_train)

xgb_res_sign_classfr.score(X_sign_test)

# Prediction on price residual sign and then access its performance (test set)
Y_pred_sign_xgb = xgb_res_sign_classfr.predict(X_sign_test)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classfr = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(gamma='scale', max_iter=500))])
rfc_params = {'rf__n_estimators': [10, 50, 100],  # default=10, future=100
              'rf__min_samples_split': [2, 3, 4]  # default=2
             }
rf_res_sign_classfr = GridSearchCV(rf_classfr, rfc_params, cv=10,
                                   scoring='accuracy',
                                   return_train_score=True,
                                   n_jobs=n_cores)

rf_res_sign_classfr.fit(X_sign_train, Y_sign_train)

# Save models to serialized files
out_file_name = os.path.join(path, 'rf_res_sign_classfr.pickle')
with open(out_file_name, 'wb') as f:
    pickle.dump(rf_res_sign_classfr, f)

# Load models
in_file_name = os.path.join(path, 'rf_res_sign_classfr.pickle')
with open(in_file_name, 'rb') as f:
    rf_res_sign_classfr = pickle.load(f)

# Accuracies on training and testing datasets
rf_res_sign_classfr.score(X_sign_train)

rf_res_sign_classfr.score(X_sign_test)

# Prediction on price residual sign and then access its performance (test set)
Y_pred_sign_rf = rf_res_sign_classfr.score(X_sign_test)


# Construct ROC curves for the various tuned methods
from sklearn.metrics import roc_curve, auc
pred_list = [Y_pred_sign_lr, Y_pred_sign_svc, Y_pred_sign_xgb, Y_pred_sign_rf]
methods = ['Logistic', 'SVM', 'GradBoost', 'RandForest']
plt.figure(size=(9, 9))
for pred_sign, method in zip(pred_list, methods):
    false_pos, true_pos = roc_curve(Y_sign_test, pred_sign)
    roc_auc = auc(false_pos, true_pos)
    plt.plot(false_pos, true_pos, label=method + '(Area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for predicted under- vs over-priced')
plt.legend(loc='lower right')
plt.show()
