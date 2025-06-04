import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

data_train = pd.read_csv('data_train_fsed.csv', index_col=0)
X_train = data_train.drop(['Discharge_Q', 'Group'], axis=1)
y_train = data_train['Discharge_Q']

#超参数优化过程
pipeline = Pipeline(
                        [
                            ('scl', StandardScaler()),
                            ('clf', linear_model.BayesianRidge(fit_intercept=True))  # 移除 normalize 参数
                        ]
                    )
#设置超参数范围
param = {"clf__alpha_1": np.round(np.random.uniform(-0.01, 1000, 100), 2),
         "clf__alpha_2": np.round(np.random.uniform(-0.01, 1000, 100), 2),
         "clf__lambda_1": np.round(np.random.uniform(-0.01, 1000, 100), 2),
         "clf__lambda_2": np.round(np.random.uniform(-0.01, 1000, 100), 2),
        }

no_of_splits = len(np.unique(data_train.Group))  # number of slits is equal to the number of groups
groups = data_train.Group
group_kfold = GroupKFold(n_splits=no_of_splits)
model = RandomizedSearchCV(pipeline, param_distributions=param, cv=group_kfold, n_iter=10, verbose=10)  # 移除 iid 参数
# fit model
model.fit(X_train, y_train, groups=groups)
# save the model to disk
pickle.dump(model, open('bayesian_ridge_model.sav', 'wb'))