import numpy as np
import pandas as pd
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
# Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
# kernels
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

'''Gaussian Process regression with RBF and Matern kernels'''


# load relevant battery dataset for training the algorithm

data_train = pd.read_csv('data_train_fsed.csv', index_col=0)

X_train = np.array(data_train.drop(['Discharge_Q', 'Group'], axis=1))
y_train = np.array(data_train['Discharge_Q'])


'''Gaussian Process Model Initiation'''

kernels = [
    1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 200.0)),  # 扩大范围
    1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-3, 1e6)),  # 调整alpha范围
    1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                        length_scale_bounds=(0.01, 50.0),  # 扩大范围
                        periodicity_bounds=(0.5, 50.0)),  # 扩大范围
    ConstantKernel(0.1, (1e-6, 100.0))  # 调整下界
    * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.01, 100.0)) ** 2),
    1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 200.0), nu=1.5)  # 扩大范围
]


gp = GaussianProcessRegressor(optimizer='fmin_l_bfgs_b',  n_restarts_optimizer=0, normalize_y=True, copy_X_train=False) # n_restarts used to be 9

pipeline = Pipeline(

    [
        ('scl', StandardScaler()),
        ('clf', gp)
    ]
)

param = {"clf__kernel": kernels,
         "clf__alpha": np.round(np.random.uniform(0.0001, 30, 1000), 2),
        }


groups = data_train.Group
no_of_splits = len(np.unique(groups))  # number of slits is equal to the number of groups
group_kfold = GroupKFold(n_splits=no_of_splits)
model = RandomizedSearchCV(pipeline, param_distributions=param, cv=group_kfold, n_iter=10, verbose=5)
# fit model
model.fit(X_train, y_train, groups=groups)  # 添加groups参数

# save the model to disk
pickle.dump(model, open('GPR_model.sav', 'wb'))
