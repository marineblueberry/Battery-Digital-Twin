from sklearn.feature_selection import RFECV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import pickle
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint, expon
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
import forestci as fci
from matplotlib import gridspec

#是否使用对抗性样本（其作用是欺骗机器学习模型让其做出错误预测，
# 对抗性学习则是一种训练方法，旨在使模型在面对对抗性样本时更加鲁棒。
adversarial = False

data_fs = pd.read_csv('dataset_for_feature_selections.csv', index_col=0)
data_training = pd.read_csv('dataset_training.csv', index_col=0)
data_test = pd.read_csv('dataset_for_test.csv', index_col=0)
data_calibration = pd.read_csv('dataset_for_calibration.csv', index_col=0)  # Added index_col=0

feature_to_drop = ['Discharge_Q', 'SOH_discharge_capacity', 'Group']
feature_to_predict = 'Discharge_Q'

X_train = data_fs.drop(feature_to_drop, axis=1)
y_train = data_fs[feature_to_predict]

no_of_features = 1 #每次迭代要删除的特征数量

#建立随机森林回归模型  决策树数量（通常是特征数10倍）   是否进行抽样训练    控制并行计算的核心数（-1为所有可用核心）
rf_tuning = RandomForestRegressor(n_estimators=200, bootstrap=True, n_jobs=-1) 
#一些超参数
param = {"max_depth": sp_randint(15, 25), #15-25, 5-10
         "max_features": [no_of_features], #[no_of_features],  # sp_randint(2, 4),
         "min_samples_split": sp_randint(2, 5),
         "min_samples_leaf": sp_randint(5, 15),
         "criterion": ['squared_error']}


no_of_splits = len(np.unique(data_fs.Group))  # number of slits is equal to the number of groups
groups = data_fs.Group
group_kfold = GroupKFold(n_splits=no_of_splits)  # inner test and train using the group KFold

model = RandomizedSearchCV(rf_tuning, param_distributions=param, cv=group_kfold, n_iter=100, # full_dataset: 150
                            refit=True, verbose=1)
model.fit(X_train, y_train, groups=groups)
RF_f_selection_model = model.best_estimator_
RF_f_selection_model_param = model.best_params_

names = list(data_fs.drop(['Discharge_Q', 'SOH_discharge_capacity', 'Group'], axis=1))
rf = RF_f_selection_model#把最优模型赋值给rf
rfe = RFECV(estimator=rf, min_features_to_select=3, cv=group_kfold, step=1,
            scoring='neg_mean_squared_error', verbose=1)  # neg_mean_squared_error, r2

selector_RF = rfe.fit(X_train, y_train, groups=groups)

ranking_features = sorted(zip(map(lambda x: round(x, 4), selector_RF.ranking_), names), reverse=False)
optimum_no_feature = selector_RF.n_features_

print(optimum_no_feature)

x = range(no_of_features, len(selector_RF.cv_results_['mean_test_score']) + no_of_features)
y = selector_RF.cv_results_['mean_test_score']

'''feature selection resuts'''
print('Feature rank: \n {}'.format(ranking_features))
# Plot number of features VS. cross-validation scores
f = plt.figure(figsize=(7, 5))
plt.plot(x, y, 'o--', color='tab:orange')
plt.plot(x[np.argmax(y)], np.max(y), 'v', markersize=15, color='k')
# plt.title('Optimum number of features based RF-RFE using neg-mse is: {}'.format(optimumum_no_feature))
plt.xlabel('Number of features selected', fontsize=15)
plt.ylabel('Cross-validation score [Negative MSE]', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(False)
plt.show()



# 获取被选中的特征名称
selected_features = [name for rank, name in ranking_features if rank == 1]

# 根据结果删除无用特征
X_training_data_opt_fet = pd.DataFrame(selector_RF.transform(data_training.drop(feature_to_drop, axis=1)), columns=selected_features)
X_test_data_opt_fet = pd.DataFrame(selector_RF.transform(data_test.drop(feature_to_drop, axis=1)), columns=selected_features)
X_calibration_data_opt_fet = pd.DataFrame(selector_RF.transform(data_calibration.drop(feature_to_drop, axis=1)), columns=selected_features)

'''把分组信息添加到数据集中'''
X_training_data_opt_fet['Group'] = np.array(data_training.Group)
X_test_data_opt_fet['Group'] = np.array(data_test.Group)
X_calibration_data_opt_fet['Group'] = np.array(data_calibration.Group)

y_train = data_training[feature_to_predict]
y_test = data_test[feature_to_predict]
y_calibration = data_calibration[feature_to_predict]


print('Total number of features selected: ',optimum_no_feature)
print('\n Ranked features: {}'.format(ranking_features))

# ''' save "regularised" datasets (features selected based on RF-RFE unsupervised) '''

data_test_fs = pd.concat([X_test_data_opt_fet, pd.DataFrame(y_test)], axis=1)
data_train_fs = pd.concat([X_training_data_opt_fet, pd.DataFrame(y_train)], axis=1)
data_calibration_fs = pd.concat([X_calibration_data_opt_fet, pd.DataFrame(y_calibration)], axis=1)


# save the data after feature selection for further used in the pipeline
data_train_fs.to_csv('data_train_fsed.csv')
data_test_fs.to_csv('data_test_fsed.csv')
data_calibration_fs.to_csv('data_calibration_fsed.csv')
