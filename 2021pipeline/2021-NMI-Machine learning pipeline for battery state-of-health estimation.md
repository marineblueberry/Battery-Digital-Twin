# 2021-NMI-Machine learning pipeline for battery state-of-health estimation

## 1. 训练特征选择模型，得到特征选取完的数据集

###### 1.1 目标：从数据集中选出对预测目标变量（Discharge_Q)最有用的特征，减少冗余特征，提高模型泛化能力和预测性能

###### 1.2 数据集：dataset_for_feature_selection.csv

###### 1.3 主要方法：

- 随机森林回归模型（RandomForestRegressor）
- 递归特征消除（RFECV）
- 交叉验证（GroupKFold）

###### 1.4 结果：从20个特征中选出了12个最优特征

![image-20250512101257213](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250512101257213.png)

它们分别为：（代号为1的）

![image-20250512101331262](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250512101331262.png)

然后根据这个结果，对我的数据集

dataset_training.csv

dataset_for_test.csv

data_calibration.csv(校准集)

进行分割（剔除无用特征），得到

data_train_fsed.csv

data_test_fsed.csv

data_calibration_fsed.csv

## 2. 训练三个预测模型

得到模型

RF_model.sav

GPR_model.sav

bayesian_ridge_model.sav

## 3. 绘制不同模型性能曲线

###### 3.1 GPR_model

![image-20250510170143864](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510170143864.png)

![image-20250510170510852](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510170510852.png)

![image-20250510170546536](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510170546536.png)

![image-20250510170609959](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510170609959.png)

![image-20250510170625175](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510170625175.png)

![image-20250510170647128](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510170647128.png)

![image-20250510170709112](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510170709112.png)

###### 3.2 Bayes_Ridge_model

![image-20250510172541943](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510172541943.png)

![image-20250510172600761](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510172600761.png)



![image-20250510172618614](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510172618614.png)

![image-20250510172632270](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510172632270.png)

![image-20250510172646892](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250510172646892.png)

###### 3.3 RF_model

由于模型生成格式与上面两个不同，模型生成成功但在绘制曲线时出现了点问题
