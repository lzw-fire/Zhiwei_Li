import numpy as np

feature_1=np.load('feature_data_011.npy')
feature_2=np.load('feature_data_022.npy')
feature_3=np.load('feature_data_033.npy')
feature_4=np.load('feature_data_044.npy')
# feature_5=np.load('feature_data_05.npy')


sample_num = 51

feature_1 = feature_1[0:sample_num, :]
feature_2 = feature_2[0:sample_num, :]
feature_3 = feature_3[0:sample_num, :]
feature_4 = feature_4[0:sample_num, :]
# feature_5 = feature_5[0:sample_num, :]

label_1 = np.ones(len(feature_1)) * 1
label_2 = np.ones(len(feature_2)) * 2
label_3 = np.ones(len(feature_3)) * 3
label_4 = np.ones(len(feature_4)) * 4
# label_5 = np.ones(len(feature_5)) * 5


import random

index = np.arange(sample_num)
random.shuffle(index)

feature_1_index = feature_1[index]
feature_2_index = feature_2[index]
feature_3_index = feature_3[index]
feature_4_index = feature_4[index]
# feature_5_index = feature_5[index]


label_1_index = label_1[index]
label_2_index = label_2[index]
label_3_index = label_3[index]
label_4_index = label_4[index]
# label_5_index = label_5[index]


train_num = 31

# train_X = np.vstack((feature_1_index[0:train_num, :], feature_2_index[0:train_num, :], feature_3_index[0:train_num, :],
#                      feature_4_index[0:train_num, :], feature_5_index[0:train_num, :]))
# train_Y = np.hstack((label_1_index[0:train_num], label_2_index[0:train_num], label_3_index[0:train_num],
#                      label_4_index[0:train_num], label_5_index[0:train_num]))
# test_X = np.vstack((feature_1_index[train_num:sample_num, :], feature_2_index[train_num:sample_num, :],
#                     feature_3_index[train_num:sample_num, :], feature_4_index[train_num:sample_num, :],
#                     feature_5_index[train_num:sample_num, :]))
# test_Y = np.hstack((label_1_index[train_num:sample_num], label_2_index[train_num:sample_num],
#                     label_3_index[train_num:sample_num], label_4_index[train_num:sample_num],
#                     label_5_index[train_num:sample_num]))

train_X = np.vstack((feature_1_index[0:train_num, :], feature_2_index[0:train_num, :], feature_3_index[0:train_num, :],
                     feature_4_index[0:train_num, :]))  #在列上合并
train_Y = np.hstack((label_1_index[0:train_num], label_2_index[0:train_num], label_3_index[0:train_num],
                     label_4_index[0:train_num]))  #在行上合并
test_X = np.vstack((feature_1_index[train_num:sample_num, :], feature_2_index[train_num:sample_num, :],
                    feature_3_index[train_num:sample_num, :], feature_4_index[train_num:sample_num, :]))
test_Y = np.hstack((label_1_index[train_num:sample_num], label_2_index[train_num:sample_num],
                    label_3_index[train_num:sample_num], label_4_index[train_num:sample_num]))

print("Data prepare over!")

#===========================================================================================

from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#train_X=np.array([[1],[2]])
#train_Y=np.array([1,2])
#test_X=np.array([[2],[3]])
#test_Y=np.array([1,2])

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train_X_minmax = min_max_scaler.fit_transform(train_X)
test_X_minmax = min_max_scaler.transform(test_X)



x = []
y = []
z = []
for C in range(1,10,1):
    for gamma in range(1,11,1):
        #参数scoring设置为roc_auc返回的是AUC，cv=5采用的是5折交叉验证
        clf = svm.SVC(C=C, kernel='rbf', gamma=gamma/10)
        clf.fit(train_X_minmax, train_Y.ravel())
        auc = clf.score(test_X_minmax, test_Y)
        # auc = cross_val_score(SVC(C=C,kernel='rbf',gamma=gamma/10),train_X_minmax,train_Y.ravel(),cv=5,scoring='roc_auc').mean()
        x.append(C)
        y.append(gamma/10)
        z.append(auc)



x = np.array(x).reshape(9,10)
y = np.array(y).reshape(9,10)
z = np.array(z).reshape(9,10)


fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(y, x, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.xlabel('Gamma')
plt.ylabel('C')
plt.show()
