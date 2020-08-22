import numpy as np
import matplotlib.pyplot as plt

feature_1=np.load('D:/data/SVM_data/O-WB-01-angle-svm.npy')
feature_2=np.load('D:/data/SVM_data/PO-WB-01-angle-svm.npy')
feature_3=np.load('D:/data/SVM_data/UP-WB-01-angle-svm.npy')
feature_4=np.load('D:/data/SVM_data/X-WB-01-angle-svm.npy')


# plt.scatter(feature_1[:,0], feature_1[:,14], marker='o')
# plt.scatter(feature_2[:,0], feature_2[:,14], marker='v')
# plt.show()

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


train_num = 41

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
from sklearn.model_selection import GridSearchCV

#train_X=np.array([[1],[2]])
#train_Y=np.array([1,2])
#test_X=np.array([[2],[3]])
#test_Y=np.array([1,2])

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
train_X_minmax = min_max_scaler.fit_transform(train_X)
test_X_minmax = min_max_scaler.transform(test_X)

#-------------------------------------
# clf=svm.SVC(kernel='rbf',C=1,gamma=1/3)
#-------------------------------------
# para_c=list(range(1,10,1))
# para_gamma=list(np.linspace(1,10,50))
# para_kernel=['rbf','poly','linear','sigmod','precomputed']
#-------------------------------------------------
parameters={'C':[1,2,3,4,5,6,7,8,9],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2]}
# parameters={'C':para_c,'kernel':['rbf'],'gamma':para_gamma}
svr=svm.SVC()
clf=GridSearchCV(svr,parameters,cv =10)  #网格搜索、交叉验证  10折交叉验证
#clf = GridSearchCV(estimator = svr, param_grid = parameters, scoring = 'accuracy',cv =10)
#print(clf.best_estimator_)
#-------------------------------------
clf.fit(train_X_minmax, train_Y.ravel())

# print clf.score(x_train, y_train)  # 精度
# y_hat = clf.predict(x_train)
# show_accuracy(y_hat, y_train, '训练集')

##################################
#print clf.score(train_X_minmax, train_Y)
y_hat_train = clf.predict(train_X_minmax)
#show_accuracy(y_hat, train_Y, '训练集')
err_svm_train=sum( int(y_hat_train[i]) != train_Y[i] for i in range(len(train_Y))) / float(len(train_Y))
print ('taining predicting, classification error=%f' % (err_svm_train))


#################################
#print clf.score(test_X_minmax, test_Y)
y_hat_test = clf.predict(test_X_minmax)
#show_accuracy(y_hat, test_Y, '测试集')
err_svm_test=sum( int(y_hat_test[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y))
print ('testing predicting, classification error=%f' % (err_svm_test))

print (clf.score(test_X_minmax, test_Y))  #准确率
