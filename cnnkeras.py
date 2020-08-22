import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dropout
from keras.layers import BatchNormalization

from keras.layers import Conv2D,MaxPooling2D,Flatten
from keras.models import Model
from keras.models import load_model
from keras.layers import Reshape


train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')
# test2_X = np.load('test2_X.npy')
# test2_Y = np.load('test2_Y.npy')

train_Y = to_categorical(train_Y)
test_Y = to_categorical(test_Y)

train_X = train_X.astype('float32') / 255.        # minmax_normalized
test_X = test_X.astype('float32') / 255.         # minmax_normalized

print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)
#transfer
subcarrier_num = 90
lenth_sample = 750#784#750



train_X = train_X.reshape(train_X.shape[0],lenth_sample,subcarrier_num,1)#(36个位置混合，(4320, 750, 90, 1))
test_X = test_X.reshape(test_X.shape[0],lenth_sample,subcarrier_num,1)#(36个位置混合，(2880, 750, 90, 1))

print(train_X.shape)
print(test_X.shape)

print('data prepare over!')

num_classes =5

#==============================================================
# build CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(lenth_sample,subcarrier_num,1)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
model.add(Conv2D(64, (5, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#===============================================================


model.add(Dense(num_classes, activation='softmax') )
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

model.summary()
#keras.utils.plot_model(model)

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
filepath = r'C:\Users\XueDing\Desktop\cc'
filepath1 = r'C:\Users\XueDing\Desktop\c'
tensorboard = TensorBoard(log_dir=filepath)
checkpoint = ModelCheckpoint(filepath=filepath1,monitor='val_acc',mode='auto' ,verbose=1, save_best_only=False, save_weights_only=False)
callback_lists=[tensorboard,checkpoint]

# keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
history = model.fit(train_X, train_Y, epochs=25, batch_size=20, validation_data=(test_X, test_Y),verbose=2,callbacks=callback_lists)#batch_size=20
# print(history)
# print(history.history)


# model.save('./saved_model/cnn_model.h5')
# model.save_weights("./saved_model/model_weights.h5")


#===============================

# from keras.callbacks import ModelCheckpoint
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#
# checkpointer = ModelCheckpoint(filepath='model.best.hdf5',
#                                verbose=1, save_best_only=True)
# model.fit(train_X, train_Y, batch_size=50, epochs=50,
#           validation_split=0.2, callbacks=[checkpointer],
#           verbose=2, shuffle=True)

#======================================================================

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

# plot history
fig,ax = plt.subplots()
plt.plot(history.history['loss'], '-r',label='train_loss',linewidth=5.0)
plt.plot(history.history['val_loss'],'b-.', label='validation_loss',linewidth=5.0)
# plt.title('')
plt.xlabel('epoch',font1)
plt.ylabel('loss',font1)
plt.legend(prop=font1)
fig.savefig('wb_cnn_paper18_1_loss.svg',dpi=600,format='svg')
fig.savefig('wb_cnn_paper18_1_loss.pdf',dpi=600,format='pdf')
plt.show()
# # 设置刻度字体大小
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# # 设置坐标标签字体大小
# ax.set_xlabel(..., fontsize=20)
# ax.set_ylabel(..., fontsize=20)
# # 设置图例字体大小
# ax.legend(..., fontsize=20)



fig,ax = plt.subplots()
plt.plot(history.history['acc'], label='train_accuracy')
plt.plot(history.history['val_acc'], label='validation_accuracy')
plt.xlabel('epoch',font1)
plt.ylabel('accuracy',font1)
plt.legend(prop=font1)
fig.savefig('wb_cnn_paper18_1_accuracy.svg',dpi=600,format='svg')
fig.savefig('wb_cnn_paper18_1_accuracy.pdf',dpi=600,format='pdf')
plt.show()

prediction = model.predict(test_X)#不应该用这个，这个相当于把验证集用最新保存的模型跑一遍，应该和最后一个epoch效果一样

#print(prediction)#prob
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def  onehot2onedim(a):
    row_num = a.shape[0]
    b = a.argmax(axis=1)
    return b

#========================================
# from sklearn import preprocessing
# enc = preprocessing.OneHotEncoder()
# # 训练onehot编码，指定标签
# enc.fit([[0],[1],[2],[3]])
# # 将标签转换成 onehot编码
# result =enc.transform([[0],[1],[2],[3]])
# print(result.toarray())

# sortmax 结果转 onehot
def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b
# sortmax 结果转 onehot
# a = [[0.2,0.3,0.5],
#      [0.7,0.3,0.5],
#      [0.7,0.9,0.5]
#     ]
#print(enc.inverse_transform(props_to_onehot(a)))
#AttributeError: 'OneHotEncoder' object has no attribute 'inverse_transform'
#===============================================
prediction = props_to_onehot(prediction)
prediction = onehot2onedim(prediction)
test_Y = onehot2onedim(test_Y)

accuracy_score = accuracy_score(test_Y, prediction)
classification_report = classification_report(test_Y, prediction, labels=None, target_names=None, sample_weight=None, digits=2)
confusion_matrix = confusion_matrix(test_Y, prediction, labels=None, sample_weight=None)

print('accuracy_score =',accuracy_score)
print('classification_report = ', classification_report)
print('confusion_matrix = ', confusion_matrix)


#========================================================
test2_X = np.load('test2_X.npy')
test2_Y = np.load('test2_Y.npy')

test2_Y = to_categorical(test2_Y)
test2_X = test2_X.astype('float32') / 255.         # minmax_normalized
test2_X = test2_X.reshape(test2_X.shape[0],lenth_sample,subcarrier_num,1)

model = load_model('./saved_model/cnn_model.h5')
scores = model.evaluate(test2_X, test2_Y)#这个是测试集的结果
print('accuracy = ', scores[1])


model1 = load_model(r'C:\Users\XueDing\Desktop\c')
scores1 = model1.evaluate(test2_X, test2_Y)#这个是测试集的结果
print('accuracy = ', scores1[1])

#===============================
# prediction2 = model.predict(test2_X)
# def plot_confusion_matrix(cm, labels_name, title):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
#     plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
#     plt.title(title)    # 图像标题
#     plt.colorbar()
#     num_local = np.array(range(len(labels_name)))
#     plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
#     plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# cm = confusion_matrix(np.argmax(test_Y, 1), np.argmax(prediction2, 1))
# # cm = confusion_matrix(np.argmax(test2_X, 1), prediction2,)
# print(cm)
# labels_name = ['1','2','3','4','5','6']
# plot_confusion_matrix(cm, labels_name, "Confusion Matrix")
# plt.savefig('cm_12c36.svg',dpi=600,format='svg')
# # plt.savefig('/HAR_cm.png', format='png')
# plt.show()
