from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
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
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dropout
from keras.layers import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.models import load_model
from keras.layers import Reshape
from keras.utils import plot_model

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


X = np.load('X.npy')
Y = np.load('Y.npy')



# Y = to_categorical(Y)



# train_X = np.random.shuffle(train_X)
# test_X = np.random.shuffle(test_X)
X = X.astype('float32') / 255.        # minmax_normalized


print(X.shape)
print(Y.shape)


# transfer
subcarrier_num = 90
lenth_sample = 750  #750


X = X.reshape(
    X.shape[0],
    lenth_sample,
    subcarrier_num,
    1)  # (4320, 750, 90, 1)



print(X.shape)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

train_Y = to_categorical(train_Y)
test_Y = to_categorical(test_Y)

print('data prepare over!')

num_classes = 4
# ==============================================================
# build CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(lenth_sample, subcarrier_num, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(64, (5, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# ===============================================================


model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])

model.summary()
# plot_model(model, to_file='modelcnn.png', show_shapes=True)


# filepath = r'D:\data\1'
# MODEL_PATH = r'D:\data\2'
# tensorboard = TensorBoard(log_dir=filepath)
# import os
# filepath1 = os.path.join(MODEL_PATH,'model.h5')
# if not os.path.exists(MODEL_PATH): #判断是否存在
#     os.makedirs(MODEL_PATH) #不存在则创建
# checkpoint = ModelCheckpoint(filepath=filepath1,monitor='val_acc',mode='auto' ,verbose=1, save_best_only=False, save_weights_only=False)
# callback_lists=[tensorboard,checkpoint]
# # callback_lists=[checkpoint]
#
# history = model.fit(train_X, train_Y, epochs=25, batch_size=20,
# validation_data=(test_X,
# test_Y),verbose=2,callbacks=callback_lists)#batch_size=20

history = model.fit(
    train_X,
    train_Y,
    epochs=20,
    batch_size=20,
    validation_data=(
        test_X,
        test_Y),
    verbose=2,
    shuffle=True)  # batch_size=20
#
# loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
#
# print('\ntest loss', loss)
# print('accuracy', accuracy)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 16,
         }

# plot history
fig, ax = plt.subplots()
plt.plot(history.history['loss'], '-r', label='train_loss', linewidth=5.0)
plt.plot(
    history.history['val_loss'],
    'b-.',
    label='validation_loss',
    linewidth=5.0)
# plt.title('')
plt.xlabel('epoch', font1)
plt.ylabel('loss', font1)
plt.legend(prop=font1)
fig.savefig('wb_cnn_paper18_1_loss.svg', dpi=600, format='svg')
fig.savefig('wb_cnn_paper18_1_loss.pdf', dpi=600, format='pdf')
plt.show()
# # 设置刻度字体大小
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# # 设置坐标标签字体大小
# ax.set_xlabel(..., fontsize=20)
# ax.set_ylabel(..., fontsize=20)
# # 设置图例字体大小
# ax.legend(..., fontsize=20)


fig, ax = plt.subplots()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('epoch', font1)
plt.ylabel('accuracy', font1)
plt.legend(prop=font1)
fig.savefig('wb_cnn_paper18_1_accuracy.svg', dpi=600, format='svg')
fig.savefig('wb_cnn_paper18_1_accuracy.pdf', dpi=600, format='pdf')
plt.show()
