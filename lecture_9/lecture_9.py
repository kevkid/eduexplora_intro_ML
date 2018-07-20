# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:31:50 2018

@author: kevin
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)
X = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
y = np.array([[0], [1], [1], [0]])

# create model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])
# Fit the model
model.fit(X, y, epochs=2000, batch_size=4)

model.predict(np.array([[0,1]])).round()

#plt.scatter(X[:,0], X[:,1], c=y.flatten())


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = X, y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape).round(),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap = ListedColormap(('red', 'green')))


#classify circles
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

X,y = make_circles(n_samples=1000, noise=0.08)
model = Sequential()
model.add(Dense(4, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['binary_accuracy'])
# Fit the model

fig, ax = plt.subplots(nrows=3, ncols=3)
for row in ax:
    for col in row:
        model.fit(X, y, epochs=100, batch_size=64)        
        X_set, y_set = X, y
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        col.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape).round(),
                     alpha = 0.5, cmap = ListedColormap(('red', 'green')))
        #col.xlim(X1.min(), X1.max())
        #col.ylim(X2.min(), X2.max())
        col.scatter(X[:,0], X[:,1], c=y.flatten(), cmap = ListedColormap(('red', 'green')))
plt.show()


#classify mnist
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Dense(128, activation='tanh', input_shape=(784,)))
#model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=40, batch_size=64)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

