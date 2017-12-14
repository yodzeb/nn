import keras
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, Conv1D, GlobalMaxPooling1D

def read_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        x = []
        y = []
        for e in data:
            y.append(e["la"]) # labesl
            x.append(e["wo"])  # words
        return np.asarray(x),np.asarray(y)
        #return x,np.asarray(y)
                 
maxlen = 128
batch_size = 32
#embedding_dims = 100
filters = 16
kernel_size = (10,10)
hidden_dims = 10
epochs = 1

x_train, y_train = read_data("train.pkl")
x_test,  y_test  = read_data("valid.pkl")
#x_train = [x_train]
#x_test = [x_test]


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print str(len(x_test))
print x_train.shape
print str(len(x_test[0]))
#print x_test[6][1]
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)



print('Build model...')
model = Sequential()
model.add(Reshape((1,64,100),
                  #input_dims=1,
                  input_shape=(6400,)))
                  #))
model.add(Conv2D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 data_format='channels_first'
                 #input_shape=(1,32,100)
                 ))
model.add(MaxPooling2D(pool_size=(2, 2),
                       data_format='channels_first'))
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(5,))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          validation_data=(x_test, y_test))
