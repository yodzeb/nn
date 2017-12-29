import matplotlib.pyplot as plt
import csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv2D, Reshape, MaxPooling2D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
import gdax
import time

i=0
real_values=[]
buffer_size=300
public_client = gdax.PublicClient()

def push (real_values, value):
    global buffer_size
    real_values.append([value])
    if (len(real_values) > buffer_size):
        real_values = real_values[1:]
    return real_values

def get_value(label):
    o_b = public_client.get_product_order_book(label)
    return float(o_b["bids"][0][0])


def get_nn():
    look_back = 1
    filters=32
    kernel_size=(2,2)
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(40, input_shape=(1, look_back)))
    model.add(Reshape((1,2,-1)))
    model.add(Conv2D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 data_format='channels_first'
                 ))
    model.add(MaxPooling2D(pool_size=(1, 1), data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
	a = dataset[i:(i+look_back), 0]
	dataX.append(a)
	dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
    

model = get_nn()

z=0
old=0

while True:
    z+=1
    i = i+1
    new_val = get_value("BTC-USD")
    st_value=str(new_val)
    if old<new_val:
        st_value+="+"
        color="green"
    elif old == new_val:
        st_value+="="
        color="black"
    else:
        st_value+="-"
        color="red"
    old = new_val
    real_values = push(real_values, new_val)
    #time.sleep(1);

    print str(new_val)+" ->> ",
    if (len(real_values) < 6):
        print "Too small"
        continue
    
    # Norm data
    dataset = real_values
    #print dataset
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    #test_size = len(dataset) - train_size
    #train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train = dataset
    train2 = dataset

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    #testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    d_test = np.reshape(trainY, (trainY.shape[0], 1, 1))#trainY.shape[1]))
    #testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    if (i%10 == 0):
        print "fitting.."+str(i)+"..",
        model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=0)
        print "done."

    print "("+str(len(real_values))+"predict:",
    d_test = np.reshape(train2, (train2.shape[0], 1, 1))

    ydata=[]
    xdata=[]

    vision=80
    
    plt.clf()
    fig1=plt.figure(1)
    fig1.text(0.45, 0.95, st_value, ha="center", va="bottom", size="large",color=color)
    #plt.title(st_value)
    axes = plt.gca()
    #axes.set_xlim(0-1, 40+len(dataset))
    #axes.set_ylim(0-1, 20000)
    line, = axes.plot(xdata, ydata, 'r-')

    #fig = plt.figure()
    l_min = min(len(dataset),vision)    
    for j in range (l_min):
        ydata.append(scaler.inverse_transform(dataset[len(dataset)-l_min+j][0]))
        xdata.append(j)
        
    for j in range(vision):
        #print "d:",
        #print d_test
        trainPredict = model.predict(d_test)
        d_test = d_test[1:] # = np.reshape(trainPredict, (trainPredict.shape[0], 1, 1))
        # dirty boy
        d_test = np.append(d_test, [trainPredict[len(trainPredict)-1]])
        d_test = np.reshape(d_test, (d_test.shape[0], 1, 1))
        #print scaler.inverse_transform(trainPredict)
        #print str(len(trainPredict)-1)
        #print trainPredict
        future = scaler.inverse_transform(trainPredict[len(trainPredict)-1][0])
        print future[0][0],
        xdata.append(vision+j)
        ydata.append(future[0][0])
    print ""        

    line.set_xdata(xdata)
    line.set_ydata(ydata)
    axes.relim()
    axes.autoscale_view()
    
    plt.draw()
    plt.pause(1e-17)
    #time.sleep(0.8)
