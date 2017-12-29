import sys
import gdax
import time
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv

public_client = gdax.PublicClient()
buffer_size   = 900
real_values   = []

def get_value(label):
    o_b = public_client.get_product_order_book(label)
    return float(o_b["bids"][0][0])

def push (real_values, value):
    global buffer_size
    real_values.append(value)
    if (len(real_values) > buffer_size):
        real_values = real_values[1:]
    return real_values

def get_nn():
    look_back = 1
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(20, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

i=0;
while True:
    new_val = get_value("LTC-USD")
    i+=1
    real_values = push(real_values, new_val)
    time.sleep(1);
    print str(i)+"-"+str(new_val)
    if (i == buffer_size):
        with open('test_file.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in real_values]
        sys.exit(0)
    
    #model.fit(trainX, trainY, epochs=20, batch_size=1)
    
