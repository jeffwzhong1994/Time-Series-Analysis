#Import Packages
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = pd.read_csv('./international-airline-passengers.csv',usecols = [1], engine = 'python', skipfooter = 3)

#X is the # of passengers at a given time (t) and Y is the # of passengers at the next time (t+1)
dataset = dataframe.values
dataset = dataset.astype('float32')
plt.plot(dataset)
plt.show()

#Covert an array of value into a dataset matrix

def create_dataset(dataset, look_back = 1):
    X, Y = [], []
    for i in range(len(dataset) - look_back-1):
        a = dataset [i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
        return numpy.array(X), numpy.array(Y)
    
# Fix random seeds and normalize the dataset
numpy.random.seed(7)
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

#Split data into training and testing set:
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0: train_size,:], dataset[train_size:len(dataset),:]

#Prepare train and test set for modeling:
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#Reshape it
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Fit LSTM Network:
Model = Sequential()
Model.add(LSTM(4, input_shape = (1, look_back)))
Model.add(Dense(1))
Model.compile(loss = 'mean_squared_error', optimizer = 'adam')
Model.fit(trainX, trainY, epochs = 100, batch_size = 1, verbose = 2)
trainPre = Model.predict(trainX)
testPre = Model.predict(testX)

# Inverse Transform:
trainPre = scaler.inverse_transform(trainPre)
trainY = scaler.inverse_transform([trainY])
testPre = scaler.inverse_transform(testPre)
testY = scaler.inverse_transform([testY])

#Print out Train & Test Score:
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPre[:,0]))
print('Train Score: %.2f RMSE'%(trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPre[:,0]))
print('Test Score: %.2f RMSE'%(testScore))

#Shift Train prediction for plotting:
trainPrePlot = numpy.empty_like(dataset)
trainPrePlot[:,:] = numpy.nan
trainPrePlot[look_back: len(trainPre)+look_back, :] = trainPre

#Shift Test prediction for plotting:
testPrePlot = numpy.empty_like(dataset)
testPrePlot[:,:] = numpy.nan
testPrePlot[len(trainPre)+(look_back*2)+1 : len(dataset)-1, :] = testPre

#plotting:
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPrePlot)
plt.plot(testPrePlot)
plt.show()
