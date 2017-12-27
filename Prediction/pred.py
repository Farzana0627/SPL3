import numpy
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras import optimizers
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json
from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
def saveModel(model,file='modelrplpwaveletmodified'):
	# save the model

	# serialize model to JSON
	model_json = model.to_json()
	with open(file+'Param', "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(file+'Weights')

def loadModel(file='model1'):

	# import the neural network model

	# load json and create model
	json_file = open(file+'Param', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	# load weights into new model
	model.load_weights(file+'Weights')
	lrate = 0.01
	decay = lrate/25
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

	#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #optimizer=optimizers.Adadelta()/optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='mean_squared_error'/ 
	#mean_absolute_error/mean_squared_logarithmic_error/sparse_categorical_crossentropy/categorical_crossentropy
	#print(model.summary())

	return model



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#read files
#rootDir = 'RPLPWavelet/'
# rootDir = 'Data/Fernando/'
# trainsetDir = rootDir + 'PLPTrainResultFolder'
# testsetDir =  rootDir + 'PLPTestResultFolder'

# trainsetDir = rootDir + 'RPLPTrainResultFolder'
# testsetDir =  rootDir + 'RPLPTestResultFolder'

# trainsetDir = rootDir + 'RPLPWaveletModifiedTrainResultFolder'
# testsetDir =  rootDir + 'RPLPWaveletModifiedTestResultFolder'

# trainsetDir = 'RPLPWaveletModified3TrainResultFolder'
testsetDir =  'RPLPWaveletModified3TestResultFolder'

Ncoef=39
Nframes=293 #plp=520 #others=353
Nclass=10

# TotalTrainfiles=1000
# X_train=numpy.empty((TotalTrainfiles,Ncoef,Nframes))
# y_train =  numpy.zeros((TotalTrainfiles))
# index=0
# folderpaths=[x[0] for x in os.walk(trainsetDir)]
# del folderpaths[0]
# for folderpath in folderpaths:
# 	filepaths = os.listdir(folderpath)
# 	for filepath in filepaths:
# 		digit= int(filepath[0]);
# 		filepath = os.path.join(folderpath, filepath)
# 		with open(filepath) as file:
# 			array2d = [[value for value in line.strip().split('\t')] for line in file]
# 			array2d= numpy.array(array2d)
		
# 		X_train[index] = array2d
# 		y_train[index]= digit
# 		index+=1

TotalTestfiles=100
X_test = numpy.empty((TotalTestfiles,Ncoef,Nframes))
y_test = numpy.zeros((TotalTestfiles))

index=0
folderpaths=[x[0] for x in os.walk(testsetDir)]
del folderpaths[0]
for folderpath in folderpaths:
	filepaths = os.listdir(folderpath)
	for filepath in filepaths:
		digit= int(filepath[0]);		
		filepath = os.path.join(folderpath, filepath)
		with open(filepath) as file:
			array2d = [[value for value in line.strip().split('\t')] for line in file]
			array2d= numpy.array(array2d)	
		X_test[index] = array2d
		y_test[index]= digit
		index+=1


# X_train = X_train.astype('float64')
X_test = X_test.astype('float64')



# X=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
# y = y_train
# print(X.shape)
# print(y.shape)





# dict={0:150, 1 : 150, 2 : 150, 3 : 150, 4 : 150, 5 : 150, 6 : 150, 7 : 150, 8:150, 9:150 }
# #X, y = SMOTE(ratio='auto', random_state=42, k=None, k_neighbors=5, m=None, m_neighbors=10, out_step=0.5, kind='regular', svm_estimator=None, n_jobs=1).fit_sample(X, y_train)
# sm = SMOTE(ratio={0:1000, 1 : 1000, 2 : 1000, 3 : 1000, 4 : 1000, 5 : 1000, 6 : 1000, 7 : 1000, 8:1000, 9:1000},random_state=21)
# X, y = sm.fit_sample(X, y)
# print(X.shape)
# print(y.shape)

# # y_test = y_test.astype('float32')
# # one hot encode outputs
# X_train=X.reshape(10000,X_train.shape[1],X_train.shape[2])
# y_train=y;
# print(X_train[0])
# print(X_train[1000])
# print(y_train.shape)


# y_train = y_train.astype('float32')
# y_test = y_test.astype('float32')
# one hot encode outputs
# y_train = np_utils.to_categorical(y_train,num_classes=10)
# y_test = np_utils.to_categorical(y_test,num_classes=10)
# num_classes = y_test.shape[1]


X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)


#Compile model
model=loadModel('modelrplpwaveletmodified')

# calculate predictions
predictions = model.predict(X_test[0:9])
rounded = [round(x[0]) for x in predictions]
print('zero',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[10:19])
rounded = [round(x[1]) for x in predictions]
print('one',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[20:29])
rounded = [round(x[2]) for x in predictions]
print('two',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[30:39])
rounded = [round(x[3]) for x in predictions]
print('three',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[40:49])
rounded = [round(x[4]) for x in predictions]
print('four',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[50:59])
rounded = [round(x[5]) for x in predictions]
print('five',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[60:69])
rounded = [round(x[6]) for x in predictions]
print('six',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[70:79])
rounded = [round(x[7]) for x in predictions]
print('seven',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[80:89])
rounded = [round(x[8]) for x in predictions]
print('eight',rounded)
print (rounded.count(1), 'out of 10')

predictions = model.predict(X_test[90:99])
rounded = [round(x[9]) for x in predictions]
print('nine',rounded)
print (rounded.count(1), 'out of 10')



# rounded = [round(x[0]) for x in predictions]
# print('zero',rounded)

#classes = model.predict(X_test, batch_size=16)
# batch_size = 50
# batch_num = int(mnist.test.num_examples / batch_size)
# test_accuracy = 0
    

