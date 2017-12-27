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

def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv2D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#read files
rootDir = 'Data/'
# rootDir = 'Data/Fernando/'
# trainsetDir = rootDir + 'PLPTrainResultFolder'
# testsetDir =  rootDir + 'PLPTestResultFolder'

# trainsetDir = rootDir + 'RPLPTrainResultFolder'
# testsetDir =  rootDir + 'RPLPTestResultFolder'

# trainsetDir = rootDir + 'RPLPWaveletModifiedTrainResultFolder'
# testsetDir =  rootDir + 'RPLPWaveletModifiedTestResultFolder'

# trainsetDir = rootDir + 'RPLPWaveletModified2TrainResultFolder'
# testsetDir =  rootDir + 'RPLPWaveletModified2TestResultFolder'

trainsetDir = rootDir + 'RPLPWaveletModified3TrainResultFolder'
testsetDir =  rootDir + 'RPLPWaveletModified3TestResultFolder'

Ncoef=39
Nframes=293 #different values for different methods. change the values manually
Nclass=10

TotalTrainfiles=1000
X_train=numpy.empty((TotalTrainfiles,Ncoef,Nframes))
y_train =  numpy.zeros((TotalTrainfiles))
index=0
folderpaths=[x[0] for x in os.walk(trainsetDir)]
del folderpaths[0]
for folderpath in folderpaths:
	filepaths = os.listdir(folderpath)
	for filepath in filepaths:
		digit= int(filepath[0]);
		filepath = os.path.join(folderpath, filepath)
		with open(filepath) as file:
			array2d = [[value for value in line.strip().split('\t')] for line in file]
			array2d= numpy.array(array2d)
		
		X_train[index] = array2d
		y_train[index]= digit
		index+=1

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
print (y_test)


X_train = X_train.astype('float64')
X_test = X_test.astype('float64')



X=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
y = y_train
print(X.shape)
print(y.shape)


sm = SMOTE(ratio={0:1000, 1 : 1000, 2 : 1000, 3 : 1000, 4 : 1000, 5 : 1000, 6 : 1000, 7 : 1000, 8:1000, 9:1000},random_state=21)
X, y = sm.fit_sample(X, y)
print(X.shape)
print(y.shape)


X_train=X.reshape(10000,X_train.shape[1],X_train.shape[2])
y_train=y;

# one hot encode outputs
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
num_classes = y_test.shape[1]

print(X_train.shape)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
print(X_train.shape)

input_shape=(X_train.shape[1],X_train.shape[2],1)




model = Sequential()
model.add(Conv2D(130, (7, 7),  input_shape=(X_train.shape[1],X_train.shape[2],1), padding='same', activation='relu', kernel_constraint=maxnorm(3)))  # use any value from 1-100 and (2,2)/(5,5)
model.add(Dropout(0.2))
model.add(Conv2D(150, (7, 7), activation='relu', padding='same', kernel_constraint=maxnorm(3))) # use any value from 1-100
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3))) #use 512
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax')) # use 'tanh'


#Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #optimizer=optimizers.Adadelta()/optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='mean_squared_error'/ 
#mean_absolute_error/mean_squared_logarithmic_error/sparse_categorical_crossentropy/categorical_crossentropy
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=16) #use batch_size=16/32/64 


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)#use verbose=1
print("Accuracy: %.2f%%" % (scores[1]*100))
classes = model.predict(X_test, batch_size=16)
saveModel(model)
