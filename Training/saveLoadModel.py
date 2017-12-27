from keras.models import model_from_json
from keras.optimizers import SGD
def saveModel(model,file='model1'):
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

	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) #optimizer=optimizers.Adadelta()/optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='mean_squared_error'/ 
	#mean_absolute_error/mean_squared_logarithmic_error/sparse_categorical_crossentropy/categorical_crossentropy
	print(model.summary())
	score = loaded_model.evaluate(X, Y, verbose=0)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

	return model
loadModel("model1")
