from __future__ import print_function

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from src.Network import NeuralNetwork
from src.Activation import ReLU, Softmax
from src.Layers import Dense, Input, Dropout
from src.Optimizers import RMSProp

np.random.seed(1337)  # for reproducibility

# Init Vars
epochs = 5
batch_size = 100

# Fetch Data
mnist   = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train = mnist.train.images 
Y_train = mnist.train.labels
X_val   = mnist.validation.images
Y_val   = mnist.validation.labels
X_test  = mnist.test.images
Y_test  = mnist.test.labels

# Preprocess Data
X_train = X_train.astype('float32')
X_val   = X_val.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_val   /= 255
X_test  /= 255

# Define Model
model = NeuralNetwork()
model.add(Input(784))
model.add(Dense(500))
model.add(ReLU())
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Softmax())
model.summary()

# Compile Model
model.compile(optim=RMSProp(decay_rate=0.3))

# Train Model
model.train(X_train, Y_train, lr=1e-2, reg=0, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))

# Test Model
print("Test Accuracy: {}".format(model.accuracy(X_test, Y_test)))