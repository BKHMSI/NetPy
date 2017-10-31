from __future__ import print_function
import numpy as np
from Network import NeuralNetwork
from Activation import ReLU, Softmax, Sigmoid
from Layers import Dense, Input

np.random.seed(1337)  # for reproducibility

# Init Vars
epochs = 22

num_train = 10000
num_test = 1000

# Generate Training Set
Xtr = np.random.randint(2, size=(num_train, 2))
ytr = Xtr[:,0] ^ Xtr[:,1]
ytr = ytr.reshape(ytr.shape[0], 1)

# Generate Testing Set
X_test = np.random.randint(2, size=(num_test, 2))
y_test = X_test[:,0] ^ X_test[:,1]
y_test = y_test.reshape(y_test.shape[0], 1)


# Define Model
model = NeuralNetwork()
model.add(Input(2))
model.add(Dense(2))
model.add(ReLU())
model.add(Dense(1))
model.add(Sigmoid())
model.summary()

# Compile Model
model.compile()

# Train Model
model.train(Xtr, ytr, lr=1e-4, epochs=epochs)

# Test Model
print("Test Accuracy: {}".format(model.accuracy(X_test, y_test)))

