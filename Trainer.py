from __future__ import print_function
import numpy as np
from Network import NeuralNetwork
from Activation import ReLU, Softmax
from Layers import Dense, Input
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(1337)  # for reproducibility

# Init Vars
epochs = 500
batch_size = 100

# num_train = 10000
# num_test = 1000

# Xtr = np.random.randint(2, size=(num_train, 2))
# ytr =  Xtr[:,0] ^ Xtr[:,1]

# Ytr = np.eye(2)[ytr]

# X_test = np.random.randint(2, size=(num_test, 2))
# y_test = X_test[:,0] ^ X_test[:,1]
# Y_test = np.eye(2)[y_test]


# Fetch Data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define Model
model = NeuralNetwork()
model.add(Input(784))
model.add(Dense(500))
model.add(ReLU())
model.add(Dense(10))
model.add(Softmax())
model.summary()

# Compile Model
model.compile()

# Train Model
for _ in range(epochs):
    batch = mnist.train.next_batch(batch_size)
    model.train(batch[0], batch[1], lr=0.01, epochs=1)

# Test Model
print("Test Accuracy: {}".format(model.accuracy(mnist.test.images, mnist.test.labels)))

