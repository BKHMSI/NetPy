from __future__ import print_function
import numpy as np
from Network import NeuralNetwork
from Activation import ReLU, Softmax, Sigmoid
from Layers import Dense, Input
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(1337)  # for reproducibility

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
for _ in range(5):
  batch = mnist.train.next_batch(100)
  model.train(batch[0], batch[1], lr=0.01, epochs=1)
# Test Model
print("Test Accuracy: {}".format(model.accuracy(mnist.test.images, mnist.test.labels)))

