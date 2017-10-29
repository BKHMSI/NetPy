from __future__ import print_function
from Layers import Input, Dense
from Activation import Softmax
import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.model = []
        self.optim = ""

    def add(self, obj):
        """ Adding Layers """
        if len(self.model) == 0:
            assert (type(obj) is Input), "Where is the Input layer?"
        self.model.append(obj)

    def train(self, X, Y, lr=1, reg=0, epochs=1, batch_size=1, verbose=1):
        for epoch in range(1):
            z = self.forward(X)
            dw = self.backward(Y, z)
            self.update(lr, dw)

            if verbose == 1:
                acc = self.accuracy(X,Y)
                loss = self.loss(z[-1],Y)
                print("Epoch {}: loss: {}, training_accuracy: {}".format(epoch, loss, acc))

    def forward(self, x):
        z = [x]
        for i, layer in enumerate(self.model[1:]):
            z.append(layer.forward(z[i]))
        return z

    def backward(self, y, z):
        dw = []
        delta = ( z[-1] -  y )
        dw.append(delta.T.dot(Dense.add_bias(z[-3])))
        for i, layer in enumerate(reversed(self.model[1:-1])):
            delta = layer.backward(delta)
            if type(layer) is not Dense:
                dw.append(delta.T.dot(Dense.add_bias(z[-i-4])))
        return dw
    
    def update(self, lr, dw):
        idx = - 1
        for layer in self.model:
            if type(layer) is Dense:
                layer.update(-lr*dw[idx])
                idx = idx - 1

    def compile(self, optim='sgd'):
        """ Initialize weights and set optimizer"""
        self.optim = optim
        units = self.model[0].units
        for layer in self.model[1:]:
            if type(layer) is Dense:
                layer.init_weights(units)
                units = layer.units

    def accuracy(self, x, Y):
        y_pred = self.predict(x)
        y = np.argmax(Y,axis=1) 
        num_correct = np.sum(y_pred == y)
        return float(num_correct) / Y.shape[0]

    def loss(self, probs, Y):
        loss = -Y*np.log(probs)
        return np.sum(loss) / Y.shape[0]

    def predict(self, x):
        probs = self.forward(x)[-1]
        return np.argmax(probs, axis=1)

    def summary(self):
        print((30*"-")+"\n"+(11*" ")+"Summary\n"+(30*"-"))
        for layer in self.model:
            print(layer.summary())
        print(30*"-")

    