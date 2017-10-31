from __future__ import print_function
from Layers import Input, Dense
from Activation import Softmax
import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.model = []
        self.optim = ""

    def add(self, obj):
        """ Adding Layers to Model """
        if len(self.model) == 0:
            assert (type(obj) is Input), "Where is the Input layer?"
        self.model.append(obj)

    def train(self, X, Y, lr=1, reg=0, epochs=1, batch_size=1, verbose=1):
        """ Training Model """
        for epoch in range(epochs):
            z = self.forward(X)
            dw = self.backward(Y, z)

            if verbose == 1:
                self.print_results(X,Y,z[-1],epoch)

            self.update(lr, dw)

    def forward(self, x):
        """ Feed Forward """
        z = [x]
        for i, layer in enumerate(self.model[1:]):
            z.append(layer.forward(z[i]))
        return z

    def backward(self, y, z):
        """ Backward Propagation """
        delta = ( z[-1] -  y )
        dw = [delta.T.dot(Dense.add_bias(z[-3]))]
        for i, layer in enumerate(reversed(self.model[1:-1])):
            delta = layer.backward(delta)
            if type(layer) is not Dense:
                dw.append(delta.T.dot(Dense.add_bias(z[-i-4])))
        return dw
    
    def update(self, lr, dw):
        """ Update Weights """
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
        """ Calculate accuracy """
        y_pred = self.predict(x)
        y = np.argmax(Y,axis=1) if Y.shape[1] != 1 else Y
        num_correct = np.sum(y_pred == y)
        return float(num_correct) / Y.shape[0]

    def loss(self, probs, Y):
        """ Calculate Loss """
        loss = -Y*np.log(Softmax.cost(probs))
        return loss.sum() / Y.shape[0]

    def predict(self, x):
        """ Predict X """
        probs = self.forward(x)[-1]
        if probs.shape[1] == 1:
            return (probs > 0.5)
        return np.argmax(probs, axis=1)

    def summary(self):
        """ Print model summary """
        print((30*"-")+"\n"+(11*" ")+"Summary\n"+(30*"-"))
        for layer in self.model:
            print(layer.summary())
        print(30*"-")

    def print_results(self, X, Y, probs, epoch):
        acc = self.accuracy(X,Y)
        loss = self.loss(probs,Y)
        print("Epoch {}: loss: {}, training_accuracy: {}".format(epoch, loss, acc))

    