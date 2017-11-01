from __future__ import print_function
from Layers import Input, Dense
from Activation import Softmax
from Optimizers import SGD
from copy import deepcopy
from tqdm import tqdm
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

    def get_batch(self, X, Y, idx, batch_size):
        start = idx*batch_size
        end = (idx+1)*batch_size
        return X[start:end], Y[start:end]

    def split_data(self, split_ratio, X, Y):
        border = int(X.shape[0]*split_ratio)
        X_valid = X[:border]
        Y_valid = Y[:border]
        X_train = X[border:]
        Y_train = Y[border:]
        return X_train, Y_train, X_valid, Y_valid

    def train(self, X, Y, lr = 1, reg = 0, epochs = 1, batch_size = 1, 
                    verbose = 1, validation_split=0, validation_data=None):

        """ Training Model """

        if validation_data is None:
            X, Y, X_val, Y_val = self.split_data(validation_split, X, Y)
        else:
            X_val, Y_val = validation_data

        batches = int(X.shape[0] / batch_size)
        
        for epoch in range(epochs):

            for batch in tqdm(range(batches)):
                x, y = self.get_batch(X, Y, batch, batch_size) 
                z = self.forward(x)
                dw = self.backward(y, z)
                self.update(dw, lr, reg, batch_size)

            if verbose == 1:
                self.print_results(epoch+1, X,Y, X_val, Y_val)


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
    
    def update(self, dw, lr, reg, batch_size):
        """ Update Weights """
        idx = - 1
        for layer in self.model:
            if type(layer) is Dense:
                layer.update(self.optim, dw[idx], lr, reg, batch_size)
                idx = idx - 1

    def compile(self, optim = None):
        """ Initialize weights and set optimizer"""
        optimizer = optim if optim != None else SGD()
        units = self.model[0].units
        for layer in self.model[1:]:
            if type(layer) is Dense:
                layer.init_weights(units)
                layer.optim = deepcopy(optimizer)
                units = layer.units

    def accuracy(self, X, Y):
        """ Calculate accuracy """
        y_pred = self.predict(X)
        y = np.argmax(Y,axis=1) if Y.shape[1] != 1 else Y
        num_correct = np.sum(y_pred == y)
        return float(num_correct) / Y.shape[0]

    def loss(self, X, Y):
        """ Calculate Loss """
        probs = self.forward(X)[-1]
        loss = -Y*np.log(Softmax.cost(probs))
        return loss.sum() / Y.shape[0]

    def predict(self, X):
        """ Predict X """
        probs = self.forward(X)[-1]
        if probs.shape[1] == 1:
            return (probs > 0.5)
        return np.argmax(probs, axis=1)

    def summary(self):
        """ Print model summary """
        print((30*"-")+"\n"+(11*" ")+"Summary\n"+(30*"-"))
        for layer in self.model:
            print(layer.summary())
        print(30*"-")

    def print_results(self, epoch, X, Y, X_val, Y_val):
        train_acc = self.accuracy(X,Y)
        train_loss = self.loss(X,Y)
        if X_val.shape[0] != 0:
            val_acc = self.accuracy(X_val, Y_val)
            val_loss = self.accuracy(X_val, Y_val)
            print("Epoch {}: train_loss: {}, training_acc: {}, val_loss: {}, val_acc: {}".format(epoch, train_loss, train_acc, val_loss, val_acc))
        else:
            print("Epoch {}: train_loss: {}, training_acc: {}".format(epoch, train_loss, train_acc))


    