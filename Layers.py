from __future__ import print_function
import numpy as np

class Dense(object):
    def __init__(self, units):
        self.units = units
        self.W = []
        self.optim = None

    def init_weights(self, H):
        """ Xavier Initialization """
        self.W = np.random.randn(self.units,H+1) / np.sqrt(float(self.units)/2)

    def forward(self, a):
        """ Feed Forward """
        return Dense.add_bias(a).dot(self.W.T)      

    def backward(self, delta):
        return delta.dot(self.W)

    def update(self, dw, lr, reg, m):
        dw /= m 
        dw[1:] += reg * (self.W[1:]**2)
        self.W += self.optim.update(lr, dw)
        
    def summary(self):
        return "Dense layer with size: {}".format(self.units)

    @staticmethod
    def add_bias(x):
        ones = np.ones((len(x), 1))
        return np.concatenate((ones, x), axis=1)



class Input(object):
    def __init__(self, units):
        self.units = units 

    def forward(self):
        pass
    
    def summary(self):
        return "Input layer with size: {}".format(self.units) 

    