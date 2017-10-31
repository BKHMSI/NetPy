from __future__ import print_function
import numpy as np

class Dense(object):
    def __init__(self, units):
        self.units = units
        self.W = []

    def init_weights(self, H):
        """ Xavier Initialization """
        self.W = np.random.randn(self.units,H+1) / np.sqrt(float(self.units)/2)

    def forward(self, a):
        """ Feed Forward """
        return Dense.add_bias(a).dot(self.W.T)      

    def backward(self, delta):
        return delta.dot(self.W)

    def update(self, dw):
        self.W += dw
        
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

    