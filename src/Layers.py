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


class Dropout(object):
    def __init__(self, drop):
        self.drop = drop
        self.dropout = None

    def forward(self, x):
        self.dropout = np.random.binomial(1, 1-self.drop, size=x.shape)
        return x * self.dropout

    def backward(self, dw):
        dropout = Dense.add_bias(self.dropout)
        return dw * dropout

    def summary(self):
        return "Dropout layer of: {}".format(self.drop)

class Flatten(object):
    def __init__(self):
        self.shape = None 

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(-1)

    def backward(self, dw):
        return dw.reshape(self.shape)


class Input(object):
    def __init__(self, units):
        self.units = units 

    def forward(self):
        pass
    
    def summary(self):
        return "Input layer with shape: {}".format(self.units) 

class Conv2D(object):
    def __init__(self, filters, kernel_size, strides=(1,1), padding=(1,1)):
        self.filters = filters  
        self.kernel_size = kernel_size
        self.strides = strides 
        self.padding = padding

        self.W = np.zeros(()) 

    def forward(self, x):
        pass 

    def backward(self):
        pass 

    def summary(self):
        pass

    