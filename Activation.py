import numpy as np

class Softmax(object):
    def __init__(self):
        pass
 
    def forward(self, x):
        s = np.max(x, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(x - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div

    def backward(self, x):
         pass

    def summary(self):
        return "Softmax activation function"

    @staticmethod
    def cost(x):
        s = np.max(x, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(x - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div

class Sigmoid(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return 1 / (1+np.exp(-x))

    def backward(self, delta):
        gg = self.forward(self.x)
        return delta[:, 1:] * (gg  * (1 - gg))

    def summary(self):
        return "Sigmoid activation function"

class Tanh(object):
    def __init__(self):
        pass

    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - np.tanh(x)**2

    def summary(self):
        return "Tanh activation function"

class ReLU(object):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0,x)
    
    def backward(self, delta):
        return delta[:, 1:] * (self.x > 0)

    def summary(self):
        return "ReLU activation function"

class PReLU(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def forward(self, x):
        return np.maximum(self.alpha*x, x)
    
    def backward(self, x):
        pass

    def summary(self):
        return "PReLU activation function with alpha: " % self.alpha


class ELU(object):
    def __init__(object):
        pass

    def forward(self, x):
        return np.maximum(np.exp(x) - 1, x)

    def backward(self):
        pass

    def summary(self):
        return "ELU activation function"