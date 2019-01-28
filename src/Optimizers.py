import numpy as np

class SGD(object):
    def __init__(self):
        pass
    
    def update(self, lr, dw):
        return -lr*dw

class Momentum(object):
    def __init__(self, momentum):
        self.alpha = momentum
        self.v = 0
    
    def update(self, lr, dw):
        self.v = self.alpha * self.v - lr * dw 
        return self.v

class Nesterov(object):
    def __init__(self, alpha = 0.5):
        self.alpha = alpha
        self.v = 0
    
    def update(self, lr, dw):
        self.v = self.alpha * self.v - lr * dw 
        return self.v

class AdaGrad(object):
    def __init__(self):
        self.acc = 0
    
    def update(self, lr, dw):
        self.acc += dw * dw
        return - lr * dw / (np.sqrt(self.acc) + 1e-7)

class RMSProp(object):
    def __init__(self, decay_rate):
        self.acc = 0
        self.decay_rate = decay_rate
    
    def update(self, lr, dw):
        self.acc = self.decay_rate*self.acc + (1-self.decay_rate) * (dw * dw)
        return - lr * dw / (np.sqrt(self.acc) + 1e-7)

class Adam(object):
    def __init__(self, beta1 = 0.9, beta2 = 0.999):
        self.beta1 = beta1 
        self.beta2 = beta2 
        self.acc = 0
        self.m = 0
    
    def update(self, lr, dw):
        self.m = self.beta1 * self.m + (1-self.beta2) * dw
        self.acc = self.beta2 * self.acc + (1-self.beta2) * (dw*dw)
        return - lr * self.m / (np.sqrt(self.acc) + 1e-8)