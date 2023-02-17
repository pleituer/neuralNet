from neuralNet.Layer import Layer
import numpy as np

#activation base class (softmax excluded)
class Activation(Layer):
    def __init__(self, activation, D_activation, outputSize):
        self.activation = activation
        self.D_activation = D_activation
        self.outputSize = outputSize
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, outputGradient, learningRate):
        return np.multiply(outputGradient, self.D_activation(self.input))
    
    def save(self):
        pass
