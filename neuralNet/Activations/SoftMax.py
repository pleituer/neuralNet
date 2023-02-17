from neuralNet.Layer import Layer
import numpy as np

#softmax activation
class SoftMax(Layer):
    def __init__(self, outputSize):
        self.outputSize = outputSize

    def forward(self, input):
        self.input = input
        tmp = np.exp(self.input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, outputGradient, learningRate):
        n = np.size(self.output)
        M = np.tile(self.output, n)
        return np.dot(M * (np.identity(n) - np.transpose(M)), outputGradient)

    def save(self):
        data = {
            "Type":"neuralNet.SoftMax",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
