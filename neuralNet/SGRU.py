import numpy as np
from Layer import Layer
from extras import tanh

#Simplistic GRU, it is a GRU but with one internal weight and one internal activation (tanh)
#update: this has been remaned to OSGRU, this is the original version, a newer version has been made
class SGRU(Layer):
    def __init__(self, inputSize, outputSize, externalWeights=None, internalWeights=None, bias=None):
        self.inputSize, self.outputSize = inputSize, outputSize
        if externalWeights is None: self.externalWeights = np.random.randn(outputSize, inputSize)
        else: self.externalWeights = externalWeights
        if internalWeights is None: self.internalWeights = np.random.randn(outputSize, 1)
        else: self.internalWeights = internalWeights
        if bias is None: self.bias = np.random.randn(outputSize, 1)
        else: self.bias = bias
        self.value = np.zeros((outputSize, 1))
    
    def forward(self, input):
        self.input = input
        self.value = np.dot(self.externalWeights, self.input) + np.multiply(self.internalWeights, self.value) + self.bias
        return self.value
    
    def backward(self, outputGradient, learningRate):
        externalWeightsGradient = np.dot(outputGradient, self.input.T)
        internalWeightsGradient = np.multiply(outputGradient, self.value)
        inputGradient = np.dot(self.externalWeights.T, outputGradient)
        self.externalWeights -=learningRate*externalWeightsGradient
        self.internalWeights -=learningRate*internalWeightsGradient
        self.bias -=learningRate*outputGradient
        self.internalWeights = tanh(self.internalWeights)
        self.value = np.zeros((self.outputSize, 1))
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.SGRU",
            "inputSize":self.inputSize,
            "outputSize":self.outputSize,
            "internalWeights":self.internalWeights,
            "externalWeights":self.externalWeights,
            "bias":self.bias
        }
        return data