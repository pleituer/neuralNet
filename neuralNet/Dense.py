import numpy as np
from Layer import Layer

#Dense, or Linear layer
class Dense(Layer):
    def __init__(self, inputSize, outputSize, wantBias=True, initWeights=None, initBias=None):
        self.inputSize, self.outpuSize = inputSize, outputSize
        if initWeights is None: self.weights = np.random.randn(outputSize, inputSize)
        else: self.weights = initWeights
        if wantBias: 
            if initBias is None: self.bias = np.random.randn(outputSize, 1)
            else: self.bias = initBias
        else: self.bias = None
        self.wantBias = wantBias
    
    def forward(self, input):
        self.input = input
        if self.wantBias: return np.dot(self.weights, self.input)  + self.bias
        else: return np.dot(self.weights, self.input)
    
    def backward(self, outputGradient, learningRate):
        weightsGradient = np.dot(outputGradient, self.input.T)
        inputGradient = np.dot(self.weights.T, outputGradient)
        self.weights -= learningRate*weightsGradient
        if self.wantBias: self.bias -=learningRate*outputGradient
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.Dense",
            "inputSize":self.inputSize,
            "outputSize":self.outpuSize,
            "weights":self.weights,
            "bias":self.bias,
            "wantBias":self.wantBias
        }
        return data