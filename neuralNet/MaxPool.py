import numpy as np
from neuralNet.Layer import Layer

#Max pooling layer
class MaxPool(Layer):
    def __init__(self, inputShape, poolShape=(2,2), stride=2):
        self.inputShape = self.inputDepth, self.inputHeight, self.inputWidth = inputShape
        self.poolShape = self.poolHeight, self.poolWidth = poolShape
        self.stride = stride
        self.outputShape = self.outputDepth, self.outputHeight, self.outputWidth = (self.inputDepth, 1 + (self.inputHeight - self.poolHeight)//self.stride, 1 + (self.inputWidth - self.poolWidth)//self.stride)

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.outputShape)
        for i in range(self.inputDepth):
            for j in range(self.outputHeight):
                for k in range(self.outputWidth):
                    self.output[i,j,k] = np.max(self.input[i, self.stride*j:self.poolHeight + self.stride*j, self.stride*k:self.poolWidth + self.stride*k])
        return self.output
    
    def backward(self, outputGradient, learningRate):
        inputGradient = np.zeros(self.inputShape)
        for i in range(self.inputDepth):
            for j in range(self.outputHeight):
                for k in range(self.outputWidth):
                    inputGradient[i, self.stride*j:self.poolHeight + self.stride*j, self.stride*k:self.poolWidth + self.stride*k] += (self.input[i, self.stride*j:self.poolHeight + self.stride*j, self.stride*k:self.poolWidth + self.stride*k]==self.output[i, j, k])*outputGradient[i, j, k]
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.MaxPool",
            "inputShape":self.inputShape,
            "poolShape":self.poolShape,
            "stride":self.stride
        }
        return data