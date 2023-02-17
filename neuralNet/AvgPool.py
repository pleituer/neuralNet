import numpy as np
from scipy import signal
from neuralNet.Layer import Layer

#Average pooling layer
class AvgPool(Layer):
    def __init__(self, inputShape, poolShape=(2,2), stride=2):
        self.inputShape = self.inputDepth, self.inputHeight, self.inputWidth = inputShape
        self.poolShape = self.poolHeight, self.poolWidth = poolShape
        self.stride = stride
        self.outputShape = self.outputDepth, self.outputHeight, self.outputWidth = (self.inputDepth, 1 + (self.inputHeight - self.poolHeight)//self.stride, 1 + (self.inputWidth - self.poolWidth)//self.stride)
        self.kernal = np.ones(self.poolShape)/np.prod(self.poolShape)
    
    def forward(self, input):
        self.input = input
        inputPad = np.pad(self.input, ((0,0), (0, (self.inputHeight - self.poolHeight)%self.stride), (0, (self.inputWidth - self.poolWidth)%self.stride)))
        self.output = np.zeros(self.outputShape)
        for i in range(self.inputDepth):
            self.output[i] = signal.convolve(inputPad[i], self.kernal, "valid")[::self.stride, ::self.stride]
        return self.output
    
    def backward(self, outputGradient, learningRate):
        inputGradient = np.zeros(self.inputShape)
        for i in range(self.inputDepth):
            for j in range(self.outputHeight):
                for k in range(self.outputWidth):
                   inputGradient[i, j*self.stride:j*self.stride + self.poolHeight, k*self.stride:k*self.stride + self.poolWidth] += outputGradient[i, j, k]*self.kernal
        return inputGradient

    def save(self):
        data = {
            "Type":"neuralNet.AvgPool",
            "inputShape":self.inputShape,
            "poolShape":self.poolShape,
            "stride":self.stride
        }
        return data