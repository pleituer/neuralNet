import numpy as np
from scipy import signal
from Layer import Layer

#Convolution without stride
class RawConvolute(Layer):
    def __init__(self, inputShape, kernalSize, depth, padding, kernal=None, bias=None):
        self.padding = padding
        self.kernalSize = kernalSize
        self.inputShape = self.inputDepth, inputHeight, inputWidth = inputShape
        self.outputShape = self.depth, self.outputHeight, self.outputWidth = (depth, inputHeight - kernalSize + 1 + 2*self.padding, inputWidth - kernalSize + 1 + 2*self.padding)
        self.kernalShape = (depth, self.inputDepth, kernalSize, kernalSize)
        if kernal is None: self.kernal = np.random.randn(*self.kernalShape)
        else: self.kernal = kernal
        if bias is None: self.bias = np.random.randn(*self.outputShape)
        else: self.bias = bias
    
    def forward(self, input):
        self.input = input
        self.inputPad = np.pad(self.input, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        self.output = np.copy(self.bias)
        for i in range(self.depth):
            for j in range(self.inputDepth):
                self.output[i] += signal.correlate2d(self.inputPad[j], self.kernal[i, j], "valid")
        return self.output
    
    def backward(self, outputGradient, learningRate):
        #outputGradient = outputGradient[:, self.padding:self.outputHeight-self.padding, self.padding:self.outputWidth-self.padding]
        kernalGradient = np.zeros(self.kernalShape)
        inputGradient = np.zeros(self.inputShape)
        for i in range(self.depth):
            for j in range(self.inputDepth):
                kernalGradient[i, j] = signal.correlate2d(self.inputPad[j], outputGradient[i], "valid")
                inputGradient[j] = signal.convolve(outputGradient[i, self.padding:self.outputHeight-self.padding, self.padding:self.outputWidth-self.padding], self.kernal[i, j], "full")
        self.kernal -= learningRate*kernalGradient
        self.bias -= learningRate*outputGradient
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.RawConvolute",
            "inputShape":self.inputShape,
            "kernalSize":self.kernalSize,
            "depth":self.depth,
            "padding":self.padding,
            "kernal":self.kernal,
            "bias":self.bias
        }
        return data