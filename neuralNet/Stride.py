import numpy as np
from neuralNet.Layer import Layer

#Stride layer, can be treated as a pooling layer that extract the top left corner, with pool size of stride and stride of stride
class Stride(Layer):
    def __init__(self, inputShape, stride):
        self.inputShape = self.inputDepth, self.inputHeight, self.inputWidth = inputShape
        self.stride = stride
        self.outputshape = self.inputDepth, round(self.inputHeight/self.stride), round(self.inputWidth, self.stride)
    
    def forward(self, input):
        self.input = input
        return self.input[:, ::self.stride, ::self.stride]
    
    def backward(self, outputGradient, learningRate):
        inputGradient = np.zeros(self.inputShape)
        inputGradient[:, ::self.stride, ::self.stride] = outputGradient
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.Stride",
            "inputShape":self.inputShape,
            "stride":self.stride
        }
        return data