import numpy as np
from scipy import signal
from Layer import Layer
from RawConvolute import RawConvolute
from Stride import Stride

#convolution layer, with stride
class Convolute(Layer):
    def __init__(self, inputShape, kernalSize, depth, padding, stride, kernal=None, bias=None):
        self.convLayer = RawConvolute(inputShape, kernalSize, depth, padding, kernal=kernal, bias=bias)
        self.strideLayer = Stride(self.convLayer.outputShape, stride)
        self.outputShape = self.strideLayer.outputshape
    
    def forward(self, input):
        self.input = input
        return self.strideLayer.forward(self.convLayer.forward(self.input))
    
    def backward(self, outputGradient, learningRate):
        return self.convLayer.backward(self.strideLayer.backward(outputGradient, learningRate), learningRate)
    
    def save(self):
        data = {
            "Type":"neuralNet.Convolute",
            "inputShape":self.convLayer.inputShape,
            "kernalSize":self.convLayer.kernalSize,
            "depth":self.convLayer.depth,
            "padding":self.convLayer.padding,
            "kernal":self.convLayer.kernal,
            "bias":self.convLayer.bias,
            "stride":self.strideLayer.stride
        }
        return data