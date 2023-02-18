import numpy as np
from neuralNet.Layer import Layer

#Reshapes the output of the previous layer to the input of the next layer
class Reshape(Layer):
    def __init__(self, inputShape, outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape
    
    def forward(self, input):
        self.input = input
        return np.reshape(self.input, self.outputShape)
    
    def backward(self, outputGradient, learningRate):
        return np.reshape(outputGradient, self.inputShape)

    def save(self):
        data = {
            "Type":"neuralNet.Reshape",
            "inputShape":self.inputShape,
            "outputShape":self.outputShape
        }
        return data