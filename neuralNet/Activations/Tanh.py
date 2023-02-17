from Activation import Activation
from extras import tanh, D_tanh

#hyperbolic tangent activation layer
class Tanh(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(tanh, D_tanh, outputSize)
    
    def save(self):
        data = {
            "Type":"neuralNet.Tanh",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
