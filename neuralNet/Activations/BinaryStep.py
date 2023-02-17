from Activation import Activation
from extras import binStep, D_binStep

#Binary Step activation layer
class BinaryStep(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(binStep, D_binStep, outputSize)
    
    def save(self):
        data = {
            "Type":"neuralNet.BinaryStep",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
