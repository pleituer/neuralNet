from Activation import Activation
from extras import gaussian, D_gaussian

#Gaussian
class Gaussian(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(gaussian, D_gaussian, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.Gaussian",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
