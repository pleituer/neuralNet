from Activation import Activation
from extras import relu, D_relu

#ReLu activation layer
class ReLU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(relu, D_relu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.ReLU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
