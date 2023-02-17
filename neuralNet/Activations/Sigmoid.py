from Activation import Activation
from extras import sigmoid, D_sigmoid

#sigmoid activation layer
class Sigmoid(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(sigmoid, D_sigmoid, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.Sigmoid",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
