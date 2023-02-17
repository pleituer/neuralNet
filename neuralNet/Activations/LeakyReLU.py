from Activation import Activation
from extras import lrelu, D_lrelu

#Leaky ReLU
class LeakyReLU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(lrelu, D_lrelu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.LeakyReLU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
