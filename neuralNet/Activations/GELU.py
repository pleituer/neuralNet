from Activation import Activation
from extras import gelu, D_gelu

#GELU activation layer
class GELU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(gelu, D_gelu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.GELU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
