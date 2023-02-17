from Activation import Activation
from extras import silu, D_silu

#SiLU
class SiLU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(silu, D_silu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.SiLU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
