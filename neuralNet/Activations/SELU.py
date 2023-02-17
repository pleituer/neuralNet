from neuralNet.Activations.Activation import Activation
from neuralNet.extras import selu, D_selu

#SELU
class SELU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(selu, D_selu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.SELU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data