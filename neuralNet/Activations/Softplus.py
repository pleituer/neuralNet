from neuralNet.Activations.Activation import Activation
from neuralNet.extras import softplus, D_softplus

#Softplus activation layer
class Softplus(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(softplus, D_softplus, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.Softplus",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
