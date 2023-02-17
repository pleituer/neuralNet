from Activation import Activation
from extras import identity, D_identity

#Identidy activation layer
class Identity(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(identity, D_identity, outputSize)
    
    def save(self):
        data = {
            "Type":"neuralNet.Identity",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data
