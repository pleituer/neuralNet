import numpy as np
from neuralNet.Layer import Layer
from neuralNet.Reshape import Reshape

#flattens the output of the previous layer
class Flatten(Reshape):
    def __init__(self, inputShape):
        self.inputShape = inputShape
        outputShape = (np.prod(inputShape), 1)
        super().__init__(inputShape, outputShape)

    def save(self):
        data = {
            "Type":"neuralNet.Flatten",
            "inputShape":self.inputShape
        }
        return data