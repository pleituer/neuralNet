from neuralNet.Activations.Activation import Activation
from neuralNet.extras import elu, D_elu
import numpy as np

#ELU
class ELU(Activation):
    def __init__(self, outputSize, alpha=None):
        self.outputSize = outputSize
        if alpha is None: self.alpha = np.random.randn(1)[0]
        else: self.alpha = alpha
        Elu = lambda x: elu(x, self.alpha)
        D_Elu = lambda x: D_elu(x, self.alpha)
        super().__init__(Elu, D_Elu, outputSize)

    def backward(self, outputGradient, learningRate):
        self.alpha -= (self.input<np.zeros(np.shape(self.input)))*learningRate*outputGradient*np.reshape((np.exp(self.input)-1), (outputGradient,))
        return np.multiply(outputGradient, self.D_activation(self.input))

    def save(self):
        data = {
            "Type":"neuralNet.ELU",
            "outputSize":self.outputSize,
            "alpha":self.alpha
        }
        return data
