from Activation import Activation
from extras import prelu, D_prelu

#PReLU
class PReLU(Activation):
    def __init__(self, outputSize, alpha=None):
        self.outputSize = outputSize
        if alpha is None: self.alpha = np.random.randn(1)[0]
        else: self.alpha = alpha
        Prelu = lambda x: prelu(x, self.alpha)
        D_Prelu = lambda x:D_prelu(x, self.alpha)
        super().__init__(Prelu, D_Prelu, outputSize)
    
    def backward(self, outputGradient, learningRate):
        self.alpha -= (self.input<np.zeros(np.shape(self.input)))*learningRate*outputGradient*np.reshape(self.input, (outputGradient,))
        return np.multiply(outputGradient, self.D_activation(self.input))

    def save(self):
        data = {
            "Type":"neuralNet.PReLU",
            "outputSize":self.outputSize,
            "alpha":self.alpha
        }
        return data
