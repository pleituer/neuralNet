import numpy as np
from Layer import Layer
from extras import sigmoid, tanh

#Long Short Term Memory, still in progress
#I do not guarentee there will be no bugs
class LSTM(Layer): 
    def __init__(self, inputSize, outputSize, W=None, U=None, B=None):
        self.inputSize, self.outputSize = inputSize, outputSize
        if W is None:
            self.forgetGateW = np.random.randn(outputSize, 1)
            self.updateGateW = np.random.randn(outputSize, 1)
            self.outputActivationW = np.random.randn(outputSize, 1)
            self.outputGateW = np.random.randn(outputSize, 1)
        else:
            self.forgetGateW, self.updateGateW, self.outputActivationW, self.outputGateW = W
        if U is None:
            self.forgetGateU = np.random.randn(outputSize, inputSize)
            self.updateGateU = np.random.randn(outputSize, inputSize)
            self.outputActivationU = np.random.randn(outputSize, inputSize)
            self.outputGateU = np.random.randn(outputSize, inputSize)
        else:
            self.forgetGateU, self.updateGateU, self.outputActivationU, self.outputGateU = U
        if B is None:
            self.forgetGateB = np.random.randn(outputSize, 1)
            self.updateGateB = np.random.randn(outputSize, 1)
            self.outputActivationB = np.random.randn(outputSize, 1)
            self.outputGateB = np.random.randn(outputSize, 1)
        else:
            self.forgetGateB, self.updateGateB, self.outputActivationB, self.outputGateB = B
        self.shortTermMem = np.zeros((outputSize, 1))
        self.longTermMem = np.zeros((outputSize, 1))
        self.sigmaH = lambda x: x #tanh(x) or x
        self.D_sigmaH = lambda x: 1 #(1 - np.power(tanh(x), 2) or 1
    
    def forward(self, input):
        self.input = input
        self.forgetGate = sigmoid(np.dot(self.forgetGateU, self.input) + self.forgetGateW * self.shortTermMem + self.forgetGateB)
        self.updateGate = sigmoid(np.dot(self.updateGateU, self.input) + self.updateGateW * self.shortTermMem + self.updateGateB)
        self.outputActivation = sigmoid(np.dot(self.outputActivationU, self.input) + self.outputActivationW * self.shortTermMem + self.outputActivationB)
        self.inputActivation = tanh(np.dot(self.outputGateU, self.input) + self.outputGateW * self.shortTermMem + self.outputGateB)
        self.longTermMem = self.forgetGate * self.longTermMem + self.updateGate * self.inputActivation
        self.shortTermMem = self.outputActivation * self.sigmaH(self.longTermMem)
        return self.shortTermMem
    
    def backward(self, outputGradient, learningRate):
        forgetGateGradient = self.outputActivation * self.D_sigmaH(self.longTermMem) * self.longTermMem * outputGradient
        updateGateGradient = self.outputActivation * self.D_sigmaH(self.longTermMem) * self.inputActivation * outputGradient
        outputActivationGradient = self.sigmaH(self.longTermMem) * outputGradient
        inputActivationGradient = self.outputActivation * self.D_sigmaH(self.longTermMem) * self.updateGate * outputGradient
        forgetGateGradient = forgetGateGradient * (1 - forgetGateGradient)
        updateGateGradient = updateGateGradient * (1 - updateGateGradient)
        outputActivationGradient = outputActivationGradient * (1 - outputActivationGradient)
        inputActivationGradient = (1 - np.power(inputActivationGradient, 2))
        forgetGateWGradient = forgetGateGradient * self.shortTermMem
        forgetGateUGradient = np.dot(forgetGateGradient, self.input.T)
        updateGateWGradient = updateGateGradient * self.shortTermMem
        updateGateUGradient = np.dot(updateGateGradient, self.input.T)
        outputActivationWGradient = outputActivationGradient * self.shortTermMem
        outputActivationUGradient = np.dot(outputActivationGradient, self.input.T)
        outputGateWGradient = inputActivationGradient * self.shortTermMem
        outputGateUGradient = np.dot(inputActivationGradient, self.input.T)
        forgetGateBGradient = forgetGateGradient
        updateGateBGradient = updateGateGradient
        outputActivationBGradient = outputActivationGradient
        outputGateBGradient = inputActivationGradient
        inputGradient = np.dot(self.outputActivationU.T, outputActivationGradient) * self.sigmaH(self.longTermMem) + self.outputActivation * (self.longTermMem * np.dot(self.forgetGateU.T, forgetGateGradient) + np.dot(self.updateGateU.T, self.inputActivation) + np.dot(self.outputGateU.T, self.updateGate))
        self.forgetGateW -= forgetGateWGradient * learningRate
        self.forgetGateU -= forgetGateUGradient * learningRate
        self.forgetGateB -= forgetGateBGradient * learningRate
        self.updateGateW -= updateGateWGradient * learningRate
        self.updateGateU -= updateGateUGradient * learningRate
        self.updateGateB -= updateGateBGradient * learningRate
        self.outputActivationW -= outputActivationWGradient * learningRate
        self.outputActivationU -= outputActivationUGradient * learningRate
        self.outputActivationB -= outputActivationBGradient * learningRate
        self.outputGateW -= outputGateWGradient * learningRate
        self.outputGateU -= outputGateUGradient * learningRate
        self.outputGateB -= outputGateBGradient * learningRate
        self.shortTermMem = np.zeros((self.outputSize, 1))
        self.longTermMem = np.zeros((self.outputSize, 1))
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.LSTM",
            "inputSize":self.inputSize,
            "outputSize":self.outputSize,
            "W":[self.forgetGateW, self.updateGateW, self.outputActivationW, self.outputGateW],
            "U":[self.forgetGateU, self.updateGateU, self.outputActivaitonU, self.outputGateU],
            "B":[self.forgetGateB, self.updateGateB, self.outputActivationB, self.outputGateB]
        }
        return data
