#this project is heavily inspire by the video: Neural Network from Scratch | Mathematics & Python Code by The Independent Code (https://www.youtube.com/watch?v=pauPCy_s0Ok)
#so class Layer(), class Dense(), and the activation classes will be virtually the same

#Note: D_ means deritative of, so for example D_tanh will mean the deritative of the tanh function
#Also this is NOT OPTIMIZED, its existance is merely a challenge and for fun

import numpy as np
import matplotlib.pyplot as plt
import time

#Mean Squared Error
def MSE(yTrue, y): return np.mean(np.power(yTrue - y, 2))
def D_MSE(yTrue, y): return 2*(y - yTrue)/np.size(yTrue)

#Cross-Entropy Loss
def CEL(yTrue, y): return -np.mean(np.multiply(yTrue, np.log(y)))
def D_CEL(yTrue, y): return -yTrue/(y*np.size(yTrue))

Errorfunctions = {'CEL':CEL, 'MSE':MSE}
D_Errorfunctions = {'CEL':D_CEL, 'MSE':D_MSE}

#hyperbolic tangent
tanh = lambda x: np.tanh(x)
D_tanh = lambda x: (1 - np.power(tanh(x), 2))
 
#Sigmoid activation
sigmoid = lambda x: 1/(1 + np.exp(-x))
D_sigmoid = lambda x: sigmoid(x)*(1 - sigmoid(x))

#ReLU
relu = lambda x: np.maximum(x, np.zeros(np.shape(x)))
D_relu = lambda x: np.greater(x, np.zeros(np.shape(x)))

#base class of all the layers in the neural network
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, outputGradient, learningRate):
        pass

#Dense, or Linear layer
class Dense(Layer):
    def __init__(self, inputSize, outputSize, wantBias=True):
        self.weights = np.random.randn(outputSize, inputSize)
        if wantBias: self.bias = np.random.randn(outputSize, 1)
        self.wantBias = wantBias
    
    def forward(self, input):
        self.input = input
        if self.wantBias: return np.dot(self.weights, self.input)  + self.bias
        else: return np.dot(self.weights, self.input)
    
    def backward(self, outputGradient, learningRate):
        weightsGradient = np.dot(outputGradient, self.input.T)
        inputGradient = np.dot(self.weights.T, outputGradient)
        self.weights -= learningRate*weightsGradient
        if self.wantBias: self.bias -=learningRate*outputGradient
        return inputGradient

#Simplistic GRU, it is a GRU but with one internal weight and one internal activation (tanh)
class SGRU(Layer):
    def __init__(self, inputSize, outputSize):
        self.externalWeights = np.random.randn(outputSize, inputSize)
        self.internalWeights = np.random.randn(outputSize, 1)
        self.bias = np.random.randn(outputSize, 1)
        self.value = np.random.randn(outputSize, 1)
    
    def forward(self, input):
        self.input = input
        self.value = np.dot(self.externalWeights, self.input) + np.multiply(self.internalWeights, self.value) + self.bias
        return self.value
    
    def backward(self, outputGradient, learningRate):
        externalWeightsGradient = np.dot(outputGradient, self.input.T)
        internalWeightsGradient = np.multiply(outputGradient, self.value)
        inputGradient = np.dot(self.externalWeights.T, outputGradient)
        self.externalWeights -=learningRate*externalWeightsGradient
        self.internalWeights -=learningRate*internalWeightsGradient
        self.bias -=learningRate*outputGradient
        self.internalWeights = tanh(self.internalWeights)
        return inputGradient

#Long Short Term Memory, still in progress
#I do not guarentee there will be no bugs
class LSTM(Layer): 
    def __init__(self, inputSize, outputSize):
        self.forgetGateW = np.random.randn(outputSize, 1)
        self.forgetGateU = np.random.randn(outputSize, inputSize)
        self.updateGateW = np.random.randn(outputSize, 1)
        self.updateGateU = np.random.randn(outputSize, inputSize)
        self.outputActivationW = np.random.randn(outputSize, 1)
        self.outputActivationU = np.random.randn(outputSize, inputSize)
        self.outputGateW = np.random.randn(outputSize, 1)
        self.outputGateU = np.random.randn(outputSize, inputSize)
        self.forgetGateB = np.random.randn(outputSize, 1)
        self.updateGateB = np.random.randn(outputSize, 1)
        self.outputActivationB = np.random.randn(outputSize, 1)
        self.outputGateB = np.random.randn(outputSize, 1)
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
        return inputGradient

#activation base class (softmax excluded)
class Activation(Layer):
    def __init__(self, activation, D_activation):
        self.activation = activation
        self.D_activation = D_activation
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, outputGradient, learningRate):
        return np.multiply(outputGradient, self.D_activation(self.input))

#hyperbolic tangent activation layer
class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, D_tanh)

#sigmoid activation layer
class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, D_sigmoid)

#ReLu activation
class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, D_relu)

#softmax activation
class SoftMax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, outputGradient, learningRate):
        n = np.size(self.output)
        M = np.tile(self.output, n)
        return np.dot(M * (np.identity(n) - np.transpose(M)), outputGradient)

#Feed Forward Neural Networks
class FFNN():
    def __init__(self, network):
        self.network = network
    
    def train(self, X, YTrue, epochs, learningRate, ErrorFunc='MSE'):
        errorFunc = Errorfunctions[ErrorFunc]
        D_errorFunc = D_Errorfunctions[ErrorFunc]
        startTime = time.time()
        errorPlot = ()
        for e in range(epochs):
            error = 0
            for x, yTrue in zip(X,YTrue):
                output = x
                for layer in self.network: output = layer.forward(output)
                error += errorFunc(yTrue, output)
                gradient = D_errorFunc(yTrue, output)
                for layer in reversed(self.network): gradient = layer.backward(gradient, learningRate)
            error /= len(X)
            print(f'Epoch {e+1}/{epochs}, error: {error}, accuracy: {1-error}')
        endTime = time.time()
        print(f'Time spent: {endTime - startTime} seconds')
        ErrorLine = plt.plot(np.linspace(0, epochs+1, num=epochs+1), (1,)+errorPlot, label="Error")
        AccuracyLine = plt.plot(np.linspace(0, epochs+1, num=epochs+1), [0]+[1-_ for _ in errorPlot], label="Accuracy")
        plt.legend(["Error", "Accuracy"])
        plt.xlabel("Epochs")
        plt.show()
    
    def test(self, X, YTrue, optionsDislay):
        for x, yTrue in zip(X, YTrue):
            output = x
            for layer in self.network: output = layer.forward(output)
            error = MSE(yTrue, output)
            sortedOuput = sorted([(optionsDislay[_], output[_]) for _ in range(len(output))], key = (lambda x: x[1]), reverse=True)
            print(f'Input: {[list(_) for _ in x]}')
            for y in range(len(sortedOuput)): print(f'Output: {sortedOuput[y][0]} ({sortedOuput[y][1]})\tExpected: {float(yTrue[sortedOuput[y][0]])}')
            print('-------------------')

#Recurrent Neural Networks
#Still in progress, the test() function needs to be coded
class RNN():
    def __init__(self, network):
        self.network = network
    
    def train(self, X, YTrue, epochs, learningRate, ErrorFunc='MSE'):
        errorFunc = Errorfunctions[ErrorFunc]
        D_errorFunc = D_Errorfunctions[ErrorFunc]
        startTime = time.time()
        errorPlot = ()
        for e in range(epochs):
            error = 0
            for x, yTrue in zip(X, YTrue):
                for t in range(len(x)):
                    output = x[t]
                    for layer in self.network: output = layer.forward(output)
                    error += errorFunc(yTrue[t], output)
                    gradient = D_errorFunc(yTrue[t], output)
                    for layer in reversed(self.network): gradient = layer.backward(gradient, learningRate)
            error /= (len(X)*len(X[0]))
            errorPlot += (error,)
            print(f'Epoch {e+1}/{epochs}, error: {error}, accuracy: {1-error}')
        endTime = time.time()
        print(f'Time spent: {endTime - startTime} seconds')
        ErrorLine = plt.plot(np.linspace(0, epochs+1, num=epochs+1), (1,)+errorPlot, label="Error")
        AccuracyLine = plt.plot(np.linspace(0, epochs+1, num=epochs+1), [0]+[1-_ for _ in errorPlot], label="Accuracy")
        plt.legend(["Error", "Accuracy"])
        plt.xlabel("Epochs")
        plt.show()
