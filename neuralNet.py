#this project is heavily inspire by the video: Neural Network from Scratch | Mathematics & Python Code by The Independent Code (https://www.youtube.com/watch?v=pauPCy_s0Ok)
#so class Layer(), class Dense(), and the activation classes will be virtually the same

#Note: D_ means deritative of, so for example D_tanh will mean the deritative of the tanh function
#Also this is NOT OPTIMIZED, its existance is merely a challenge and for fun

import numpy as np
from scipy.stats import norm
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

#identity
identity = lambda x: x
D_identity = lambda x: np.ones(np.shape(x))

#Binary Step
binStep = lambda x: np.greater(x, np.zeros(np.shape(x)))
D_binStep = lambda x: np.zeros(np.shape(x))

#hyperbolic tangent
tanh = lambda x: np.tanh(x)
D_tanh = lambda x: (1 - np.power(tanh(x), 2))
 
#Sigmoid activation
sigmoid = lambda x: 1/(1 + np.exp(-x))
D_sigmoid = lambda x: sigmoid(x)*(1 - sigmoid(x))

#ReLU
relu = lambda x: np.maximum(x, np.zeros(np.shape(x)))
D_relu = lambda x: np.greater(x, np.zeros(np.shape(x)))

#GELU
gelu = lambda x: x*norm.cdf(x)
D_gelu = lambda x: norm.cdf(x) + x*norm.pdf(x)

#ELU
elu = lambda x, alpha: (x>=np.zeros(np.shape(x)))*x + (x<np.zeros(np.shape(x)))*alpha*(np.exp(x)-1)
D_elu = lambda x, alpha: (x>=np.zeros(np.shape(x)))*1 + (x<np.zeros(np.shape(x)))*alpha*np.exp(x)

#SELU
selu = lambda x: 1.0507*elu(x, 1.67326)
D_selu = lambda x: 1.0507*D_elu(x, 167326)

#Leaky ReLu
lrelu = lambda x: np.maximum(x, 0.01*x)
D_lrelu = lambda x: (x>=0.01*x)*1 + (x<0.01*x)*0.01

#PReLU
prelu = lambda x, alpha: (x>=np.zeros(np.shape(x)))*x + (x<np.zeros(np.shape(x)))*alpha*x
D_prelu = lambda x, alpha: (x>=np.zeros(np.shape(x)))*1 + (x<np.zeros(np.shape(x)))*alpha

#SiLU
silu = lambda x: x*sigmoid(x)
D_silu = lambda x: sigmoid(x) + x*D_sigmoid(x)

#Gaussian
gaussian = lambda x: np.exp(-np.power(x, 2))
D_gaussian = lambda x: -2*x*gaussian(x)

#Softplus
softplus = lambda x: np.log(1 + np.exp(x))
D_softplus = lambda x: 1/(1 + np.exp(-x))

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

#GRU
#experimental (will break)
class GRU(Layer):
    def __init__(self, inputSize, outputSize, internalSize, internalFunc, D_internalFunc, trimFunc):
        self.externalWeights = np.random.randn(outputSize, inputSize)
        self.internalWeights = np.random.randn(outputSize, internalSize)
        self.bias = np.random.randn(outputSize, internalSize)
        self.value = np.random.randn(outputSize, 1)
        self.internalFunc = internalFunc
        self.D_internalFunc = D_internalFunc
        self.trimFunc = trimFunc
    
    def forward(self, input):
        self.input = input
        self.value = np.dot(self.externalWeights, self.input) + self.internalFunc(self.internalWeights, self.value, self.bias)
        return self.value
    
    def backward(self, outputGradient, learningRate):
        externalWeightsGradient = np.dot(outputGradient, self.input.T)
        inputGradient, internalWeightsGradient, biasGradient = self.D_internalFunc(outputGradient, self.externalWeights, self.internalWeights, self.value, self.bias)
        self.externalWeights -=learningRate*externalWeightsGradient
        self.internalWeights -=learningRate*internalWeightsGradient
        self.bias -=learningRate*biasGradient
        self.internalWeights = self.trimFunc(self.internalWeights)
        return inputGradient

#This is the newer version, a direct child from class GRU
#Simplistic GRU, it is a GRU but with one internal weight and one internal activation (tanh)
class SGRU(GRU):
    def __init__(self, inputSize, outputSize):
        internalSize = 1
        def internalFunc(internalWeights, value, bias):
            return np.multiply(self.internalWeights, self.value + self.bias)
        def D_internalFunc(outputGradient, externalWeights, internalWeights, value, bias):
            return np.dot(externalWeights.T, outputGradient), np.multiply(outputGradient, self.value), outputGradient
        super().__init__(inputSize, outputSize, internalSize, internalFunc, D_internalFunc, tanh)

#Simplistic GRU, it is a GRU but with one internal weight and one internal activation (tanh)
#update: this has been remaned to OSGRU, this is the original version, a newer version has been made
class OSGRU(Layer):
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
    def __init__(self, activation, D_activation, outputSize):
        self.activation = activation
        self.D_activation = D_activation
        self.outputSize = outputSize
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, outputGradient, learningRate):
        return np.multiply(outputGradient, self.D_activation(self.input))

#Identidy activation layer
class Identity(Activation):
    def __init__(self, outputSize):
        super().__init__(identity, D_identity, outputSize)

#Binary Step activation layer
class BinaryStep(Activation):
    def __init__(self, outputSize):
        super().__init__(binStep, D_binStep, outputSize)

#hyperbolic tangent activation layer
class Tanh(Activation):
    def __init__(self, outputSize):
        super().__init__(tanh, D_tanh, outputSize)

#sigmoid activation layer
class Sigmoid(Activation):
    def __init__(self, outputSize):
        super().__init__(sigmoid, D_sigmoid, outputSize)

#ReLu activation layer
class ReLU(Activation):
    def __init__(self, outputSize):
        super().__init__(relu, D_relu, outputSize)

#GELU activation layer
class GELU(Activation):
    def __init__(self, outputSize):
        super().__init__(gelu, D_gelu, outputSize)

#ELU
class ELU(Activation):
    def __init__(self, outputSize):
        self.alpha = np.random.randn(1)[0]
        Elu = lambda x: elu(x, self.alpha)
        D_Elu = lambda x: D_elu(x, self.alpha)
        super().__init__(Elu, D_Elu, outputSize)

    def backward(self, outputGradient, learningRate):
        self.alpha -= (self.input<np.zeros(np.shape(self.input)))*learningRate*outputGradient*(np.exp(self.input)-1)
        return np.multiply(outputGradient, self.D_activation(self.input))

#SELU
class SELU(Activation):
    def __init__(self, outputSize):
        super().__init__(selu, D_selu, outputSize)

#Leaky ReLU
class LeakyReLU(Activation):
    def __init__(self, oututSize):
        super().__init__(lrelu, D_lrelu, outputSize)

#PReLU
class PReLU(Activation):
    def __init__(self, outputSize):
        self.alpha = np.random.randn(1)[0]
        Prelu = lambda x: prelu(x, self.alpha)
        D_Prelu = lambda x:D_prelu(x, self.alpha)
        super().__init__(Prelu, D_Prelu, outputSize)
    
    def backward(self, outputGradient, learningRate):
        self.alpha -= (self.input<snp.zeros(np.shape(self.input)))*leaningRate*outputGradient*self.input
        return np.multiply(outputGradient, self.D_activation(self.input))

#SiLU
class SiLU(Activation):
    def __init__(self, outputSize):
        super().__init__(silu, D_silu, outputSize)

#Softplus activation layer
class Softplus(Activation):
    def __init__(self, outputSize):
        super().__init__(softplus, D_softplus, outputSize)

#Gaussian
class Gaussian(Activation):
    def __init__(self, outputSize):
        super().__init__(gaussian, D_gaussian, outputSize)

#softmax activation
class SoftMax(Layer):
    def __init__(self, outputSize):
        self.outputSize = outputSize

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
    
    def forward(self, x):
        output = x
        for layer in self.network: output = layer.forward(output)
        return output
    
    def train(self, X, Y, epochs, learningRate, ErrorFunc='MSE', test=True, testPercentage=0.9):
        #split test and train data
        if test:
            trainX = X[:int(testPercentage*len(X))]
            testX = X[int(testPercentage*len(X)):]
            trainY = Y[:int(testPercentage*len(Y))]
            testY = Y[int(testPercentage*len(Y)):]
        else:
            trainX = X
            trainY = Y
        #error functions
        errorFunc = Errorfunctions[ErrorFunc]
        D_errorFunc = D_Errorfunctions[ErrorFunc]
        #stores to plot
        trainingErrorPlot = ()
        testingErrorPlot = ()
        #starts timeing
        startTime = time.time()
        #loops through each epoch
        for e in range(epochs):
            #resets error
            error = 0
            #loops through traingin data
            for x, y in zip(trainX,trainY):
                #forward
                output = x
                for layer in self.network: output = layer.forward(output)
                error += errorFunc(y, output)
                #backward
                gradient = D_errorFunc(y, output)
                for layer in reversed(self.network): gradient = layer.backward(gradient, learningRate)
            #normalize error
            error /= len(X)
            #prints error
            print(f'Epoch {e+1}/{epochs}, training error: {error}, training accuracy: {1-error}')
            #stores error
            trainingErrorPlot += (error,)
            if test:
                #resets error
                error = 0
                #loops through testing data
                for x, y in zip(testX, testY):
                    #forward
                    output = x
                    for layer in self.network:output = layer.forward(output)
                    error += errorFunc(y, output)
                    #no backwards for test
                #normalizes error
                error /= len(X)
                #prints error
                print(f'Epoch {e+1}/{epochs}, testing error: {error}, testing accuracy: {1-error}')
                #stores error
                testingErrorPlot += (error,)
        #stop timing
        endTime = time.time()
        #prints time
        print(f'Time spent: {endTime - startTime} seconds')
        #plots
        plt.plot(np.linspace(0, epochs+1, num=epochs+1), (1,)+trainingErrorPlot, label="Training Error")
        plt.plot(np.linspace(0, epochs+1, num=epochs+1), [0]+[1-_ for _ in trainingErrorPlot], label="Training Accuracy")
        if test:
            plt.plot(np.linspace(0, epochs+1, num=epochs+1), (1,)+testingErrorPlot, label="Testing Error")
            plt.plot(np.linspace(0, epochs+1, num=epochs+1), [0]+[1-_ for _ in testingErrorPlot], label="Testing Accuracy")
        if test: plt.legend(["Training Error", "Training Accuracy", "Testing Error", "Testing Accuracy"])
        else: plt.legend(["Training Error", "Training Accuracy"])
        plt.xlabel("Epochs")
        plt.show()
    
    def visualize(self, X, Y, optionsDisplay=None):
        #set to default
        if not optionsDisplay: optionsDisplay = ['Neuron ' + str(i) for i in range(self.network[-1].outputSize)]
        #loops through visualizing data
        for x, y in zip(X, Y):
            #forward
            output = x
            for layer in self.network: output = layer.forward(output)
            error = MSE(y, output)
            #no backward
            #for each looping, rearranges the neurons to that the one with the highest activation is on top
            sortedOutput = sorted(zip(optionsDisplay, output), key = (lambda x: x[1]), reverse=True)
            #prints
            print(f'Input: {[list(_) for _ in x]}')
            for j in range(len(sortedOutput)): print(f'Output: {sortedOutput[j][0]} ({sortedOutput[j][1][0]})\tExpected: {float(y[j][0])}')
            print('-------------------')

#Recurrent Neural Networks
class RNN():
    def __init__(self, network):
        self.network = network
    
    def train(self, X, Y, epochs, learningRate, ErrorFunc='MSE', test=True, testPercentage=0.9):
        #extracts test and train data
        if test:
            trainX = X[:int(testPercentage*len(X))]
            testX = X[int(testPercentage*len(X)):]
            trainY = Y[:int(testPercentage*len(Y))]
            testY = Y[int(testPercentage*len(Y)):]
        else:
            trainX = X
            trainY = Y
        #error function
        errorFunc = Errorfunctions[ErrorFunc]
        D_errorFunc = D_Errorfunctions[ErrorFunc]
        #stores errors to plot
        trainErrorPlot = ()
        testErrorPlot = ()
        #starts timing
        startTime = time.time()
        for e in range(epochs):
            #loops through each epochs
            #resets error
            error = 0
            #loops through each training data
            for x, y in zip(trainX, trainY):
                #loops through each timestep of the data
                for t in range(len(x)):
                    #forward
                    output = x[t]
                    for layer in self.network: output = layer.forward(output)
                    error += errorFunc(y[t], output)
                    #backwards
                    gradient = D_errorFunc(y[t], output)
                    for layer in reversed(self.network): gradient = layer.backward(gradient, learningRate)
            #normalizes error
            error /= (len(trainX)*len(trainX[0]))
            #stores error
            trainErrorPlot += (error,)
            #prints error
            print(f'Epoch {e+1}/{epochs}, training error: {error}, training accuracy: {1-error}')
            if test:
                #resets it
                error = 0
                #loops through each testing data
                for x, y in zip(testX, testY):
                    #loops through each timestep of the data
                    for t in range(len(x)):
                        #forward
                        output = x[t]
                        for layer in self.network: output = layer.forward(output)
                        error += errorFunc(y[t], output)
                        #no backward for test
                #normalizes error
                error /= (len(testX)*len(testX[0]))
                #stores error
                testErrorPlot += (error, )
                #prints error
                print(f'Epoch {e+1}/{epochs}, testing error: {error}, testing accuracy: {1-error}')
        #stops timing
        endTime = time.time()
        #prints time
        print(f'Time spent: {endTime - startTime} seconds')
        #plots
        plt.plot(np.linspace(0, epochs+1, num=epochs+1), (1,)+trainErrorPlot, label="Training Error")
        plt.plot(np.linspace(0, epochs+1, num=epochs+1), [0]+[1-_ for _ in trainErrorPlot], label="Training Accuracy")
        if test:
            plt.plot(np.linspace(0, epochs+1, num=epochs+1), (1,)+testErrorPlot, label="Testing Error")
            plt.plot(np.linspace(0, epochs+1, num=epochs+1), [0]+[1-_ for _ in testErrorPlot], label="Testing Accuracy")
        if test: plt.legend(["Training Error", "Training Accuracy", "Testing Error", "Testing Accuracy"])
        else: plt.legend(["Training Error", "Training Accuracy"])
        plt.xlabel("Epochs")
        plt.show()
