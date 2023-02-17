#this project is heavily inspire by the video: Neural Network from Scratch | Mathematics & Python Code by The Independent Code (https://www.youtube.com/watch?v=pauPCy_s0Ok)
#so class Layer(), class Dense(), convolution classes and the activation classes will be virtually the same

#Note: D_ means deritative of, so for example D_tanh will mean the deritative of the tanh function
#Also this is NOT OPTIMIZED, its existance is merely a challenge and for fun

import numpy as np
from scipy.stats import norm
from scipy import signal
import matplotlib.pyplot as plt

import time
import json
import os
import base64

#Mean Squared Error
def MSE(yTrue, y): return np.mean(np.power(yTrue - y, 2))
def D_MSE(yTrue, y): return 2*(y - yTrue)/np.size(yTrue)

#Cross-Entropy Loss
def CEL(yTrue, y): return -np.mean(np.multiply(yTrue, np.log(y)))
def D_CEL(yTrue, y): return -yTrue/(y*np.size(yTrue))

def BCEL(yTrue, y): return -np.mean(np.multiply(yTrue, np.log(y)) + np.multiply(1 - yTrue, np.log(1 - y)))
def D_BCEL(yTrue, y): return ((1 - yTrue)/(1 - y) - yTrue/y)/np.size(yTrue)

Errorfunctions = {'CEL':CEL, 'MSE':MSE, 'BCEL':BCEL}
D_Errorfunctions = {'CEL':D_CEL, 'MSE':D_MSE, 'BCEL':D_BCEL}

#identity
identity = lambda x: x
D_identity = lambda x: 1

#Binary Step
binStep = lambda x: np.greater(x, 0)
D_binStep = lambda x: 0

#hyperbolic tangent
tanh = lambda x: np.tanh(x)
D_tanh = lambda x: (1 - np.power(tanh(x), 2))
 
#Sigmoid activation
sigmoid = lambda x: 1/(1 + np.exp(-x))
D_sigmoid = lambda x: sigmoid(x)*(1 - sigmoid(x))

#ReLU
relu = lambda x: np.maximum(x, 0)
D_relu = lambda x: np.greater(x, 0)

#GELU
gelu = lambda x: x*norm.cdf(x)
D_gelu = lambda x: norm.cdf(x) + x*norm.pdf(x)

#ELU
elu = lambda x, alpha: (x>=0)*x + (x<0)*alpha*(np.exp(x)-1)
D_elu = lambda x, alpha: (x>=0)*1 + (x<0)*alpha*np.exp(x)

#SELU
selu = lambda x: 1.0507*elu(x, 1.67326)
D_selu = lambda x: 1.0507*D_elu(x, 167326)

#Leaky ReLu
lrelu = lambda x: np.maximum(x, 0.01*x)
D_lrelu = lambda x: (x>=0.01*x)*1 + (x<0.01*x)*0.01

#PReLU
prelu = lambda x, alpha: (x>=0)*x + (x<0)*alpha*x
D_prelu = lambda x, alpha: (x>=0)*1 + (x<0)*alpha

#SiLU
silu = lambda x: x*sigmoid(x)
D_silu = lambda x: sigmoid(x) + x*D_sigmoid(x)

#Gaussian
gaussian = lambda x: np.exp(-np.power(x, 2))
D_gaussian = lambda x: -2*x*gaussian(x)

#Softplus
softplus = lambda x: np.log(1 + np.exp(x))
D_softplus = lambda x: 1/(1 + np.exp(-x))

class FileNameError(Exception):
    def __init__(self, filename):
        super().__init__(f'File has to be a .json file, not .{filename.split(".")[-1]} as in {filename}')

class jsonSpecialEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#reference: https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array/24375113#24375113
def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

#base class of all the layers in the neural network
class Layer():
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass

    def backward(self, outputGradient, learningRate):
        pass
    
    def save(self, filename):
        pass

#Dense, or Linear layer
class Dense(Layer):
    def __init__(self, inputSize, outputSize, wantBias=True, initWeights=None, initBias=None):
        self.inputSize, self.outpuSize = inputSize, outputSize
        if initWeights is None: self.weights = np.random.randn(outputSize, inputSize)
        else: self.weights = initWeights
        if wantBias: 
            if initBias is None: self.bias = np.random.randn(outputSize, 1)
            else: self.bias = initBias
        else: self.bias = None
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
    
    def save(self):
        data = {
            "Type":"neuralNet.Dense",
            "inputSize":self.inputSize,
            "outputSize":self.outpuSize,
            "weights":self.weights,
            "bias":self.bias,
            "wantBias":self.wantBias
        }
        return data

#GRU (Not available)

#Simplistic GRU, it is a GRU but with one internal weight and one internal activation (tanh)
class SGRU(Layer):
    def __init__(self, inputSize, outputSize, externalWeights=None, internalWeights=None, bias=None):
        self.inputSize, self.outputSize = inputSize, outputSize
        if externalWeights is None: self.externalWeights = np.random.randn(outputSize, inputSize)
        else: self.externalWeights = externalWeights
        if internalWeights is None: self.internalWeights = np.random.randn(outputSize, 1)
        else: self.internalWeights = internalWeights
        if bias is None: self.bias = np.random.randn(outputSize, 1)
        else: self.bias = bias
        self.value = np.zeros((outputSize, 1))
    
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
        self.value = np.zeros((self.outputSize, 1))
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.SGRU",
            "inputSize":self.inputSize,
            "outputSize":self.outputSize,
            "internalWeights":self.internalWeights,
            "externalWeights":self.externalWeights,
            "bias":self.bias
        }
        return data

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

#Convolution without stride
class RawConvolute(Layer):
    def __init__(self, inputShape, kernalSize, depth, padding, kernal=None, bias=None):
        self.padding = padding
        self.kernalSize = kernalSize
        self.inputShape = self.inputDepth, inputHeight, inputWidth = inputShape
        self.outputShape = self.depth, self.outputHeight, self.outputWidth = (depth, inputHeight - kernalSize + 1 + 2*self.padding, inputWidth - kernalSize + 1 + 2*self.padding)
        self.kernalShape = (depth, self.inputDepth, kernalSize, kernalSize)
        if kernal is None: self.kernal = np.random.randn(*self.kernalShape)
        else: self.kernal = kernal
        if bias is None: self.bias = np.random.randn(*self.outputShape)
        else: self.bias = bias
    
    def forward(self, input):
        self.input = input
        self.inputPad = np.pad(self.input, ((0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        self.output = np.copy(self.bias)
        for i in range(self.depth):
            for j in range(self.inputDepth):
                self.output[i] += signal.correlate2d(self.inputPad[j], self.kernal[i, j], "valid")
        return self.output
    
    def backward(self, outputGradient, learningRate):
        #outputGradient = outputGradient[:, self.padding:self.outputHeight-self.padding, self.padding:self.outputWidth-self.padding]
        kernalGradient = np.zeros(self.kernalShape)
        inputGradient = np.zeros(self.inputShape)
        for i in range(self.depth):
            for j in range(self.inputDepth):
                kernalGradient[i, j] = signal.correlate2d(self.inputPad[j], outputGradient[i], "valid")
                inputGradient[j] = signal.convolve(outputGradient[i, self.padding:self.outputHeight-self.padding, self.padding:self.outputWidth-self.padding], self.kernal[i, j], "full")
        self.kernal -= learningRate*kernalGradient
        self.bias -= learningRate*outputGradient
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.RawConvolute",
            "inputShape":self.inputShape,
            "kernalSize":self.kernalSize,
            "depth":self.depth,
            "padding":self.padding,
            "kernal":self.kernal,
            "bias":self.bias
        }
        return data

#Stride layer, can be treated as a pooling layer that extract the top left corner, with pool size of stride and stride of stride
class Stride(Layer):
    def __init__(self, inputShape, stride):
        self.inputShape = self.inputDepth, self.inputHeight, self.inputWidth = inputShape
        self.stride = stride
        self.outputshape = self.inputDepth, round(self.inputHeight/self.stride), round(self.inputWidth, self.stride)
    
    def forward(self, input):
        self.input = input
        return self.input[:, ::self.stride, ::self.stride]
    
    def backward(self, outputGradient, learningRate):
        inputGradient = np.zeros(self.inputShape)
        inputGradient[:, ::self.stride, ::self.stride] = outputGradient
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.Stride",
            "inputShape":self.inputShape,
            "stride":self.stride
        }
        return data

#convolution layer, with stride
class Convolute(Layer):
    def __init__(self, inputShape, kernalSize, depth, padding, stride, kernal=None, bias=None):
        self.convLayer = RawConvolute(inputShape, kernalSize, depth, padding, kernal=kernal, bias=bias)
        self.strideLayer = Stride(self.convLayer.outputShape, stride)
        self.outputShape = self.strideLayer.outputshape
    
    def forward(self, input):
        self.input = input
        return self.strideLayer.forward(self.convLayer.forward(self.input))
    
    def backward(self, outputGradient, learningRate):
        return self.convLayer.backward(self.strideLayer.backward(outputGradient, learningRate), learningRate)
    
    def save(self):
        data = {
            "Type":"neuralNet.Convolute",
            "inputShape":self.convLayer.inputShape,
            "kernalSize":self.convLayer.kernalSize,
            "depth":self.convLayer.depth,
            "padding":self.convLayer.padding,
            "kernal":self.convLayer.kernal,
            "bias":self.convLayer.bias,
            "stride":self.strideLayer.stride
        }
        return data

#Max pooling layer
class MaxPool(Layer):
    def __init__(self, inputShape, poolShape=(2,2), stride=2):
        self.inputShape = self.inputDepth, self.inputHeight, self.inputWidth = inputShape
        self.poolShape = self.poolHeight, self.poolWidth = poolShape
        self.stride = stride
        self.outputShape = self.outputDepth, self.outputHeight, self.outputWidth = (self.inputDepth, 1 + (self.inputHeight - self.poolHeight)//self.stride, 1 + (self.inputWidth - self.poolWidth)//self.stride)

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.outputShape)
        for i in range(self.inputDepth):
            for j in range(self.outputHeight):
                for k in range(self.outputWidth):
                    self.output[i,j,k] = np.max(self.input[i, self.stride*j:self.poolHeight + self.stride*j, self.stride*k:self.poolWidth + self.stride*k])
        return self.output
    
    def backward(self, outputGradient, learningRate):
        inputGradient = np.zeros(self.inputShape)
        for i in range(self.inputDepth):
            for j in range(self.outputHeight):
                for k in range(self.outputWidth):
                    inputGradient[i, self.stride*j:self.poolHeight + self.stride*j, self.stride*k:self.poolWidth + self.stride*k] += (self.input[i, self.stride*j:self.poolHeight + self.stride*j, self.stride*k:self.poolWidth + self.stride*k]==self.output[i, j, k])*outputGradient[i, j, k]
        return inputGradient
    
    def save(self):
        data = {
            "Type":"neuralNet.MaxPool",
            "inputShape":self.inputShape,
            "poolShape":self.poolShape,
            "stride":self.stride
        }
        return data

#Average pooling layer
class AvgPool(Layer):
    def __init__(self, inputShape, poolShape=(2,2), stride=2):
        self.inputShape = self.inputDepth, self.inputHeight, self.inputWidth = inputShape
        self.poolShape = self.poolHeight, self.poolWidth = poolShape
        self.stride = stride
        self.outputShape = self.outputDepth, self.outputHeight, self.outputWidth = (self.inputDepth, 1 + (self.inputHeight - self.poolHeight)//self.stride, 1 + (self.inputWidth - self.poolWidth)//self.stride)
        self.kernal = np.ones(self.poolShape)/np.prod(self.poolShape)
    
    def forward(self, input):
        self.input = input
        inputPad = np.pad(self.input, ((0,0), (0, (self.inputHeight - self.poolHeight)%self.stride), (0, (self.inputWidth - self.poolWidth)%self.stride)))
        self.output = np.zeros(self.outputShape)
        for i in range(self.inputDepth):
            self.output[i] = signal.convolve(inputPad[i], self.kernal, "valid")[::self.stride, ::self.stride]
        return self.output
    
    def backward(self, outputGradient, learningRate):
        inputGradient = np.zeros(self.inputShape)
        for i in range(self.inputDepth):
            for j in range(self.outputHeight):
                for k in range(self.outputWidth):
                   inputGradient[i, j*self.stride:j*self.stride + self.poolHeight, k*self.stride:k*self.stride + self.poolWidth] += outputGradient[i, j, k]*self.kernal
        return inputGradient

    def save(self):
        data = {
            "Type":"neuralNet.AvgPool",
            "inputShape":self.inputShape,
            "poolShape":self.poolShape,
            "stride":self.stride
        }
        return data

#Reshapes the output of the previous layer to the input of the next layer
class Reshape(Layer):
    def __init__(self, inputShape, outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape
    
    def forward(self, input):
        self.input = input
        return np.reshape(self.input, self.outputShape)
    
    def backward(self, outputGradient, learningRate):
        return np.reshape(outputGradient, self.inputShape)

    def save(self):
        data = {
            "Type":"neuralNet.Reshape",
            "inputShape":self.inputShape,
            "outputShape":self.outputShape
        }
        return data

#flattens the output of the previous layer
class Flatten(Reshape):
    def __init__(self, inputShape):
        self.inputShape = inputShape
        outputShape = (np.prod(inputShape), 1)
        super().__init__(inputShape, outputShape)

    def save(self):
        data = {
            "Type":"neuralNet.Flatten",
            "inputshape":self.inputShape
        }
        return data

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
    
    def save(self):
        pass

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

#Binary Step activation layer
class BinaryStep(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(binStep, D_binStep, outputSize)
    
    def save(self):
        data = {
            "Type":"neuralNet.BinaryStep",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

#hyperbolic tangent activation layer
class Tanh(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(tanh, D_tanh, outputSize)
    
    def save(self):
        data = {
            "Type":"neuralNet.Tanh",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

#sigmoid activation layer
class Sigmoid(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(sigmoid, D_sigmoid, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.Sigmoid",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

#ReLu activation layer
class ReLU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(relu, D_relu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.ReLU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

#GELU activation layer
class GELU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(gelu, D_gelu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.GELU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

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

#Leaky ReLU
class LeakyReLU(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(lrelu, D_lrelu, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.LeakyReLU",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

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

#Gaussian
class Gaussian(Activation):
    def __init__(self, outputSize):
        self.outputSize = outputSize
        super().__init__(gaussian, D_gaussian, outputSize)

    def save(self):
        data = {
            "Type":"neuralNet.Gaussian",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

#softmax activation
class SoftMax(Layer):
    def __init__(self, outputSize):
        self.outputSize = outputSize

    def forward(self, input):
        self.input = input
        tmp = np.exp(self.input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, outputGradient, learningRate):
        n = np.size(self.output)
        M = np.tile(self.output, n)
        return np.dot(M * (np.identity(n) - np.transpose(M)), outputGradient)

    def save(self):
        data = {
            "Type":"neuralNet.SoftMax",
            "outputSize":self.outputSize,
            "alpha":None
        }
        return data

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
            error /= len(trainX)
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
                error /= len(testX)
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

    def trainWithTest(self, trainX, trainY, testX, testY, epochs, learningRate, ErrorFunc='MSE'):
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
            error /= len(trainX)
            #prints error
            print(f'Epoch {e+1}/{epochs}, training error: {error}, training accuracy: {1-error}')
            #stores error
            trainingErrorPlot += (error,)
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
            error /= len(testX)
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
        plt.plot(np.linspace(0, epochs+1, num=epochs+1), (1,)+testingErrorPlot, label="Testing Error")
        plt.plot(np.linspace(0, epochs+1, num=epochs+1), [0]+[1-_ for _ in testingErrorPlot], label="Testing Accuracy")
        plt.legend(["Training Error", "Training Accuracy", "Testing Error", "Testing Accuracy"])
        plt.xlabel("Epochs")
        plt.show()
    
    def visualize(self, X, Y, optionsDisplay=None, displayInput=True):
        #set to default
        if optionsDisplay is None: optionsDisplay = ['Neuron ' + str(i) for i in range(self.network[-1].outputSize)]
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
            if displayInput: print(f'Input: {[list(_) for _ in x]}')
            for j in range(len(sortedOutput)): print(f'Output: {sortedOutput[j][0]} ({sortedOutput[j][1][0]})\tExpected: {float(y[j][0])}')
            print('-------------------')
    
    def save(self, fp):
        if fp.split('.')[-1] != 'json': raise FileNameError(fp)
        data = {"Type":"neuralNet.FFNN", "LayerNum":len(self.network)}
        print('Extracting and Saving Data...')
        startTime = time.time()
        for l in range(len(self.network)): data['Layer ' + str(l)] = self.network[l].save()
        with open(fp, "w") as f: json.dump(data, f, cls=jsonSpecialEncoder, indent=4)
        endTime = time.time()
        print(f'Finished extracting and saving data, took {endTime - startTime} seconds')

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

    def save(self, fp):
        if fp.split('.')[-1] != 'json': raise FileNameError(fp)
        data = {"Type":"neuralNet.RNN", "LayerNum":len(self.network)}
        print('Extracting and Saving Data...')
        startTime = time.time()
        for l in range(len(self.network)): data['Layer ' + str(l)] = self.network[l].save()
        with open(fp, "w") as f: json.dump(data, f, cls=jsonSpecialEncoder, indent=4)
        endTime = time.time()
        print(f'Finished extracting and saving data, took {endTime - startTime} seconds')


def loadNeuralNet(fp):
    if not os.path.exists(fp): raise FileNotFoundError(f'{fp} doesn\'t exist')
    with open(fp, 'r') as f: nNData = json.load(f, object_hook=json_numpy_obj_hook)
    network = []
    nNType = nNData["Type"]
    nNLen = nNData["LayerNum"]
    for l in range(nNLen):
        layer = nNData["Layer " + str(l)]
        layerType = layer["Type"]
        if layer["Type"] == "neuralNet.Dense": network.append(Dense(layer["inputSize"], layer["outputSize"], wantBias=layer["wantBias"], initWeights=layer["weights"], initBias=layer["bias"]))
        elif layer["Type"] == "neuralNet.SGRU": network.append(SGRU(layer["inputSize"], layer["outputSize"], externalWeights=layer["externalWeights"], internalWeights=layer["internalWeights"], bias=layer["bias"]))
        elif layer["Type"] == "neuralNet.LSTM": network.append(LSTM(layer["inputSize"], layer["outputSize"], W=layer["W"], U=layer["U"], B=layer["B"]))
        elif layer["Type"] == "neuralNet.RawConvolute": network.append(RawConvolute(layer["inputShape"], layer["kernalSize"], layer["depth"], layer["padding"], kernal=layer["kernal"], bias=layer["bias"]))
        elif layer["Type"] == "neuralNet.Stride": network.append(Stride(layer["inputShape"], layer["stride"]))
        elif layer["Type"] == "neuralNet.Convolute": network.append(Convolute(layer["inputShape"], layer["kernalSize"], layer["depth"], layer["padding"], layer["stride"], kernal=layer["kernal"], bias=layer["bias"]))
        elif layer["Type"] == "neuralNet.MaxPool": network.append(MaxPool(layer["inputShape"], layer["poolShape"], layer["stride"]))
        elif layer["Type"] == "neuralNet.AvgPool": network.append(AvgPool(layer["inputshape"], layer["poolShape"], layer["stride"]))
        else: 
            network.append({
                "neuralNet.Idenity":Identity(layer["outputSize"]),
                "neuralNet.BinaryStep":BinaryStep(layer["outputSize"]),
                "neuralNet.Tanh":Tanh(layer["outputSize"]),
                "neuralNet.Sigmoid":Sigmoid(layer["outputSize"]),
                "neuralNet.ReLU":ReLU(layer["outputSize"]),
                "neuralNet.GELU":GELU(layer["outputSize"]),
                "neuralNet.ELU":ELU(layer["outputSize"], alpha=layer["alpha"]),
                "neuralNet.SELU":SELU(layer["outputSize"]),
                "neuralNet.LeakyReLU":LeakyReLU(layer["outputSize"]),
                "neuralNet.PReLU":PReLU(layer["outputSize"], alpha=layer["alpha"]),
                "neuralNet.SiLU":SiLU(layer["outputSize"]),
                "neuralNet.Softplus":Softplus(layer["outputSize"]),
                "neuralNet.SoftMax":SoftMax(layer["outputSize"]),
                "neuralNet.Gaussian":Gaussian(layer["outputSize"]),
            }[layerType])
    if nNType == "neuralNet.FFNN": return FFNN(network=network)
    elif nNType == "neuralNet.RNN": return RNN(network=network)
