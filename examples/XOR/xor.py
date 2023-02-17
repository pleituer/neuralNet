#this is an example in using feed-forward neural networks
#using the classical example, the XOR Gate

import numpy as np
import random

import neuralNet

#configs
Epochs = 100
learningRate = 0.1
trainNum = 50
visNum = 10

network = [
    neuralNet.Dense(2, 3),
    neuralNet.Activations.Tanh(3),
    neuralNet.Dense(3, 1),
    neuralNet.Activations.Tanh(1)
]

#xor output generation (its not that efficient)
def xor(In): return {0:[0], 1:[1], 3:[0], 2:[1]}[In[0]*2+In[1]]

#training and testing data
X = [random.choices([0, 1], k=2) for _ in range(trainNum)]
Y = [xor(x) for x in X]

inputShape = (2, 1)
outputShape = (1, 1)

X = np.reshape(X, (trainNum, *inputShape))
Y = np.reshape(Y, (trainNum, *outputShape))

#visulizing data
Xvis = [random.choices([0, 1], k=2) for _ in range(visNum)]
Yvis = [xor(x) for x in Xvis]

Xvis = np.reshape(Xvis, (visNum, *inputShape))
Yvis = np.reshape(Yvis, (visNum, *outputShape))

#the ffnn itself
ffnn = neuralNet.FFNN(network)
ffnn.train(X, Y, epochs=Epochs, learningRate=learningRate, ErrorFunc='MSE', test=True, testPercentage=0.9)
ffnn.visualize(Xvis, Yvis)
ffnn.save('xorData.json')
ffnn2 = neuralNet.loadNeuralNet('xorData.json')
ffnn2.visualize(Xvis, Yvis)
