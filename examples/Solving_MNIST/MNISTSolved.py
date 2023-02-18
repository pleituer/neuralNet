#solving the mnist dataset

print('importing libraries...')
#imports tensorflow to load the dataset
import tensorflow.keras.datasets as tf
import numpy as np
import time

import neuralNet
import neuralNet.Activations as Activations

#loads the data
startTime = time.time()
print('Loading data...')
(trainX, trainY), (testX, testY) = tf.mnist.load_data(
    path='mnist.npz'
)
endTime = time.time()
print(f'Loaded data, took {endTime - startTime} seconds')

#input shape: (1, 28, 28)
#output shaoe: (10, 1)
dataXLen = 28
dataYLen = 28

trainNum = 100
testNum = 10

inputShape = (1, dataYLen, dataXLen)
outputShape = (10, 1)

#preprocess
startTime = time.time()
print('Preproccessing data...')
trainX = np.reshape(trainX[:trainNum]/255, (trainNum, *inputShape))
trainY = np.reshape([[int(n==trainY[d]) for n in range(10)] for d in range(trainNum)], (trainNum, *outputShape))
testX = np.reshape(testX[:testNum]/255, (testNum, *inputShape))
testY = np.reshape([[int(n==testY[d]) for n in range(10)] for d in range(testNum)], (testNum, *outputShape))
endTime = time.time()
print(f'Finished prerpocessing data, took {endTime - startTime} seconds')
print('Starts training...')

Epochs=400
learningRate=0.1

#structure is LeNet
network = [
    neuralNet.Convolute(inputShape, kernalSize=5, depth=6, padding=2, stride=1),
    Activations.Sigmoid((6,28,28)),
    neuralNet.AvgPool((6,28,28), (2,2), 2),
    neuralNet.Convolute((6, 14, 14), kernalSize=5, depth=16, padding=0, stride=1),
    Activations.Sigmoid((16, 10, 10)),
    neuralNet.AvgPool((16, 10, 10), (2,2), 2),
    neuralNet.Flatten((16, 5, 5)),
    neuralNet.Dense(16*5*5, 120),
    Activations.Sigmoid(120),
    neuralNet.Dense(120, 84),
    Activations.Sigmoid(84),
    neuralNet.Dense(84, outputShape[0]),
    Activations.SoftMax(outputShape[0])
]

ffnn = neuralNet.FFNN(network=network)
ffnn.trainWithTest(trainX, trainY, testX, testY, epochs=Epochs, learningRate=learningRate, ErrorFunc='CEL')
ffnn.save('SolvingMNIST.json')
ffnn2 = neuralNet.loadNeuralNet('SolvingMNIST.json')
