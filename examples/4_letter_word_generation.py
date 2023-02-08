#The following is an example to how the neuralNet can be applied

import numpy as np
import time
import random
import math

import neuralNet

#Goal: making the ai remember the 500 words in words.txt (results aren't that good, but expected from the small size of the ai)
#I didn't split the data into train and test, as we do want it to memorize the training data

#read from training data
with open('words.txt', 'r', encoding='utf-8') as f: fullText = f.read()

#encoding process
chars = sorted(list(set(fullText)))
charSize = len(chars)
stoi = {chars[i]:i for i in range(charSize)}
itos = {i:chars[i] for i in range(charSize)}

def encodeString(string): return [stoi[s] for s in string]
def decodeString(encodedString): return ''.join(itos[i] for i in encodedString)

trainData = encodeString(fullText)

#configs
blockSize = 4
EpochNum = 250
learningRate = 0.1

trainNum = 250
testNum = 10

#network architecture
network = [
    neuralNet.Dense(charSize, charSize),
    neuralNet.Sigmoid(),
    neuralNet.SGRU(charSize, charSize),
    neuralNet.Tanh(),
    neuralNet.Dense(charSize, charSize),
    neuralNet.SoftMax()
]

#modify it to match input
def modify(d): return [int(_==d) for _ in range(charSize)]
def reverseModify(d): return np.where(d==[sorted(d, key=(lambda o:o[0]))[-1]])[0][0]


#tokenization
def getBatch(split):
    offsets = [(blockSize+1)*random.randrange(0, (len(trainData) - blockSize)//(blockSize+1)) for _ in range(trainNum)]
    x = [[modify(d) for d in trainData[i:i+blockSize-1]] for i in offsets]
    y = [[modify(d) for d in trainData[i+1:i+blockSize]] for i in offsets]
    return x, y

XTrain, YTrain = getBatch('train')
inputShape = (blockSize-1, charSize, 1)
outputShape = (blockSize-1, charSize, 1)

#reshape (transposes each input, (1, n) -> (n, 1))
X = np.reshape(XTrain, (trainNum,*inputShape))
YTrue = np.reshape(YTrain, (trainNum,*outputShape))

#giving the ai a starting letter for testing purpose, so we know how is the performance directly
startsWith = random.choices(chars, k=testNum)
startsWith = [modify(stoi[t]) for t in startsWith]
RNNtestShape = (charSize, 1)

#test data reshape (since we are only providing the first letter, we will feed it's output to its input)
XTest = np.reshape(startsWith, (len(startsWith), *RNNtestShape))

#given a list initial letter, the following function will return a list of size+1 long letters
def test(rnn, inputs, size):
    outputs = ()
    for x in inputs:
        word = (x,)
        output = x[:]
        for t in range(size):
            for layer in rnn.network: output = layer.forward(output)
            word += (output, )
        outputs += (word,)
    return outputs

#the network (RNN)
rnn = neuralNet.RNN(network)     
rnn.train(X, YTrue, epochs=EpochNum, learningRate=learningRate, ErrorFunc='CEL')
output = test(rnn, XTest, 3)

#reverse the encoding and tokenization process so we can understand
visOutput = [''.join(itos[reverseModify(char)] for char in word) for word in output]
for word in range(len(visOutput)): print(f'Word starts with Letter {visOutput[word][0]}: {visOutput[word]}')
