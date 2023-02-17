import numpy as np
import time

import neuralNet
from ticTacToeClasses import *

#epochs, learning rate
Epochs = 500
learningRate = 0.01

#configurations, like training number, input size, output size
with open("ticTacToeTRAINDATA.txt", "r") as f: data = f.read().split('\n')
trainNum, inputColumnSize, outputColumnSize = tuple(map(int, data[0][2:].split()))

X = []
Y = []

#extracts data
startTime = time.time()
print("Extracting data...")
for line in data[:-1]:
    if line[0] == "X": X.append(list(map(int, line[2:].split())))
    if line[0] == "Y": Y.append(list(map(float, line[2:].split())))
endTime = time.time()
print(f'Finished extracting data, took {endTime - startTime} seconds')

#reshapes
X = np.reshape(X, (trainNum, inputColumnSize, 1))
Y = np.reshape(Y, (trainNum, outputColumnSize, 1))

#the network structure
network = [
    neuralNet.Dense(inputColumnSize, 3*inputColumnSize),
    neuralNet.Activations.Tanh(3*inputColumnSize),
    neuralNet.Dense(3*inputColumnSize, 2*outputColumnSize),
    neuralNet.Activations.Sigmoid(2*outputColumnSize),
    neuralNet.Dense(2*outputColumnSize, outputColumnSize),
    neuralNet.Activations.Tanh(outputColumnSize)
]

#the neural network itself (2 lines)
ffnn = neuralNet.FFNN(network=network)
ffnn.train(X, Y, epochs=Epochs, learningRate=learningRate, ErrorFunc='MSE', test=False)
ffnn.save('ticTacToeData.json')

#plays with user
while input('Wanna play? [y/n]: ').lower() == 'y':
    #new board per play
    t = TicTacToe()
    #O means bot's ture, X will be user, (user first)
    botSym = o
    user = UserPlayer(changeSym(botSym))
    #just in case the bot decided to put in an invalid spot, although I already tried my best to prevent it
    filt = lambda m: m[1] in t.emptySpots()

    #this is just like a typical tic tac toe code
    while t.status is None:
        if t.turn == user.symbol:
            t.printout()
            t.makeMove(user.getMove(t), user.symbol)
        t.check()
        if t.turn == botSym and t.status is None:
            t.printout()
            rawBotMove = np.reshape(ffnn.forward(np.reshape(t.squish(), (inputColumnSize, 1))), (outputColumnSize, ))
            modBotMove = [(rawBotMove[i], (i//t.size, i%t.size)) for i in range(outputColumnSize)]
            #print(modBotMove)
            midBotMove = list(filter(filt, modBotMove))
            #print(midBotMove)
            sortedBotMove = sorted(midBotMove, key=(lambda m: m[0]))[::-1]
            botMove = sortedBotMove[0][1]
            print('Bot moved:', botMove)
            t.makeMove(botMove, botSym)
        t.check()
    t.printout()
    print(['Tie!', 'X Wins!', 'O Wins!'][t.status])
