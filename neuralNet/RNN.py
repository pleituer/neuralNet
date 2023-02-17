import numpy as np
import matplotlib.pyplot as plt
import time
import json
from neuralNet.extras import Errorfunctions, D_Errorfunctions, FileNameError, jsonSpecialEncoder

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