import time
import matplotlib.pyplot as plt
import numpy as np
import json
from extras import Errorfunctions, D_Errorfunctions, FileNameError, jsonSpecialEncoder

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
    
    def visualize(self, X, Y, optionsDisplay=None, displayInput=True, ErrorFunc='MSE'):
        #set to default
        if optionsDisplay is None: optionsDisplay = ['Neuron ' + str(i) for i in range(self.network[-1].outputSize)]
        #loops through visualizing data
        for x, y in zip(X, Y):
            #forward
            output = x
            for layer in self.network: output = layer.forward(output)
            error = Errorfunctions[ErrorFunc](y, output)
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