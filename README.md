# neuralNet

## Basic Info

This is an attempt to recreate neural networks by using built-in modules and numpy, no pytorch, no tensorflow

### Current version: v0.3

Last Update Added

- Convolutional Neural Networks
- Pooling Layers
- Save & Load functions
- New example [Solving_MNIST](https://github.com/pleituer/neuralNet/tree/main/examples/Solving_MNIST)

## Current Progress

Implemented
- Dense Layer
- Activations (tanh, sigmoid, ReLU, variations of ReLU, softmax, and more)
- GRUs (a simplistic one with one internal weight and one internal activation function, tanh, and its called `SGRU`)
- LSTM (still in progress of debugging)
- FFNNs
- RNNs
- CNNs
- Training, testing, and visualizing the outputs for FFNNs and RNNs (not visualizing yet though)

Will implement
- visualizing of RNNs
- other GRUs
- Custom Layers

## How to use:

### Step 0:

Be sure to satisfy all the requirements, namely
```
numpy
matplotlib
scipy
```
And install using the following command
```
pip install -i https://test.pypi.org/simple/ neuralNet
```

### Step 1:
Import
```python
import neuralNet
import neuralNet.Activations as Activations #this is recommended to save coding time
```

### Step 2:
Setup
```python
ffnn = neuralNet.FFNN(network=network)
#if you just need feed-foward neural networks
```
or
```python
rnn = neuralNet.RNN(network=network)
#if you need a recurrent neural network (like GRUs or LSTMs)
```
where `network` is the structure of your AI, so as in the [4-letter-word-generation example](https://github.com/pleituer/neuralNet/blob/main/examples/4_letter_word_generation/4_letter_word_generation.py), `network` is defined as follows
```python
network = [
    neuralNet.Dense(charSize, charSize),
    Activations.Sigmoid(charSize),
    neuralNet.SGRU(charSize, charSize),
    Activations.Tanh(charSize),
    neuralNet.Dense(charSize, charSize),
    Activations.SoftMax(charSize)
]
```
Note that the activation functions are **SEPERATE** from the Dense Layer, so if you want an activation Layer, just don't forget to include the layer after the Dense Layer.

### Step 3:
Train & test
```python
ffnn.train(X, Y, epochs=epochs, learningRate=learningRate,
    ErrorFunc=ErrorFunc, test=True, testPercentage=0.9)
```
or
```python
rnn.train(X, YTrue, epochs=epochs, learningRate=learningRate, 
    ErrorFunc=ErrorFunc, test=True, testPercentage=0.9)
```
where `X` is your training input, `Y` is your correct training output, `epochs` is the number of epochs, `learningRate` is the learningRate, and `ErrorFunc` can currently be Mean Squared Error(use `ErrorFunc="MSE"`), or Cross-Entropy Loss(use `ErrorFunc="CEL"`) And `test=True` if you wanted to give it tests, also specify the `testPercentage` (default is 90%) to specify how much of the data (`X` and `Y`) is used for testing.

### Step 4:
Wait & visualize

Wait for the results to flow, and if you wanted, you can also visualize the output, as below:
```python
ffnn.visualize(X, Y, optionsDisplay=optionsDisplay, displayInput=True)
```
Where `X` and `Y` as still defined as above, and `optionsDisplay` is the meaning of each neuron firing, so if the first neuron fires, and it corresponds to `'Dog'`, and the second corresponds to `'Cat'`, then the `optionsDisplay` should be set as `optionsDisplay=['Dog', 'Cat']`. This is optional.

This will print out the input given to the neural network, it's predicted output and the expected output.

### Step 5:
Save your progress

After everything, you can run the line
```python
ffnn.save(filePath) #or rnn.save(filePath)
```
to save your progress, note that it is saved in a `.json` file, so make sure you file path ends in `.json`

Then, you might want to continue your progress, you can load the neural network simply with the following line
```python
neuralNetwork = neuralNet.load(filePath)
```
Just make sure the `filePath` actually exist

I haven't implement visualizing for recurrent neural networks, so you can't test it (unless you code it yourself)

## Layers

### 1. Dense

This layer can be created by passing the line
```python
neuralNet.Dense(inputSize, outputSize)
```
This creates a dense (or linear) layer.

### 2. Activation

These layers can be created by passing any of the following lines, depending on your own need.
```python
Activations.Sigmoid(outputSize) #sigmoid activation layer
Activations.Tanh(ouputSize) #tanh activation layer
Activations.ReLU(outputSize) #ReLU activation layer
Activations.Softmax(outputSize) #softmax activation layer
Activations.Identity(outputSize) #Identity activation layer, x = x, just in case if anyone wanted to use
Activations.BinaryStep(outputSize) #Binary Step activation layer
Activations.GELU(outputSize) #GELU activation layer
Activations.ELU(outputSize) #ELU activation layer
Activations.SELU(outputSize) #SELU activation layer
Activations.LeakyReLU(outputSize) #Leaky ReLU activation layer
Activations.PReLU(outputSize) #PReLU activation layer
Activations.SiLU(outputSize) #SiLU activation layer
Activations.Softplus(outputSize) #Softplus activation layer
Activations.Gaussian(outputSize) #Gaussian activation layer
```
Any one of these line will create an activation layer, with Sigmoid, Tanh, ReLU, and Softmax to choose from.

### 3. RNNs

There are 2 types of RNNs available.

The following will create a GRU with one internal weight and one internal activation function, thus a simplistic one (SGRU)
```python
neuralNet.SGRU(inputSize, outputSize)
```
The following will creat a LSTM (currently not fully functional, you might encounter some bugs, or it might not work properly)
```python
neuralNet.LSTM(inputSize, outputSize)
```

### 4. CNNs

There are 2 types of Convolutional layers available and 2 types of pooling layers you can use

```python
neuralNet.RawConvolute(inputShape, kernalSize, depth, padding) #Convolution with stride set to 1
neuralNet.Convolute(inputShape, kernalSize, depth, padding, stride) #If you need strides not 1
```
Or you may want to check out the following pooling layers
```python
neuralNet.MaxPool(inputShape, poolingShape, stride) #Max pooling
neuralNet.AvgPool(inputShape, poolingShape, stride) #Average pooling
```
I don't know if this is a pooling layer but it functions like one:
```python
neuralNet.Stride(inputShape, stride) #it is a striding layer
#and it can be used as a pooling layer, although it is incorporated in the neuralNet.Convolute object
```

These are all the current available layer types, as time goes on, I will add more different layers to it!

For more info, please check out the [exmaples](https://github.com/pleituer/neuralNet/tree/main/examples)

**Note: This project is started and heavily influenced by a video by the YouTube channel The Independent Code, [Neural Network from Scratch | Mathematics & Python Code](https://www.youtube.com/watch?v=pauPCy_s0Ok). Thus, part of the code will be the same.**
