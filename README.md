# neuralNet

## Basic Info

This is an attempt to recreate neural networks by using built-in modules and numpy, no pytorch, no tensorflow

Current version: v0.2

Last update added:
- testing for FFNNs
- visualizing for FFNNs
- an new example ([XOR](https://github.com/pleituer/neuralNet/tree/main/examples/XOR))
- Fixed the visualizer

## Current Progress

Implemented
- Dense Layer
- Activations (tanh, sigmoid, ReLu, softmax)
- GRUs (a simplistic one with one internal weight and one internal activation function, tanh, and its called `SGRU`)
- allowing users to implement other custom GRUs
- LSTM (still in progress of debugging)
- FFNNs
- RNNs
- Training, testing, and visualizing the outputs for FFNNs and RNNs (not visualizing yet though)

Will implement
- visualizing of RNNs
- Convolutional Neural Networks
- other GRUs
- Transformers?

## How to use:

### Step 0:

Be sure to satisfy all the requirements, namely
```
numpy
matplotlib
```

### Step 1:
Import
```python
import neuralNet
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
    neuralNet.Sigmoid(charSize),
    neuralNet.SGRU(charSize, charSize),
    neuralNet.Tanh(charSize),
    neuralNet.Dense(charSize, charSize),
    neuralNet.SoftMax(charSize)
]
```
Note that the activation functions are **SEPERATE** from the Dense Layer, so if you want an activation Layer, just don't forget to include the layer after the Dense Layer.

### Step 3:
Train & test
```python
ffnn.train(X, Y, epochs=epochs, learningRate=learningRate, ErrorFunc=ErrorFunc, test=True, testPercentage=0.9)
```
or
```python
rnn.train(X, YTrue, epochs=epochs, learningRate=learningRate, ErrorFunc=ErrorFunc, test=True, testPercentage=0.9)
```
where `X` is your training input, `Y` is your correct training output, `epochs` is the number of epochs, `learningRate` is the learningRate, and `ErrorFunc` can currently be Mean Squared Error(use `ErrorFunc="MSE"`), or Cross-Entropy Loss(use `ErrorFunc="CEL"`) And `test=True` if you wanted to give it tests, also specify the `testPercentage` (default is 90%) to specify how much of the data (`X` and `Y`) is used for testing.

### Step 4:
Wait & visualize

Wait for the results to flow, and if you wanted, you can also visualize the output, as below:
```python
ffnn.visualize(X, Y, optionsDisplay=optionsDisplay)
```
Where `X` and `Y` as still defined as above, and `optionsDisplay` is the meaning of each neuron firing, so if the first neuron fires, and it corresponds to `'Dog'`, and the second corresponds to `'Cat'`, then the `optionsDisplay` should be set as `optionsDisplay=['Dog', 'Cat']`. This is optional.

This will print out the input given to the neural network, it's predicted output and the expected output.

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
neuralNet.Sigmoid(outputSize) #sigmoid activation layer
neuralNet.Tanh(ouputSize) #tanh activation layer
neuralNet.ReLU(outputSize) #ReLU activation layer
neuralNet.Softmax(outputSize) #softmax activation layer
```
Any one of these line will create an activation layer, with Sigmoid, Tanh, ReLU, and Softmax to choose from.

### 3. RNNs

There are 2 types of RNNs available.

The following will create a GRU with one internal weight and one internal activation function, thus a simplistic one (SGRU)
```python
neuralNet.SGRU(inputSize, outputSize)
```
Or the following which will create a custom GRU
```python
neuralNet.GRU(inputSize, outputSize, internalSize, internalFunc, D_internalFunc, trimFunc)
```
The following will creat a LSTM (currently not fully functional, you might encounter some bugs, or it might not work properly)
```python
neuralNet.LSTM(inputSize, outputSize)
```

These are all the current available layer types, as time goes on, I will add more layers to it!

For more info, please check out the [exmaples](https://github.com/pleituer/neuralNet/tree/main/examples)

**Note: This project is started and heavily influenced by a video by the YouTube channel The Independent Code, [Neural Network from Scratch | Mathematics & Python Code](https://www.youtube.com/watch?v=pauPCy_s0Ok). Thus, part of the code will be the same.**
