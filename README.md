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
- LSTM (still in progress of debugging)
- FFNNs
- RNNs
- Training, testing, and visualizing the outputs for FFNNs

Will implement
- test function for RNNs
- Convolutional Neural Networks
- other GRUs
- Transformers?

## How to use:

Step 0:

Be sure to satisfy all the requirements, namely
```
numpy
matplotlib
```

Step 1:
Import
```python
import neuralNet
```

Step 2:
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

Step 3:
Train & test
```python
ffnn.train(X, Y, epochs=epochs, learningRate=learningRate, ErrorFunc=ErrorFunc, test=True, testPercentage=0.9)
```
or
```python
rnn.train(X, YTrue, epochs=epochs, learningRate=learningRate, ErrorFunc=ErrorFunc)
```
where `X` is your training input, `Y` is your correct training output, `epochs` is the number of epochs, `learningRate` is the learningRate, and `ErrorFunc` can currently be Mean Squared Error(use `ErrorFunc="MSE"`), or Cross-Entropy Loss(use `ErrorFunc="CEL"`) And `test=True` if you wanted to give it tests (it's currently available on FFNNs, not RNNs), also specify the `testPercentage` (default is 90%) to specify how much of the data (`X` and `Y`) is used for testing.

Step 4:
Wait & visualize

Wait for the results to flow, and if you wanted, you can also visualize the output, as below:
```python
ffnn.visualize(X, Y, optionsDisplay=optionsDisplay)
```
Where `X` and `Y` as still defined as above, and `optionsDisplay` is the meaning of each neuron firing, so if the first neuron fires, and it corresponds to `'Dog'`, and the second corresponds to `'Cat'`, then the `optionsDisplay` should be set as `optionsDisplay=['Dog', 'Cat']`. This is optional.

This will print out the input given to the neural network, it's predicted output and the expected output.

I haven't implement testing and visualizing for recurrent neural networks, so you can't test it (unless you code it yourself)

For more info, please check out the [exmaples](https://github.com/pleituer/neuralNet/tree/main/examples)

**Note: This project is started and heavily influenced by a video by the YouTube channel The Independent Code, [Neural Network from Scratch | Mathematics & Python Code](https://www.youtube.com/watch?v=pauPCy_s0Ok). Thus, part of the code will be the same.**
