import json
import os
from neuralNet.extras import json_numpy_obj_hook
from neuralNet.Dense import Dense
from neuralNet.SGRU import SGRU
from neuralNet.LSTM import LSTM
from neuralNet.RawConvolute import RawConvolute
from neuralNet.Stride import Stride
from neuralNet.Convolute import Convolute
from neuralNet.MaxPool import MaxPool
from neuralNet.AvgPool import AvgPool
from neuralNet.Reshape import Reshape
from neuralNet.Flatten import Flatten
from neuralNet.FFNN import FFNN
from neuralNet.RNN import RNN
from neuralNet.Activations.Identity import Identity
from neuralNet.Activations.BinaryStep import BinaryStep
from neuralNet.Activations.Tanh import Tanh
from neuralNet.Activations.Sigmoid import Sigmoid
from neuralNet.Activations.ReLU import ReLU
from neuralNet.Activations.GELU import GELU
from neuralNet.Activations.ELU import ELU
from neuralNet.Activations.SELU import SELU
from neuralNet.Activations.LeakyReLU import LeakyReLU
from neuralNet.Activations.PReLU import PReLU
from neuralNet.Activations.SiLU import SiLU
from neuralNet.Activations.Gaussian import Gaussian
from neuralNet.Activations.Softplus import Softplus
from neuralNet.Activations.SoftMax import SoftMax

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
        elif layer["Type"] == "neuralNet.AvgPool": network.append(AvgPool(layer["inputShape"], layer["poolShape"], layer["stride"]))
        elif layer["Type"] == "neuralNet.Reshape": network.append(Reshape(layer["inputShape"], layer["outputShape"]))
        elif layer["Type"] == "neuralNet.Flatten": network.append(Flatten(layer["inputShape"]))
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