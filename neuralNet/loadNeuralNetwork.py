import json
import os
from extras import json_numpy_obj_hook
from Dense import Dense
from SGRU import SGRU
from LSTM import LSTM
from RawConvolute import RawConvolute
from Stride import Stride
from Convolute import Convolute
from MaxPool import MaxPool
from AvgPool import AvgPool
from FFNN import FFNN
from RNN import RNN
from Activations.Identity import Identity
from Activations.BinaryStep import BinaryStep
from Activations.Tanh import Tanh
from Activations.Sigmoid import Sigmoid
from Activations.ReLU import ReLU
from Activations.GELU import GELU
from Activations.ELU import ELU
from Activations.SELU import SELU
from Activations.LeakyReLU import LeakyReLU
from Activations.PReLU import PReLU
from Activations.SiLU import SiLU
from Activations.Gaussian import Gaussian
from Activations.Softplus import Softplus
from Activations.SoftMax import SoftMax

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
        elif layer["Type"] == "neuralNet.AvgPool": network.append(AvgPool(layer["inputshape"], layer["poolShape"], layer["stride"]))
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