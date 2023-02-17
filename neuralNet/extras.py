import numpy as np
from scipy.stats import norm
import json
import base64

#Mean Squared Error
def MSE(yTrue, y): return np.mean(np.power(yTrue - y, 2))
def D_MSE(yTrue, y): return 2*(y - yTrue)/np.size(yTrue)

#Cross-Entropy Loss
def CEL(yTrue, y): return -np.mean(np.multiply(yTrue, np.log(y)))
def D_CEL(yTrue, y): return -yTrue/(y*np.size(yTrue))

#Binomial Cross Entropy Loss
def BCEL(yTrue, y): return -np.mean(np.multiply(yTrue, np.log(y)) + np.multiply(1 - yTrue, np.log(1 - y)))
def D_BCEL(yTrue, y): return ((1 - yTrue)/(1 - y) - yTrue/y)/np.size(yTrue)

Errorfunctions = {'CEL':CEL, 'MSE':MSE, 'BCEL':BCEL}
D_Errorfunctions = {'CEL':D_CEL, 'MSE':D_MSE, 'BCEL':D_BCEL}

#identity
identity = lambda x: x
D_identity = lambda x: 1

#Binary Step
binStep = lambda x: np.greater(x, 0)
D_binStep = lambda x: 0

#hyperbolic tangent
tanh = lambda x: np.tanh(x)
D_tanh = lambda x: (1 - np.power(tanh(x), 2))
 
#Sigmoid activation
sigmoid = lambda x: 1/(1 + np.exp(-x))
D_sigmoid = lambda x: sigmoid(x)*(1 - sigmoid(x))

#ReLU
relu = lambda x: np.maximum(x, 0)
D_relu = lambda x: np.greater(x, 0)

#GELU
gelu = lambda x: x*norm.cdf(x)
D_gelu = lambda x: norm.cdf(x) + x*norm.pdf(x)

#ELU
elu = lambda x, alpha: (x>=0)*x + (x<0)*alpha*(np.exp(x)-1)
D_elu = lambda x, alpha: (x>=0)*1 + (x<0)*alpha*np.exp(x)

#SELU
selu = lambda x: 1.0507*elu(x, 1.67326)
D_selu = lambda x: 1.0507*D_elu(x, 167326)

#Leaky ReLu
lrelu = lambda x: np.maximum(x, 0.01*x)
D_lrelu = lambda x: (x>=0.01*x)*1 + (x<0.01*x)*0.01

#PReLU
prelu = lambda x, alpha: (x>=0)*x + (x<0)*alpha*x
D_prelu = lambda x, alpha: (x>=0)*1 + (x<0)*alpha

#SiLU
silu = lambda x: x*sigmoid(x)
D_silu = lambda x: sigmoid(x) + x*D_sigmoid(x)

#Gaussian
gaussian = lambda x: np.exp(-np.power(x, 2))
D_gaussian = lambda x: -2*x*gaussian(x)

#Softplus
softplus = lambda x: np.log(1 + np.exp(x))
D_softplus = lambda x: 1/(1 + np.exp(-x))

class FileNameError(Exception):
    def __init__(self, filename):
        super().__init__(f'File has to be a .json file, not .{filename.split(".")[-1]} as in {filename}')

class jsonSpecialEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#reference: https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array/24375113#24375113
def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct
