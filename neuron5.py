import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3)

class layerDense:
    def __init__(self, nInputs, nNeurons):
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        self.biases = np.zeros((1, nNeurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class activationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = layerDense(2,5)
activation1 = activationReLU()

layer1.forward(X)

activation1.forward(layer1.output)
print(activation1.output)
