import numpy as np 

# Utilizando camadas de neurônios como objetos

# Seed fixa para receber os mesmos resultados aleatórios
np.random.seed()

# 3 Exemplos com 4 características cada
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Class: Instruções de como criar a camada (Define suas características e ações)
class layerDense:
    # Inicia a camada com suas características próprias (self)
    def __init__(self, nInputs, nNeurons):
        # Multiplica por 0.10 para manter pesos pequenos prevenindo overfitting
        self.weights = 0.10 * np.random.randn(nInputs, nNeurons)
        # Viés inicializado como zero
        self.biases = np.zeros((1, nNeurons))
    # Método que ensina a camada a realizar a ação dela e guardar no seu output
    def forward(self, inputs):
        # Entradas * pesos + viés
        self.output = np.dot(inputs, self.weights) + self.biases


# Os inputs da segunda camada deve ter a mesma quantidade de neurônios da primeira 
layer1 = layerDense(4,5)
layer2 = layerDense(5,2)

# Envia os dados de X pára o método foward da primeira camada
layer1.forward(X)

# Envia os dados da primeira camada pára o método foward da segunda camada
layer2.forward(layer1.output)
print("  Neurônio 1 Neurônio 2")
print(layer2.output)