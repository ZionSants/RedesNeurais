# Simulação de camada de neurônios de saída

entradas = [1, 2, 3, 2.5]

# Determina a importância da informação de entrada para a saída do neurônio
pesos1 = [0.2, 0.8, -0.5, 1.0]
pesos2 = [0.5, -0.91, 0.26, -0.5]
pesos3 = [-0.26, -0.27, 0.17, 0.87]

# Constante que determina a sensibilidade para o neurônio ativar sua saída
vies1 = 2
vies2 = 3
vies3 = 0.5

saida = [entradas[0]*pesos1[0] + entradas[1]*pesos1[1] + entradas[2]*pesos1[2] + entradas[3]*pesos1[3] + vies1,
         entradas[0]*pesos2[0] + entradas[1]*pesos2[1] + entradas[2]*pesos2[2] + entradas[3]*pesos2[3] + vies2,
         entradas[0]*pesos3[0] + entradas[1]*pesos3[1] + entradas[2]*pesos3[2] + entradas[3]*pesos3[3] + vies3]

print(saida)