"""
=====================================================================
 PERCEPTRON MULTICAMADAS (MLP - Multi Layer Perceptron)
=====================================================================
Implementação didática de uma rede MLP totalmente conectada, treinada
com o algoritmo de Retropropagação do Erro (Backpropagation), usando
função de ativação sigmoide logística e gradiente descendente.

Arquitetura genérica: permite definir qualquer número de camadas
ocultas e neurônios, através da lista `layers`.
Ex.: layers=[2, 4, 1] -> 2 entradas, 1 camada oculta com 4 neurônios,
                          1 neurônio de saída.

Algoritmo (resumo):
  1. Forward pass: propaga a entrada calculando as ativações de cada
     camada.
  2. Cálculo do erro na camada de saída.
  3. Backward pass: retropropaga o erro calculando os gradientes
     locais de cada camada (regra da cadeia).
  4. Atualização dos pesos e bias com taxa de aprendizagem (e
     opcionalmente momento).

=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class MLP:
    """Rede Perceptron Multicamadas treinada com Backpropagation."""

    def __init__(self, layers, learning_rate=0.5, epochs=5000, momentum=0.9, random_state=42):
        """
        Parâmetros
        ----------
        layers : list[int]
            Número de neurônios em cada camada, incluindo entrada e saída.
            Ex: [2, 4, 1] -> 2 entradas, 4 neurônios ocultos, 1 saída.
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.momentum = momentum
        self.mse_per_epoch = []

        rng = np.random.default_rng(random_state)
        self.weights = []
        self.biases = []
        self.prev_dw = []  # para o termo de momento
        self.prev_db = []

        # Inicializa pesos e bias entre cada par de camadas consecutivas
        for i in range(len(layers) - 1):
            w = rng.uniform(-1, 1, (layers[i], layers[i + 1]))
            b = rng.uniform(-1, 1, (1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
            self.prev_dw.append(np.zeros_like(w))
            self.prev_db.append(np.zeros_like(b))

    @staticmethod
    def sigmoid(u):
        return 1.0 / (1.0 + np.exp(-u))

    @staticmethod
    def sigmoid_derivative(saida_ativada):
        """Derivada da sigmoide expressa em função da própria saída ativada: f'(u) = f(u)*(1-f(u))."""
        return saida_ativada * (1 - saida_ativada)

    def forward(self, X):
        """Executa a propagação direta (forward pass) e retorna as ativações de todas as camadas."""
        ativacoes = [X]
        entrada_atual = X
        for w, b in zip(self.weights, self.biases):
            u = np.dot(entrada_atual, w) + b
            saida = self.sigmoid(u)
            ativacoes.append(saida)
            entrada_atual = saida
        return ativacoes

    def backward(self, ativacoes, d):
        """Executa a retropropagação do erro e atualiza pesos/bias."""
        n_camadas = len(self.weights)
        saida_rede = ativacoes[-1]

        # Erro e gradiente local (delta) da camada de saída
        erro = d - saida_rede
        delta = erro * self.sigmoid_derivative(saida_rede)
        deltas = [delta]

        # Propaga os deltas para trás, das camadas ocultas até a primeira
        for l in range(n_camadas - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[l].T) * self.sigmoid_derivative(ativacoes[l])
            deltas.insert(0, delta)

        # Atualiza pesos e bias de cada camada com termo de momento
        for l in range(n_camadas):
            grad_w = np.dot(ativacoes[l].T, deltas[l])
            grad_b = np.sum(deltas[l], axis=0, keepdims=True)

            dw = self.learning_rate * grad_w + self.momentum * self.prev_dw[l]
            db = self.learning_rate * grad_b + self.momentum * self.prev_db[l]

            self.weights[l] += dw
            self.biases[l] += db

            self.prev_dw[l] = dw
            self.prev_db[l] = db

        return np.mean(erro ** 2)

    def fit(self, X, d, verbose=True, print_every=500):
        for epoch in range(self.epochs):
            ativacoes = self.forward(X)
            mse = self.backward(ativacoes, d)
            self.mse_per_epoch.append(mse)
            if verbose and (epoch + 1) % print_every == 0:
                print(f"Época {epoch + 1:5d}/{self.epochs} - EQM: {mse:.6f}")
        return self

    def predict(self, X):
        return self.forward(X)[-1]

    def plot_eqm(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.mse_per_epoch, color='seagreen')
        plt.title("MLP - Erro Quadrático Médio (EQM) por época")
        plt.xlabel("Época")
        plt.ylabel("EQM")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("mlp_convergencia.png", dpi=120)
        print("Gráfico salvo em 'mlp_convergencia.png'")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Exemplo clássico: Porta lógica XOR (NÃO linearmente separável,
    # não pode ser resolvida por Perceptron/Adaline simples -> precisa de MLP)
    # ------------------------------------------------------------
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)
    d = np.array([[0], [1], [1], [0]], dtype=float)

    print("=" * 60)
    print("Treinando MLP (2-4-1) para resolver a porta lógica XOR")
    print("=" * 60)

    mlp = MLP(layers=[2, 4, 1], learning_rate=0.5, epochs=5000, momentum=0.9)
    mlp.fit(X, d, print_every=1000)

    print("\nTestando a rede treinada:")
    saida = mlp.predict(X)
    for xi, target, y in zip(X, d, saida):
        print(f"Entrada: {xi} -> Saída da rede: {y[0]:.4f} (~{round(y[0])}) | Esperado: {target[0]}")

    mlp.plot_eqm()
