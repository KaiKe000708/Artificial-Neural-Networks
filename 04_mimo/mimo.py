"""
=====================================================================
 REDE MIMO (Múltiplas Entradas e Múltiplas Saídas)
=====================================================================
Implementação didática de uma rede de camada única generalizada para
múltiplas saídas (MIMO - Multiple Input, Multiple Output), treinada
com a Regra Delta (LMS), assim como o Adaline, porém com uma matriz
de pesos que conecta todas as entradas a todos os neurônios de saída.

Cada neurônio de saída "j" possui seu próprio vetor de pesos e bias,
mas todos são treinados simultaneamente a cada apresentação de
amostra:

    U = X . W + b                (net input de todos os neurônios)
    erro = D - U                 (erro em relação ao net input)
    W(t+1) = W(t) + eta * X^T . erro
    b(t+1) = b(t) + eta * soma(erro)

A saída final é obtida aplicando a função de ativação degrau a cada
neurônio de saída, permitindo resolver simultaneamente várias tarefas
de classificação/mapeamento com as MESMAS entradas.

=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class RedeMIMO:
    """Rede de camada única com múltiplas entradas e múltiplas saídas (Adaline generalizado)."""

    def __init__(self, n_inputs, n_outputs, learning_rate=0.05, epochs=200, tol=1e-6, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        rng = np.random.default_rng(random_state)
        # Matriz de pesos (n_entradas x n_saidas) e vetor de bias (1 x n_saidas)
        self.weights = rng.uniform(-0.5, 0.5, (n_inputs, n_outputs))
        self.bias = rng.uniform(-0.5, 0.5, (1, n_outputs))
        self.mse_per_epoch = []

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    @staticmethod
    def activation(U):
        """Função degrau bipolar aplicada a cada saída: {-1, +1}."""
        return np.where(U >= 0, 1, -1)

    def predict(self, X):
        return self.activation(self.net_input(X))

    def fit(self, X, D, verbose=True):
        """
        Treina a rede MIMO usando a Regra Delta (LMS), amostra a amostra.

        Parâmetros
        ----------
        X : ndarray (n_amostras, n_entradas)
        D : ndarray (n_amostras, n_saidas) -> matriz de saídas desejadas
        """
        eqm_anterior = np.inf

        for epoch in range(self.epochs):
            erro_quadratico_total = 0.0
            for xi, di in zip(X, D):
                xi = xi.reshape(1, -1)
                di = di.reshape(1, -1)

                u = self.net_input(xi)           # net input (1 x n_saidas)
                erro = di - u                    # erro (1 x n_saidas)

                # Atualização da matriz de pesos e do bias (Regra Delta)
                self.weights += self.learning_rate * np.dot(xi.T, erro)
                self.bias += self.learning_rate * erro

                erro_quadratico_total += np.sum(erro ** 2)

            eqm = erro_quadratico_total / (len(X) * D.shape[1])
            self.mse_per_epoch.append(eqm)

            if verbose:
                print(f"Época {epoch + 1:3d}/{self.epochs} - EQM: {eqm:.6f}")

            if abs(eqm_anterior - eqm) < self.tol:
                if verbose:
                    print(f"\n>> Convergência atingida na época {epoch + 1}!\n")
                break
            eqm_anterior = eqm
        return self

    def plot_eqm(self):
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.mse_per_epoch) + 1), self.mse_per_epoch, marker='o', color='purple')
        plt.title("Rede MIMO - Erro Quadrático Médio (EQM) por época")
        plt.xlabel("Época")
        plt.ylabel("EQM")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("mimo_convergencia.png", dpi=120)
        print("Gráfico salvo em 'mimo_convergencia.png'")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Exemplo: uma única rede aprendendo DUAS portas lógicas ao
    # mesmo tempo (AND e OR), a partir das MESMAS entradas.
    # Saída 1 -> AND | Saída 2 -> OR   (codificação bipolar -1/+1)
    # ------------------------------------------------------------
    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
    ])

    #           AND   OR
    D = np.array([
        [-1,  -1],   # (-1,-1) -> AND=-1, OR=-1
        [-1,   1],   # (-1, 1) -> AND=-1, OR= 1
        [-1,   1],   # ( 1,-1) -> AND=-1, OR= 1
        [ 1,   1],   # ( 1, 1) -> AND= 1, OR= 1
    ])

    print("=" * 60)
    print("Treinando rede MIMO para aprender AND e OR simultaneamente")
    print("=" * 60)

    mimo = RedeMIMO(n_inputs=2, n_outputs=2, learning_rate=0.05, epochs=200, tol=1e-6)
    mimo.fit(X, D)

    print("Matriz de pesos final:\n", mimo.weights)
    print("Bias final:", mimo.bias)

    print("\nTestando a rede treinada:")
    saidas = mimo.predict(X)
    for xi, di, yi in zip(X, D, saidas):
        print(f"Entrada: {xi} -> Saída da rede: {yi} | Esperado (AND, OR): {di}")

    mimo.plot_eqm()
