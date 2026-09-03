"""
=====================================================================
 ADALINE (ADAptive LInear NEuron)
=====================================================================
Implementação didática da rede Adaline, proposta por Widrow e Hoff.

Diferente do Perceptron, o Adaline ajusta os pesos com base no erro
calculado sobre o net input (saída linear), e não sobre a saída após
a função de ativação. Isso caracteriza a Regra Delta (LMS - Least
Mean Squares):

    erro = d - u        (u = net input, saída linear)
    w(t+1) = w(t) + eta * erro * x
    b(t+1) = b(t) + eta * erro

A função de ativação (degrau) só é usada na hora de classificar/
apresentar a saída final, não durante o treinamento.

=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class Adaline:
    """Rede Adaline treinada pela Regra Delta (gradiente descendente / LMS)."""

    def __init__(self, n_inputs, learning_rate=0.01, epochs=100, tol=1e-5, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol  # critério de parada: variação mínima do erro (EQM)
        rng = np.random.default_rng(random_state)
        self.weights = rng.uniform(-0.5, 0.5, n_inputs)
        self.bias = rng.uniform(-0.5, 0.5)
        self.mse_per_epoch = []

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    @staticmethod
    def activation(u):
        """Função degrau bipolar: saída em {-1, +1}."""
        return np.where(u >= 0, 1, -1)

    def predict(self, X):
        return self.activation(self.net_input(X))

    def fit(self, X, d, verbose=True):
        """
        Treina o Adaline usando a Regra Delta (LMS), amostra a amostra.

        Parâmetros
        ----------
        X : ndarray (n_amostras, n_entradas)
        d : ndarray (n_amostras,) -> saídas desejadas (ex: -1 ou +1)
        """
        eqm_anterior = np.inf

        for epoch in range(self.epochs):
            erro_quadratico_total = 0
            for xi, target in zip(X, d):
                u = self.net_input(xi)          # saída linear (sem ativação)
                erro = target - u
                self.weights += self.learning_rate * erro * xi
                self.bias += self.learning_rate * erro
                erro_quadratico_total += erro ** 2

            eqm = erro_quadratico_total / len(X)  # Erro Quadrático Médio
            self.mse_per_epoch.append(eqm)

            if verbose:
                print(f"Época {epoch + 1:3d}/{self.epochs} - EQM: {eqm:.6f}")

            # Critério de parada: variação do EQM entre épocas menor que tol
            if abs(eqm_anterior - eqm) < self.tol:
                if verbose:
                    print(f"\n>> Convergência atingida na época {epoch + 1} (variação do EQM < {self.tol})!\n")
                break
            eqm_anterior = eqm
        return self

    def plot_eqm(self):
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.mse_per_epoch) + 1), self.mse_per_epoch, marker='o', color='darkorange')
        plt.title("Adaline - Erro Quadrático Médio (EQM) por época")
        plt.xlabel("Época")
        plt.ylabel("EQM")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("adaline_convergencia.png", dpi=120)
        print("Gráfico salvo em 'adaline_convergencia.png'")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Exemplo: Porta lógica AND (saídas bipolares -1 / +1)
    # ------------------------------------------------------------
    X = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
    ])
    d = np.array([-1, -1, -1, 1])  # saída da porta AND (bipolar)

    print("=" * 60)
    print("Treinando Adaline para a porta lógica AND (entradas bipolares)")
    print("=" * 60)

    adaline = Adaline(n_inputs=2, learning_rate=0.05, epochs=100, tol=1e-6)
    adaline.fit(X, d)

    print("Pesos finais :", adaline.weights)
    print("Bias final   :", adaline.bias)

    print("\nTestando a rede treinada:")
    for xi, target in zip(X, d):
        saida = adaline.predict(xi)
        print(f"Entrada: {xi} -> Saída da rede: {saida} | Esperado: {target}")

    adaline.plot_eqm()
