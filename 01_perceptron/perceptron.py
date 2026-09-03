"""
=====================================================================
 PERCEPTRON SIMPLES (Rede de Camada Única)
=====================================================================
Implementação didática do Perceptron de Rosenblatt para problemas de
classificação binária linearmente separáveis.

Regra de aprendizagem (Regra do Perceptron):
    w(t+1) = w(t) + eta * (d - y) * x
    b(t+1) = b(t) + eta * (d - y)

Onde:
    d = saída desejada (target)
    y = saída obtida pela rede (função degrau aplicada ao net input)
    eta = taxa de aprendizagem
=====================================================================

"""

import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """Perceptron de camada única com função de ativação degrau bipolar/binária."""

    def __init__(self, n_inputs, learning_rate=0.1, epochs=100, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        rng = np.random.default_rng(random_state)
        # Pesos iniciais pequenos e aleatórios (+1 para o bias)
        self.weights = rng.uniform(-0.5, 0.5, n_inputs)
        self.bias = rng.uniform(-0.5, 0.5)
        self.errors_per_epoch = []

    @staticmethod
    def activation(u):
        """Função de ativação degrau (step function) -> saída binária {0, 1}."""
        return np.where(u >= 0, 1, 0)

    def net_input(self, X):
        """Calcula o net input (combinação linear + bias)."""
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """Realiza a predição para um conjunto de amostras X."""
        return self.activation(self.net_input(X))

    def fit(self, X, d, verbose=True):
        """
        Treina o Perceptron.

        Parâmetros
        ----------
        X : ndarray (n_amostras, n_entradas)
        d : ndarray (n_amostras,)  -> saídas desejadas
        """
        for epoch in range(self.epochs):
            erro_total = 0
            for xi, target in zip(X, d):
                y = self.predict(xi)
                erro = target - y
                # Atualização dos pesos apenas se houver erro de classificação
                if erro != 0:
                    self.weights += self.learning_rate * erro * xi
                    self.bias += self.learning_rate * erro
                    erro_total += 1
            self.errors_per_epoch.append(erro_total)
            if verbose:
                print(f"Época {epoch + 1:3d}/{self.epochs} - erros de classificação: {erro_total}")
            # Critério de parada: convergência (nenhum erro na época)
            if erro_total == 0:
                if verbose:
                    print(f"\n>> Convergência atingida na época {epoch + 1}!\n")
                break
        return self

    def plot_erros(self):
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.errors_per_epoch) + 1), self.errors_per_epoch, marker='o')
        plt.title("Perceptron - Erros de classificação por época")
        plt.xlabel("Época")
        plt.ylabel("Nº de amostras classificadas erradas")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("perceptron_convergencia.png", dpi=120)
        print("Gráfico salvo em 'perceptron_convergencia.png'")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Exemplo: Porta lógica AND (problema linearmente separável)
    # ------------------------------------------------------------
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    d = np.array([0, 0, 0, 1])  # saída da porta AND

    print("=" * 60)
    print("Treinando Perceptron para a porta lógica AND")
    print("=" * 60)

    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, epochs=20)
    perceptron.fit(X, d)

    print("Pesos finais :", perceptron.weights)
    print("Bias final   :", perceptron.bias)

    print("\nTestando a rede treinada:")
    for xi, target in zip(X, d):
        saida = perceptron.predict(xi)
        print(f"Entrada: {xi} -> Saída da rede: {saida} | Esperado: {target}")

    perceptron.plot_erros()
