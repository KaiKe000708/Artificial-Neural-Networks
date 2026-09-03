"""
=====================================================================
 LVQ - LEARNING VECTOR QUANTIZATION
=====================================================================
Implementação didática da rede LVQ (Kohonen), um método de
aprendizagem SUPERVISIONADA baseado em protótipos (vetores de
referência) e aprendizagem competitiva.

Ideia geral:
  1. Cada classe é representada por um ou mais protótipos (vetores).
  2. Para cada amostra de treinamento, encontra-se o protótipo mais
     próximo (vencedor), geralmente pela distância euclidiana.
  3. Se o protótipo vencedor pertence à MESMA classe da amostra, ele
     é aproximado da amostra (recompensa):
         w_vencedor = w_vencedor + eta * (x - w_vencedor)
  4. Se pertence a uma classe DIFERENTE, ele é afastado da amostra
     (punição):
         w_vencedor = w_vencedor - eta * (x - w_vencedor)
  5. A taxa de aprendizagem eta decai ao longo das épocas.

Após o treinamento, uma nova amostra é classificada atribuindo-se a
classe do protótipo mais próximo dela (regra do vizinho mais próximo
usando os protótipos aprendidos).

=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class LVQ:
    """Rede LVQ (Learning Vector Quantization) - versão LVQ1."""

    def __init__(self, n_prototypes_per_class=1, learning_rate=0.1, epochs=50,
                 lr_decay=0.95, random_state=42):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate_inicial = learning_rate
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.random_state = random_state
        self.prototypes = None          # vetores de referência (protótipos)
        self.prototype_labels = None    # classe de cada protótipo
        self.history_lr = []

    def _inicializar_prototipos(self, X, y):
        """Inicializa os protótipos escolhendo amostras aleatórias de cada classe."""
        rng = np.random.default_rng(self.random_state)
        classes = np.unique(y)
        prototypes = []
        labels = []

        for classe in classes:
            X_classe = X[y == classe]
            indices = rng.choice(len(X_classe), size=self.n_prototypes_per_class, replace=False)
            for idx in indices:
                prototypes.append(X_classe[idx].copy())
                labels.append(classe)

        self.prototypes = np.array(prototypes, dtype=float)
        self.prototype_labels = np.array(labels)

    @staticmethod
    def _distancia_euclidiana(x, prototypes):
        return np.linalg.norm(prototypes - x, axis=1)

    def fit(self, X, y, verbose=True):
        """
        Treina a rede LVQ.

        Parâmetros
        ----------
        X : ndarray (n_amostras, n_features)
        y : ndarray (n_amostras,) -> rótulos das classes
        """
        self._inicializar_prototipos(X, y)
        eta = self.learning_rate_inicial

        for epoch in range(self.epochs):
            indices_embaralhados = np.random.default_rng(self.random_state + epoch).permutation(len(X))

            for idx in indices_embaralhados:
                xi, yi = X[idx], y[idx]

                # Encontra o protótipo vencedor (mais próximo)
                distancias = self._distancia_euclidiana(xi, self.prototypes)
                vencedor = np.argmin(distancias)

                # Atualização: aproxima se a classe bate, afasta caso contrário
                if self.prototype_labels[vencedor] == yi:
                    self.prototypes[vencedor] += eta * (xi - self.prototypes[vencedor])
                else:
                    self.prototypes[vencedor] -= eta * (xi - self.prototypes[vencedor])

            self.history_lr.append(eta)
            eta *= self.lr_decay  # decaimento da taxa de aprendizagem

            if verbose and (epoch + 1) % 10 == 0:
                acc = self.score(X, y)
                print(f"Época {epoch + 1:3d}/{self.epochs} - eta: {eta:.4f} - acurácia treino: {acc:.2%}")

        return self

    def predict(self, X):
        """Classifica cada amostra atribuindo-a à classe do protótipo mais próximo."""
        predicoes = []
        for xi in X:
            distancias = self._distancia_euclidiana(xi, self.prototypes)
            vencedor = np.argmin(distancias)
            predicoes.append(self.prototype_labels[vencedor])
        return np.array(predicoes)

    def score(self, X, y):
        """Retorna a acurácia da rede em um conjunto de dados."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def plot_2d(self, X, y):
        """Plota as amostras e os protótipos aprendidos (apenas para dados 2D)."""
        if X.shape[1] != 2:
            print("Plot 2D disponível apenas para dados com 2 características.")
            return

        plt.figure(figsize=(6, 5))
        for classe in np.unique(y):
            plt.scatter(X[y == classe, 0], X[y == classe, 1], label=f"Classe {classe}", alpha=0.5)

        plt.scatter(self.prototypes[:, 0], self.prototypes[:, 1],
                    c='black', marker='X', s=200, label='Protótipos')
        plt.title("LVQ - Amostras e Protótipos aprendidos")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("lvq_prototipos.png", dpi=120)
        print("Gráfico salvo em 'lvq_prototipos.png'")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Exemplo: duas nuvens de pontos (classes 0 e 1) geradas
    # artificialmente em torno de dois centros distintos.
    # ------------------------------------------------------------
    rng = np.random.default_rng(1)
    classe0 = rng.normal(loc=[2, 2], scale=0.6, size=(30, 2))
    classe1 = rng.normal(loc=[6, 6], scale=0.6, size=(30, 2))

    X = np.vstack([classe0, classe1])
    y = np.array([0] * 30 + [1] * 30)

    print("=" * 60)
    print("Treinando LVQ para classificar duas nuvens de pontos (2 classes)")
    print("=" * 60)

    lvq = LVQ(n_prototypes_per_class=2, learning_rate=0.1, epochs=50, lr_decay=0.95)
    lvq.fit(X, y)

    acc_final = lvq.score(X, y)
    print(f"\nAcurácia final no conjunto de treino: {acc_final:.2%}")

    lvq.plot_2d(X, y)
