"""
=====================================================================
 RBF - REDE NEURAL COM FUNÇÃO DE BASE RADIAL
=====================================================================
Implementação didática de uma Rede Neural de Base Radial (RBF - Radial
Basis Function Network), composta por:

  Camada 1 (entrada): apenas repassa as variáveis de entrada.
  Camada 2 (oculta)  : neurônios com função de ativação gaussiana,
                        cada um centrado em um protótipo (centro).
                        A saída de cada neurônio "j" para uma entrada
                        x é dada por:

            phi_j(x) = exp( - ||x - c_j||^2 / (2 * sigma_j^2) )

  Camada 3 (saída)   : combinação LINEAR das saídas da camada oculta
                        (pesos treinados por mínimos quadrados).

Etapas do treinamento (abordagem híbrida clássica):
  1. Definição dos centros (c_j) das funções radiais via K-Means.
  2. Definição da largura (sigma_j) de cada gaussiana, geralmente
     usando a distância média aos centros vizinhos.
  3. Cálculo da matriz de ativações da camada oculta (Phi).
  4. Treinamento da camada de saída (linear) via Mínimos Quadrados
     (pseudo-inversa), resolvendo: W = Phi^+ . D

=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


class RBF:
    """Rede Neural de Base Radial (RBF) com centros definidos por K-Means."""

    def __init__(self, n_centros, sigma=None, random_state=42):
        """
        Parâmetros
        ----------
        n_centros : int
            Número de neurônios da camada oculta (funções radiais).
        sigma : float ou None
            Largura das gaussianas. Se None, é calculada automaticamente
            com base na distância máxima entre os centros.
        """
        self.n_centros = n_centros
        self.sigma = sigma
        self.random_state = random_state
        self.centros = None
        self.pesos_saida = None

    # ---------- Etapa 1: K-Means simples para encontrar os centros ----------
    def _kmeans(self, X, max_iter=100, tol=1e-4):
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.n_centros, replace=False)
        centros = X[indices].copy()

        for _ in range(max_iter):
            # Atribui cada ponto ao centro mais próximo
            distancias = np.linalg.norm(X[:, None, :] - centros[None, :, :], axis=2)
            clusters = np.argmin(distancias, axis=1)

            novos_centros = np.array([
                X[clusters == k].mean(axis=0) if np.any(clusters == k) else centros[k]
                for k in range(self.n_centros)
            ])

            if np.linalg.norm(novos_centros - centros) < tol:
                centros = novos_centros
                break
            centros = novos_centros

        return centros

    # ---------- Etapa 2: cálculo da matriz de ativação da camada oculta ----------
    def _funcao_gaussiana(self, X):
        distancias = np.linalg.norm(X[:, None, :] - self.centros[None, :, :], axis=2)
        return np.exp(-(distancias ** 2) / (2 * self.sigma ** 2))

    def fit(self, X, D):
        """
        Treina a rede RBF.

        Parâmetros
        ----------
        X : ndarray (n_amostras, n_entradas)
        D : ndarray (n_amostras, n_saidas) -> saídas desejadas
        """
        if D.ndim == 1:
            D = D.reshape(-1, 1)

        # 1) Define os centros das funções radiais via K-Means
        self.centros = self._kmeans(X)

        # 2) Define sigma automaticamente (se não informado) usando a
        #    distância máxima entre os centros, técnica clássica:
        #    sigma = d_max / sqrt(2 * n_centros)
        if self.sigma is None:
            distancias_centros = np.linalg.norm(
                self.centros[:, None, :] - self.centros[None, :, :], axis=2
            )
            d_max = distancias_centros.max()
            self.sigma = d_max / np.sqrt(2 * self.n_centros)

        # 3) Calcula a matriz de ativação da camada oculta (Phi)
        Phi = self._funcao_gaussiana(X)

        # Adiciona uma coluna de bias (constante = 1) à matriz Phi
        Phi_bias = np.hstack([Phi, np.ones((Phi.shape[0], 1))])

        # 4) Treina a camada de saída (linear) via mínimos quadrados
        #    W = Phi^+ . D   (pseudo-inversa de Moore-Penrose)
        self.pesos_saida = np.linalg.pinv(Phi_bias).dot(D)

        return self

    def predict(self, X):
        Phi = self._funcao_gaussiana(X)
        Phi_bias = np.hstack([Phi, np.ones((Phi.shape[0], 1))])
        return Phi_bias.dot(self.pesos_saida)

    def plot_superficie_2d(self, X, D, resolucao=100):
        """Plota a superfície de decisão para problemas 2D binários (ex.: XOR)."""
        if X.shape[1] != 2:
            print("Plot disponível apenas para entradas com 2 dimensões.")
            return

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucao),
                              np.linspace(y_min, y_max, resolucao))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.7)
        plt.colorbar(label="Saída da rede")
        plt.scatter(X[:, 0], X[:, 1], c=D.ravel(), cmap='coolwarm', edgecolors='k', s=100)
        plt.scatter(self.centros[:, 0], self.centros[:, 1], c='black', marker='X', s=150, label='Centros RBF')
        plt.title("Rede RBF - Superfície de decisão")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.tight_layout()
        plt.savefig("rbf_superficie.png", dpi=120)
        print("Gráfico salvo em 'rbf_superficie.png'")


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Exemplo clássico: porta lógica XOR (problema não linearmente
    # separável, resolvido de forma elegante por uma rede RBF)
    # ------------------------------------------------------------
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)
    D = np.array([0, 1, 1, 0], dtype=float)

    print("=" * 60)
    print("Treinando rede RBF para resolver a porta lógica XOR")
    print("=" * 60)

    rbf = RBF(n_centros=4, sigma=None, random_state=42)
    rbf.fit(X, D)

    print("Centros encontrados (K-Means):\n", rbf.centros)
    print("Sigma calculado:", rbf.sigma)

    print("\nTestando a rede treinada:")
    saida = rbf.predict(X)
    for xi, di, yi in zip(X, D, saida):
        print(f"Entrada: {xi} -> Saída da rede: {yi[0]:.4f} (~{round(yi[0])}) | Esperado: {di}")

    rbf.plot_superficie_2d(X, D)
