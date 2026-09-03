# Redes-Neurais-Artificiais

Implementações didáticas de Redes Neurais Artificiais clássicas, desenvolvidas do zero em Python (apenas com `numpy` e `matplotlib`, sem frameworks de Deep Learning), para fins de estudo dos fundamentos matemáticos e algorítmicos de cada modelo.

Cada script é independente, comentado e inclui um exemplo de execução (`if __name__ == "__main__":`) que treina a rede e demonstra seu funcionamento.

---

## 📁 Estrutura do repositório

```
Redes-Neurais-Artificiais/
├── 01_perceptron/
│   └── perceptron.py
├── 02_adaline/
│   └── adaline.py
├── 03_perceptron_multicamada/
│   └── mlp.py
├── 04_mimo/
│   └── mimo.py
├── 05_lvq/
│   └── lvq.py
├── 06_rbf/
│   └── rbf.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧠 Redes implementadas

| # | Rede | Descrição | Tipo de aprendizagem |
|---|---|---|---|
| 01 | **Perceptron** | Rede de camada única (Rosenblatt), para classificação binária de problemas linearmente separáveis. Ajusta os pesos com a Regra do Perceptron. | Supervisionada |
| 02 | **Adaline** (ADAptive LInear NEuron) | Proposta por Widrow e Hoff. Ajusta os pesos com base no erro sobre a saída linear (net input), usando a Regra Delta (LMS - Least Mean Squares). | Supervisionada |
| 03 | **MLP** (Perceptron Multicamadas) | Rede totalmente conectada com número configurável de camadas ocultas, treinada com Retropropagação do Erro (Backpropagation) e ativação sigmoide. | Supervisionada |
| 04 | **MIMO** (Múltiplas Entradas e Saídas) | Generalização de camada única para múltiplas saídas simultâneas, treinada com a Regra Delta (matriz de pesos conectando todas entradas a todos os neurônios de saída). | Supervisionada |
| 05 | **LVQ** (Learning Vector Quantization) | Rede de Kohonen baseada em protótipos e aprendizagem competitiva: aproxima o protótipo vencedor da amostra quando a classe está correta, e o afasta quando está errada. | Supervisionada |
| 06 | **RBF** (Radial Basis Function Network) | Rede híbrida com camada oculta de neurônios gaussianos (centros definidos via K-Means) e camada de saída linear treinada por Mínimos Quadrados (pseudo-inversa). | Híbrida (não supervisionada + supervisionada) |

---

## ⚙️ Instalação

```bash
git clone https://github.com/KaiKe000708/Redes-Neurais-Artificiais.git
cd Redes-Neurais-Artificiais
pip install -r requirements.txt
```

## ▶️ Como usar

Cada script pode ser executado de forma independente e já vem com um exemplo pronto (geralmente resolvendo uma porta lógica ou um pequeno conjunto de dados sintético):

```bash
python 01_perceptron/perceptron.py
```

Ao final da execução, um gráfico de convergência do treinamento é salvo automaticamente na pasta atual.

---

## 🖥️ Requisitos

- Python 3.9 ou superior
- `numpy` e `matplotlib` (ver `requirements.txt`)

---

## 📄 Licença

Este projeto está sob a licença MIT | veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## ✏️ Notas

Implementações desenvolvidas em contexto acadêmico, durante estudos de Redes Neurais Artificiais. O objetivo é fixar os fundamentos matemáticos de cada modelo através da implementação manual dos algoritmos de treinamento, sem depender de bibliotecas prontas como TensorFlow, PyTorch ou scikit-learn.
