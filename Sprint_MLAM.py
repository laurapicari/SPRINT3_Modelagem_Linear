import numpy as np
import pandas as pd
import scipy.stats.norm as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==============================================================================
# CARREGAMENTO DOS DADOS
# Substitua a linha abaixo para carregar seu arquivo (ex: pd.read_csv('seu_arquivo.csv'))
# ==============================================================================
np.random.seed(42)
pd.read_csv('base.Dados02')
dados = pd.DataFrame({
    'X': np.linspace(1, 100, 100),
    'Y': 2.5 * np.linspace(1, 100, 100) + np.random.normal(0, 15, 100)
})

# ==============================================================================
# QUESTÃO 01: Probabilidade acima da Mediana (2,5 pontos)
# ==============================================================================
print("--- QUESTÃO 01 ---")

# a) Cálculo da mediana
mediana = dados['Y'].median()
print(f"a) Mediana de Y: {mediana:.4f}")

# b) Cálculo da probabilidade P(Y > Mediana)
prob_acima_mediana = (dados['Y'] > mediana).mean()
print(f"b) Probabilidade P(Y > Mediana): {prob_acima_mediana:.4f} ({prob_acima_mediana * 100:.2f}%)")

# d) Classificação do evento
# Padrão de classificação: Raro (<5%), Pouco provável (5%-30%), Provável (30%-70%), Quase certo (>70%)
if prob_acima_mediana < 0.05:
    classificacao_q1 = "Raro"
elif prob_acima_mediana < 0.30:
    classificacao_q1 = "Pouco provável"
elif prob_acima_mediana <= 0.70:
    classificacao_q1 = "Provável"
else:
    classificacao_q1 = "Quase certo"

print(f"d) Classificação do evento: {classificacao_q1}\n")

# ==============================================================================
# QUESTÃO 02: Probabilidade dentro do intervalo (Média ± 2s) (2,5 pontos)
# ==============================================================================
print("--- QUESTÃO 02 ---")

# a) Cálculo da média, desvio padrão e intervalo
media = dados['Y'].mean()
desvio_padrao = dados['Y'].std()

limite_inferior = media - 2 * desvio_padrao
limite_superior = media + 2 * desvio_padrao

print(f"a) Média: {media:.4f}")
print(f"   Desvio Padrão (s): {desvio_padrao:.4f}")
print(f"   Intervalo (Média ± 2s): [{limite_inferior:.4f}, {limite_superior:.4f}]")

# b) Cálculo da probabilidade empírica dentro do intervalo
dentro_intervalo = (dados['Y'] >= limite_inferior) & (dados['Y'] <= limite_superior)
prob_intervalo = dentro_intervalo.mean()
print(f"b) Probabilidade P(Média - 2s <= Y <= Média + 2s): {prob_intervalo:.4f} ({prob_intervalo * 100:.2f}%)")

# d) Classificação do evento
if prob_intervalo < 0.05:
    classificacao_q2 = "Raro"
elif prob_intervalo < 0.30:
    classificacao_q2 = "Pouco provável"
elif prob_intervalo <= 0.70:
    classificacao_q2 = "Provável"
else:
    classificacao_q2 = "Quase certo"

print(f"d) Classificação do evento: {classificacao_q2}\n")

# ==============================================================================
# QUESTÃO 03: Modelagem com Regressão Linear (3,0 pontos)
# ==============================================================================
print("--- QUESTÃO 03 ---")

# a) Código da Regressão Linear
X = dados[['X']]  # Variável independente (Matriz)
y = dados['Y']    # Variável dependente (Vetor)

modelo = LinearRegression()
modelo.fit(X, y)

intercepto = modelo.intercept_
inclinacao = modelo.coef_[0]

print(f"a) Modelo treinado com sucesso.")
print(f"   Intercepto (beta_0): {intercepto:.4f}")
print(f"   Inclinação (beta_1): {inclinacao:.4f}")

# b) Gráfico da Reta Ajustada
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.6, label='Dados Observados')
plt.plot(X, modelo.predict(X), color='red', linewidth=2, label=f'Reta Ajustada: Y = {intercepto:.2f} + {inclinacao:.2f}X')
plt.title('Regressão Linear Simples', fontsize=14)
plt.xlabel('Variável Independente (X)', fontsize=12)
plt.ylabel('Variável Dependente (Y)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# c) Interpretação dos coeficientes
print("\nc) Interpretação dos Coeficientes:")
print(f"   - Intercepto (beta_0 = {intercepto:.4f}): Representa o valor esperado de Y quando X for igual a 0.")
print(f"   - Coeficiente de Inclinação (beta_1 = {inclinacao:.4f}): Indica que para cada aumento de 1 unidade em X, espera-se uma alteração média de {inclinacao:.4f} unidades em Y.")