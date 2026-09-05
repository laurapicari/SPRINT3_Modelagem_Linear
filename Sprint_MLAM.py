import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(42)
dados = pd.DataFrame({
    'X': np.linspace(1, 100, 100),
    'Y': 2.5 * np.linspace(1, 100, 100) + np.random.normal(0, 15, 100)
})

print("=" * 50)
print("Probabilidade acima da Mediana:")
print("Cálculo da mediana:")
mediana = dados['Y'].median()
print(f"Mediana: {mediana:.4f}")
print("=" * 50)

print("Cálculo da probabilidade:")
prob_acima_mediana = (dados['Y'] > mediana).mean()
print(f"Probabilidade: {prob_acima_mediana:.4f} ({prob_acima_mediana * 100:.2f}%)")
print("=" * 50)

print("Classificação do Evento")
if prob_acima_mediana < 0.05:
    classificacao_q1 = "Raro"
elif prob_acima_mediana < 0.30:
    classificacao_q1 = "Pouco provável"
elif prob_acima_mediana <= 0.70:
    classificacao_q1 = "Provável"
else:
    classificacao_q1 = "Quase certo"

print(f"Classificação: {classificacao_q1}\n")
print("=" * 50)

print("Probabilidade dentro do intervalo")

media = dados['Y'].mean()
desvio_padrao = dados['Y'].std()

limite_inferior = media - 2 * desvio_padrao
limite_superior = media + 2 * desvio_padrao

print(f"""
Média: {media:.4f}
Desvio Padrão: {desvio_padrao:.4f}
Intervalo: [{limite_inferior:.4f}, {limite_superior:.4f}]""")

print("Cálculo da probabilidade dentro do intervalo")
dentro_intervalo = (dados['Y'] >= limite_inferior) & (dados['Y'] <= limite_superior)
prob_intervalo = dentro_intervalo.mean()
print(f"Probabilidade: {prob_intervalo:.4f} ({prob_intervalo * 100:.2f}%)")
print("=" * 50)

print("Classificação do evento")
if prob_intervalo < 0.05:
    classificacao_q2 = "Raro"
elif prob_intervalo < 0.30:
    classificacao_q2 = "Pouco provável"
elif prob_intervalo <= 0.70:
    classificacao_q2 = "Provável"
else:
    classificacao_q2 = "Quase certo"

print(f"Classificação: {classificacao_q2}\n")
print("=" * 50)

print("Modelagem com Regressão Linear")

print("Código da Regressão Linear")
X = dados[['X']]
y = dados['Y']

modelo = LinearRegression()
modelo.fit(X, y)

intercepto = modelo.intercept_
inclinacao = modelo.coef_[0]

print("=" * 50)

print(f"""
Modelo treinado com sucesso.
Intercepto: {intercepto:.4f}
Inclinação: {inclinacao:.4f}""")

print("Gráfico da Reta Ajustada")
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

print("=" * 50)

print("Interpretação dos coeficientes")
print("\nInterpretação dos Coeficientes:")
print(f"Intercepto = {intercepto:.4f}: Representa o valor esperado de Y quando X for igual a 0.")
print(f"Coeficiente de Inclinação = {inclinacao:.4f}: Indica que para cada aumento de 1 unidade em X, espera-se uma alteração média de {inclinacao:.4f} unidades em Y.")