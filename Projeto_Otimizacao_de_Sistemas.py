#Importação das bibliotecas
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Função para calcular o retorno esperado e a matriz de covariância
def obter_dados(tickers):
    dados = yf.download(tickers, start="2020-01-01", end="2023-01-01")['Adj Close']
    retornos = dados.pct_change().dropna()
    media_retorno = retornos.mean() * 252  # Retorno anualizado
    matriz_covariancia = retornos.cov() * 252  # Covariância anualizada
    return media_retorno, matriz_covariancia, retornos

# Função para calcular a fronteira eficiente
def fronteira_eficiente(media_retorno, matriz_covariancia):
    n = len(media_retorno)

    # Função objetivo (minimizar a volatilidade)
    def objetivo(pesos):
        return np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia, pesos)))

    # Restrição: soma dos pesos deve ser 1
    cons = {'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1}

    # Criar lista para armazenar retornos e riscos ao longo da fronteira eficiente
    retorno_eficiente = []
    risco_eficiente = []

    # Calcular fronteira eficiente variando o retorno alvo
    for retorno_alvo in np.linspace(min(media_retorno), max(media_retorno), 100):
        limite_retorno = {'type': 'eq', 'fun': lambda pesos: np.dot(pesos, media_retorno) - retorno_alvo}
        resultado = minimize(objetivo, n * [1./n], method='SLSQP', bounds=[(0, 1)] * n, constraints=[cons, limite_retorno])
        if resultado.success:
            retorno_eficiente.append(retorno_alvo)
            risco_eficiente.append(objetivo(resultado.x))

    # Plotar a fronteira eficiente
    plt.figure(figsize=(10, 6))
    plt.plot(risco_eficiente, retorno_eficiente, 'g--', linewidth=3)
    plt.title('Fronteira Eficiente')
    plt.xlabel('Risco (Desvio Padrão)')
    plt.ylabel('Retorno Esperado')
    plt.grid(True)
    plt.show()

# Função principal do código
def main():
    tickers = ['ABEV3.SA', 'BBDC4.SA', 'BRFS3.SA', 'PETR4.SA', 'TAEE11.SA']

    
    # Obter dados de retorno e covariância
    media_retorno, matriz_covariancia, retornos = obter_dados(tickers)

    # Limitar o total de investimentos a 1000
    limite_total = 1000
    total_investido = 0
    pesos_entrada = []

    print(f"O valor total disponível para investir é: R$ {limite_total:.2f}")

    # Permitindo a entrada de valores individuais, limitando o valor a 1000
    for ticker in tickers:
        while True:
            try:
                investimento = float(input(f"Quanto deseja investir em {ticker} (R$): "))
                if total_investido + investimento > limite_total:
                    print(f"O total investido não pode ultrapassar R$ {limite_total:.2f}. Você ainda pode investir até R$ {limite_total - total_investido:.2f}.")
                else:
                    total_investido += investimento
                    pesos_entrada.append(investimento)
                    break
            except ValueError:
                print("Por favor, insira um valor numérico válido.")
    
    print(f"\nO total investido foi: R$ {total_investido:.2f}")

    # Calcular pesos com base no total investido
    pesos = [p / total_investido for p in pesos_entrada]

    print("\nPesos da carteira:")
    for ticker, peso in zip(tickers, pesos):
        print(f"{ticker}: {peso * 100:.2f}%")

    # Plotar a fronteira eficiente
    fronteira_eficiente(media_retorno, matriz_covariancia)

    # Gráfico de pizza para ilustrar a distribuição/alocação de ativos
    plt.figure(figsize=(8, 6))
    plt.pie(pesos, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Garantir que o gráfico seja um círculo
    plt.title('Alocação de Ativos')
    plt.show()

    # Histórico de desempenho da carteira
    desempenho_carteira = retornos @ pesos  # Retorno ponderado da carteira
    desempenho_acumulado = (1 + desempenho_carteira).cumprod()  # Retorno acumulado

    plt.figure(figsize=(10, 6))
    plt.plot(desempenho_acumulado, label='Retorno Acumulado')
    plt.title('Histórico de Desempenho da Carteira')
    plt.xlabel('Dias')
    plt.ylabel('Retorno Acumulado')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
