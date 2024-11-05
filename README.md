
# Otimização de Carteira de Investimentos

Este projeto consiste em uma aplicação Python que permite ao usuário calcular a fronteira eficiente de uma carteira de investimentos usando ações da Bovespa, além de exibir a alocação dos ativos e o histórico de desempenho da carteira. O objetivo é auxiliar no entendimento de alocação de ativos, risco e retorno de uma carteira, considerando restrições de investimento.

---

## Índice

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Funcionamento do Código](#funcionamento-do-código)
- [Estrutura do Código](#estrutura-do-código)
- [Exemplo de Uso](#exemplo-de-uso)
- [Referências](#referências)

---

## Pré-requisitos

Para executar este código, é necessário ter o Python 3 instalado, além das bibliotecas:

- `numpy`
- `pandas`
- `yfinance`
- `matplotlib`
- `scipy`

Essas bibliotecas podem ser instaladas com o seguinte comando:

```bash
pip install numpy pandas yfinance matplotlib scipy
```

## Instalação

1. Clone este repositório ou navegue até o diretório onde deseja salvar o código.
2. Certifique-se de que as bibliotecas mencionadas acima estão instaladas.
3. Salve o arquivo do código Python (por exemplo, `portfolio_optimization.py`) no diretório escolhido.

## Como Usar

1. Execute o script no terminal:
   ```bash
   python portfolio_optimization.py
   ```
2. Ao iniciar, o código exibirá o valor total disponível para investimento (R$ 1000,00).
3. O usuário deverá inserir o valor a ser investido em cada uma das ações listadas (máximo de 5 ações).
4. O código calculará automaticamente a fronteira eficiente, a distribuição de ativos e o histórico de desempenho da carteira com base nos investimentos inseridos.

---

## Funcionamento do Código

### 1. Função de Coleta de Dados
A função `obter_dados()` utiliza a biblioteca `yfinance` para baixar os dados históricos de preços ajustados das ações. A função calcula:
   - Retorno anualizado médio.
   - Matriz de covariância anualizada dos retornos.

### 2. Cálculo da Fronteira Eficiente
A função `fronteira_eficiente()` encontra a fronteira eficiente variando o retorno alvo e minimizando o risco (volatilidade). Ela plota um gráfico mostrando a relação entre risco e retorno.

### 3. Alocação de Ativos
Com base nos valores inseridos pelo usuário, o código calcula a proporção (peso) de cada ativo na carteira e exibe essa distribuição em um gráfico de pizza.

### 4. Histórico de Desempenho da Carteira
O desempenho acumulado da carteira é calculado e exibido em um gráfico de linha, representando o crescimento do investimento ao longo do tempo.

---

## Estrutura do Código

```python
# Importação das bibliotecas
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
    def objetivo(pesos):
        return np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia, pesos)))
    cons = {'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1}
    retorno_eficiente = []
    risco_eficiente = []
    for retorno_alvo in np.linspace(min(media_retorno), max(media_retorno), 100):
        limite_retorno = {'type': 'eq', 'fun': lambda pesos: np.dot(pesos, media_retorno) - retorno_alvo}
        resultado = minimize(objetivo, n * [1./n], method='SLSQP', bounds=[(0, 1)] * n, constraints=[cons, limite_retorno])
        if resultado.success:
            retorno_eficiente.append(retorno_alvo)
            risco_eficiente.append(objetivo(resultado.x))
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
    media_retorno, matriz_covariancia, retornos = obter_dados(tickers)
    limite_total = 1000
    total_investido = 0
    pesos_entrada = []
    print(f"O valor total disponível para investir é: R$ {limite_total:.2f}")
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
    pesos = [p / total_investido for p in pesos_entrada]
    print("\nPesos da carteira:")
    for ticker, peso in zip(tickers, pesos):
        print(f"{ticker}: {peso * 100:.2f}%")
    fronteira_eficiente(media_retorno, matriz_covariancia)
    plt.figure(figsize=(8, 6))
    plt.pie(pesos, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Alocação de Ativos')
    plt.show()
    desempenho_carteira = retornos @ pesos
    desempenho_acumulado = (1 + desempenho_carteira).cumprod()
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
```

---

## Exemplo de Uso

1. **Executar o Código**: Execute o código e insira os valores para cada ação quando solicitado.
2. **Visualizar Resultados**: O código gera três gráficos:
   - **Fronteira Eficiente**: Equilíbrio entre retorno e risco da carteira.
   - **Alocação de Ativos**: Distribuição dos investimentos nos ativos escolhidos.
   - **Histórico de Desempenho**: Evolução do retorno acumulado da carteira ao longo do tempo.

---

## Referências

- [Documentação do yfinance](https://github.com/ranaroussi/yfinance)
- [Numpy Documentation](https://numpy.org/doc/)
- [Scipy Optimize Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---

Espero que este guia ajude no entendimento e utilização do código. Se tiver dúvidas, consulte as referências ou entre em contato!
