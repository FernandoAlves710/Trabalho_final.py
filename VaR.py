import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Função para calcular o VaR histórico
def historical_var(returns, confidence_level):
    var = np.percentile(returns, 100 * (1 - confidence_level))
    return var

# Função para calcular o VaR paramétrico
def parametric_var(returns, confidence_level):
    mean_return = returns.mean()
    std_return = returns.std()
    var = mean_return - std_return * stats.norm.ppf(confidence_level)
    return var

# Função para calcular o VaR de simulação Monte Carlo
def monte_carlo_var(returns, confidence_level, num_simulations=10000):
    log_returns = np.log(1 + returns)
    mean_return = log_returns.mean()
    std_return = log_returns.std()

    simulations = np.random.normal(mean_return, std_return, num_simulations)
    var = np.percentile(simulations, 100 * (1 - confidence_level))
    return var

# Função para realizar o backtest do VaR
def backtest_var(returns, var_series):
    breaches = (returns < var_series).sum()
    return breaches

# Configuração da página
st.title("Calculadora de VaR com Backtest")
st.sidebar.header("Configurações")

# Seleção das ações do Yahoo Finance
stocks = st.sidebar.multiselect(
    'Selecione as ações:',
    ('AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'BRK-A', 'V', 'JNJ', 'WMT'),
    ('AAPL', 'MSFT')
)

# Input de exposição (valor aplicado)
investment = st.sidebar.number_input("Exposição (valor aplicado):", min_value=0.0, value=10000.0)

# Input de intervalo de confiança
confidence_level = st.sidebar.slider("Intervalo de Confiança:", min_value=0.90, max_value=0.99, value=0.95, step=0.01)

# Input de período de retenção
holding_period = st.sidebar.number_input("Período de Retenção (dias):", min_value=1, max_value=252, value=10)

# Escolha do tipo de VaR
var_type = st.sidebar.selectbox("Tipo de VaR:", ("Histórico", "Paramétrico", "Monte Carlo"))

# Download dos dados
if stocks:
    try:
        data = yf.download(stocks, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))['Adj Close']
        
        if data.empty:
            st.error("Erro ao baixar os dados: Nenhum dado retornado.")
        else:
            st.header("Dados das Ações Selecionadas")
            st.line_chart(data)
            
            # Calcular os retornos diários
            returns = data.pct_change().dropna()
            
            # Cálculo do VaR
            if var_type == "Histórico":
                var_function = historical_var
            elif var_type == "Paramétrico":
                var_function = parametric_var
            else:
                var_function = monte_carlo_var

            var_values = returns.apply(var_function, confidence_level=confidence_level)
            var_values_adjusted = var_values * np.sqrt(holding_period)
            var_value = investment * var_values_adjusted.mean()
            var_percent = var_values_adjusted.mean() * 100

            st.write(f"O Valor em Risco (VaR) é de: R$ {var_value:,.2f} ({var_percent:.2f}%) para o tipo de VaR {var_type}")

            # Backtest do VaR
            st.header("Backtest do VaR")
            fig, ax = plt.subplots(figsize=(10, 6))

            for stock in stocks:
                stock_returns = returns[stock]
                stock_var_series = stock_returns.rolling(window=holding_period).apply(var_function, kwargs={'confidence_level': confidence_level}).dropna()

                # Alinhando os índices de returns e var_series para comparações
                aligned_stock_returns = stock_returns.loc[stock_var_series.index]

                breaches = backtest_var(aligned_stock_returns, stock_var_series)

                aligned_stock_returns.plot(ax=ax, label=f'Retorno Diário {stock}', linewidth=2)
                stock_var_series.plot(ax=ax, label=f'VaR {stock}', linestyle='--', color='red', linewidth=2)

                st.write(f"Número de violações para {stock}: {breaches}")

            ax.set_title("Backtest do VaR", fontsize=16)
            ax.set_xlabel("Data", fontsize=14)
            ax.set_ylabel("Retorno / VaR", fontsize=14)
            ax.legend(fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao baixar os dados: {e}")
else:
    st.warning("Por favor, selecione pelo menos uma ação.")
