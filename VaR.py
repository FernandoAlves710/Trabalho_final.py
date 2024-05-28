import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def calculate_var(returns, confidence_level):
    var = np.percentile(returns, 100 * (1 - confidence_level))
    return var

def backtest_var(returns, var_series):
    breaches = (returns < var_series).sum()
    return breaches

st.title("Calculadora de VaR com Backtest")
st.sidebar.header("Configurações")

stocks = st.sidebar.multiselect(
    'Selecione as ações:',
    ('AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'BRK-A', 'V', 'JNJ', 'WMT'),
    ('AAPL', 'MSFT')
)

investment = st.sidebar.number_input("Exposição (valor aplicado):", min_value=0.0, value=10000.0)

confidence_level = st.sidebar.slider("Intervalo de Confiança:", min_value=0.90, max_value=0.99, value=0.95, step=0.01)

holding_period = st.sidebar.number_input("Período de Retenção (dias):", min_value=1, max_value=252, value=10)

if stocks:
    try:
        data = yf.download(stocks, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))['Adj Close']
        
        if data.empty:
            st.error("Erro ao baixar os dados: Nenhum dado retornado.")
        else:
            st.header("Dados das Ações Selecionadas")
            st.line_chart(data)
            
            returns = data.pct_change().dropna()
            
            var_values = returns.apply(calculate_var, confidence_level=confidence_level)
            var_values_adjusted = var_values * np.sqrt(holding_period)
            var_value = investment * var_values_adjusted.mean()

            st.write(f"O Valor em Risco (VaR) é de: R$ {var_value:,.2f}")

            st.header("Backtest do VaR")
            var_series = returns.rolling(window=holding_period).apply(calculate_var, kwargs={'confidence_level': confidence_level}).dropna()

            aligned_returns = returns.loc[var_series.index]

            breaches = backtest_var(aligned_returns, var_series)

            fig, ax = plt.subplots()
            aligned_returns.plot(ax=ax, label='Retornos Diários')
            var_series.plot(ax=ax, label='VaR', color='red')
            ax.legend()
            st.pyplot(fig)

            st.write(f"Número de violações: {breaches}")
    except Exception as e:
        st.error(f"Erro ao baixar os dados: {e}")
else:
    st.warning("Por favor, selecione pelo menos uma ação.")
