import streamlit as st
import numpy as np
from scipy.stats import norm

# Black-Scholes com dividendos
def bs_call_price(S, K, T, r, sigma, q=0.0):
    if T == 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Volatilidade implícita com dividendos
def implied_vol_bisection(C_market, S, K, T, r, q=0.0, tol=1e-5, max_iter=100):
    low = 1e-5
    high = 3.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = bs_call_price(S, K, T, r, mid, q)
        if abs(price - C_market) < tol:
            return mid
        elif price > C_market:
            high = mid
        else:
            low = mid
    return mid

# Monte Carlo para europeia e asiática
def monte_carlo_option_price(S, K, T, r, sigma, tipo='europeia', q=0.0, num_sim=10000):
    np.random.seed(42)

    if tipo == 'europeia':
        ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.randn(num_sim))
        payoff = np.maximum(ST - K, 0)

    elif tipo == 'asiatica':
        steps = max(1, int(T * 252))
        dt = T / steps
        payoff = []
        for _ in range(num_sim):
            prices = [S]
            for _ in range(steps):
                drift = (r - q - 0.5 * sigma ** 2) * dt
                shock = sigma * np.sqrt(dt) * np.random.randn()
                prices.append(prices[-1] * np.exp(drift + shock))
            ST_avg = np.mean(prices)
            payoff.append(max(ST_avg - K, 0))

    return np.exp(-r * T) * np.mean(payoff)

# Binomial americano
def binomial_american_call(S, K, T, r, sigma, q=0.0, steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    payoff = np.maximum(ST - K, 0)

    for i in range(steps - 1, -1, -1):
        ST = ST[:-1] / u
        payoff = np.exp(-r * dt) * (p * payoff[1:] + (1 - p) * payoff[:-1])
        payoff = np.maximum(payoff, ST - K)

    return payoff[0]

# STREAMLIT APP
st.title("📈 Calculadora de Opções e Volatilidade Implícita")

opcao = st.sidebar.selectbox("Escolha a operação", ["Preço da Opção", "Volatilidade Implícita"])

if opcao == "Preço da Opção":
    tipo = st.selectbox("Tipo de Opção", ["Europeia", "Asiática", "Americana"])
    S = st.number_input("Preço do ativo (S)", min_value=0.01)
    K = st.number_input("Strike (K)", min_value=0.01)
    T = st.number_input("Tempo até o vencimento (em anos)", min_value=0.01, value=0.12)
    r = st.number_input("Taxa de juros anual (r)", min_value=0.0, value=0.1475)
    q = st.number_input("Dividend yield anual (q)", min_value=0.0, value=0.00)
    sigma = st.number_input("Volatilidade (σ)", min_value=0.01, value=0.3)

    if st.button("Calcular Preço"):
        if tipo == 'Europeia':
            preco = bs_call_price(S, K, T, r, sigma, q)
        elif tipo == 'Asiática':
            preco = monte_carlo_option_price(S, K, T, r, sigma, tipo='asiatica', q=q)
        elif tipo == 'Americana':
            preco = binomial_american_call(S, K, T, r, sigma, q=q)
        st.success(f"💰 Preço estimado da opção {tipo.lower()}: R$ {preco:.4f}")

elif opcao == "Volatilidade Implícita":
    C_market = st.number_input("Preço de mercado da opção (C)", min_value=0.01)
    S = st.number_input("Preço do ativo (S)", min_value=0.01)
    K = st.number_input("Strike (K)", min_value=0.01)
    T = st.number_input("Tempo até o vencimento (em anos)", min_value=0.01, value=0.12)
    r = st.number_input("Taxa de juros anual (r)", min_value=0.0, value=0.1475)
    q = st.number_input("Dividend yield anual (q)", min_value=0.0, value=0.00)

    if st.button("Calcular Volatilidade"):
        vol = implied_vol_bisection(C_market, S, K, T, r, q)
        st.success(f"📊 Volatilidade implícita estimada: {vol:.4f}")
