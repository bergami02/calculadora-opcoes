import streamlit as st
import numpy as np
from scipy.stats import norm

# Black-Scholes para opção europeia
def bs_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Volatilidade implícita via bisseção
def implied_vol_bisection(C_market, S, K, T, r, tol=1e-5, max_iter=100):
    low = 1e-5
    high = 3.0
    for i in range(max_iter):
        mid = (low + high) / 2
        price = bs_call_price(S, K, T, r, mid)
        if abs(price - C_market) < tol:
            return mid
        elif price > C_market:
            high = mid
        else:
            low = mid
    return mid

# Monte Carlo para europeia e asiática
def monte_carlo_option_price(S, K, T, r, sigma, tipo='europeia', num_sim=10000):
    np.random.seed(42)
    dt = T
    if tipo == 'europeia':
        ST = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(num_sim))
        payoff = np.maximum(ST - K, 0)
    elif tipo == 'asiatica':
        dt = T / 100
        steps = int(T / dt)
        ST = np.zeros(num_sim)
        for i in range(num_sim):
            prices = S * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt +
                                          sigma * np.sqrt(dt) * np.random.randn(steps)))
            ST[i] = np.mean(prices)
        payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

# Modelo binomial para opção americana
def binomial_american_call(S, K, T, r, sigma, steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # preços possíveis no vencimento
    ST = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    payoff = np.maximum(ST - K, 0)

    # iterando de trás pra frente
    for i in range(steps - 1, -1, -1):
        ST = ST[:-1] / u
        payoff = np.exp(-r * dt) * (p * payoff[1:] + (1 - p) * payoff[:-1])
        payoff = np.maximum(payoff, ST - K)  # exercício antecipado
    return payoff[0]

# Interface Streamlit
st.title("Calculadora de Opções e Volatilidade Implícita")

opcao = st.sidebar.selectbox("Escolha a operação", ["Preço da Opção", "Volatilidade Implícita"])

if opcao == "Preço da Opção":
    tipo = st.selectbox("Tipo de Opção", ["Europeia", "Asiática", "Americana"])
    S = st.number_input("Preço do ativo (S)", min_value=0.01)
    K = st.number_input("Strike (K)", min_value=0.01)
    T = st.number_input("Tempo até o vencimento (em anos)", min_value=0.01)
    r = st.number_input("Taxa de juros (r)", min_value=0.0)
    sigma = st.number_input("Volatilidade (σ)", min_value=0.01)

    if st.button("Calcular Preço"):
        if tipo == 'Europeia' or tipo == 'Asiática':
            preco = monte_carlo_option_price(S, K, T, r, sigma, tipo=tipo.lower())
        elif tipo == 'Americana':
            preco = binomial_american_call(S, K, T, r, sigma)
        st.success(f"Preço estimado da opção {tipo.lower()}: R$ {preco:.2f}")

elif opcao == "Volatilidade Implícita":
    C_market = st.number_input("Preço de mercado da opção (C)", min_value=0.01)
    S = st.number_input("Preço do ativo (S)", min_value=0.01)
    K = st.number_input("Strike (K)", min_value=0.01)
    T = st.number_input("Tempo até o vencimento (em anos)", min_value=0.01)
    r = st.number_input("Taxa de juros (r)", min_value=0.0)

    if st.button("Calcular Volatilidade"):
        vol = implied_vol_bisection(C_market, S, K, T, r)
        st.success(f"Volatilidade implícita estimada: {vol:.4f}")
