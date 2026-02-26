"""
Análise de ciclos BTC-USD com base no histórico salvo em btc_usd_history.json.

Lógica:
  1. Encontra o maior preço (pico de alta) nos últimos 395 dias do histórico.
  2. A partir dessa data, avança 395 dias e encontra o menor preço (pico de baixa) nesse intervalo.
  3. A partir da data do pico de baixa, avança 1065 dias → essa é a previsão do próximo pico de alta.
"""

import json
from datetime import datetime, timedelta

# ── Carregar histórico ──
with open("btc_usd_history.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Converter para lista ordenada por data
prices = []
for date_str, values in data.items():
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    prices.append((dt, values["Close"]))

prices.sort(key=lambda x: x[0])

# ── 1. Maior preço nos últimos 395 dias do histórico ──
ultima_data = prices[-1][0]
inicio_janela = ultima_data - timedelta(days=395)

ultimos_395 = [(dt, close) for dt, close in prices if dt >= inicio_janela]
pico_alta = max(ultimos_395, key=lambda x: x[1])

print("=" * 60)
print("  ANÁLISE DE CICLOS BTC-USD")
print("=" * 60)
print(f"\n  1. PICO DE ALTA (últimos 395 dias)")
print(f"     Data:  {pico_alta[0].strftime('%Y-%m-%d')}")
print(f"     Preço: ${pico_alta[1]:,.2f}")

# ── 2. Pico de baixa projetado: pico de alta + 395 dias ──
data_pico_baixa = pico_alta[0] + timedelta(days=395)
print(f"\n  2. PICO DE BAIXA PROJETADO (pico de alta + 395 dias)")
print(f"     Data:  {data_pico_baixa.strftime('%Y-%m-%d')}")

# ── 3. Próximo pico de alta projetado: pico de baixa + 1065 dias ──
proximo_pico_alta = data_pico_baixa + timedelta(days=1065)
print(f"\n  3. PRÓXIMO PICO DE ALTA PROJETADO (pico de baixa + 1065 dias)")
print(f"     Data:  {proximo_pico_alta.strftime('%Y-%m-%d')}")

print()
