"""
Script para obter todas as informações do ticker BTC-USD usando yfinance.

Requisitos:
    pip install yfinance

Uso:
    python btc_info.py
"""

import yfinance as yf
import json

def get_btc_info():
    ticker = yf.Ticker("BTC-USD")

    # ── 1. Informações gerais (.info) ──
    info = ticker.info

    print("=" * 60)
    print("  1. INFORMAÇÕES GERAIS (.info)")
    print("=" * 60)
    for key, value in sorted(info.items()):
        print(f"  {key}: {value}")
    print(f"\n  Total de campos: {len(info)}\n")

    # ── 2. Histórico de preços (últimos 5 dias) ──
    print("=" * 60)
    print("  2. HISTÓRICO DE PREÇOS (últimos 5 dias)")
    print("=" * 60)
    hist = ticker.history(period="5d")
    print(hist.to_string())
    print()

    # ── 3. Metadados do histórico ──
    print("=" * 60)
    print("  3. METADADOS DO HISTÓRICO")
    print("=" * 60)
    if hasattr(ticker, "history_metadata") and ticker.history_metadata:
        for key, value in ticker.history_metadata.items():
            print(f"  {key}: {value}")
    print()

    # ── 4. Ações / Dividendos / Splits ──
    print("=" * 60)
    print("  4. DIVIDENDOS")
    print("=" * 60)
    dividends = ticker.dividends
    print(dividends if not dividends.empty else "  Nenhum dividendo registrado.")
    print()

    print("=" * 60)
    print("  5. SPLITS")
    print("=" * 60)
    splits = ticker.splits
    print(splits if not splits.empty else "  Nenhum split registrado.")
    print()

    # ── 5. Fast Info ──
    print("=" * 60)
    print("  6. FAST INFO")
    print("=" * 60)
    fast_info = ticker.fast_info
    fast_info_attrs = [
        "currency", "day_high", "day_low", "exchange",
        "fifty_day_average", "last_price", "last_volume",
        "market_cap", "open", "previous_close",
        "regular_market_previous_close", "ten_day_average_volume",
        "three_month_average_volume", "timezone",
        "two_hundred_day_average", "year_change", "year_high", "year_low",
    ]
    for attr in fast_info_attrs:
        try:
            print(f"  {attr}: {getattr(fast_info, attr)}")
        except Exception:
            pass
    print()

    # ── 6. Opções (se disponíveis) ──
    print("=" * 60)
    print("  7. DATAS DE EXPIRAÇÃO DE OPÇÕES")
    print("=" * 60)
    try:
        options = ticker.options
        if options:
            for opt in options:
                print(f"  {opt}")
        else:
            print("  Nenhuma opção disponível.")
    except Exception:
        print("  Opções não disponíveis para este ticker.")
    print()

    # ── 7. Exportar tudo como JSON ──
    output_file = "btc_usd_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    print(f"Informações completas salvas em: {output_file}")


if __name__ == "__main__":
    get_btc_info()

    ticker = yf.Ticker("BTC-USD")
    hist = ticker.history(period="max")

    # Salva o histórico de preços em JSON (sobrescreve a cada execução)
    hist_file = "btc_usd_history.json"
    hist.index = hist.index.strftime("%Y-%m-%d %H:%M:%S")
    hist_dict = hist.to_dict(orient="index")
    with open(hist_file, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"Histórico de preços salvo em: {hist_file}")