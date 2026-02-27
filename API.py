"""
API.py — Coleta e armazenamento do histórico BTC-USD via yfinance.

Melhorias:
  - Atualização incremental (só busca dados novos)
  - Separação de responsabilidades em funções
  - Tratamento de erros robusto
  - Tipagem explícita

Requisitos:
    pip install yfinance
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

# ── Configuração de log ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HISTORY_FILE = Path("btc_usd_history.json")
INFO_FILE = Path("btc_usd_info.json")
TICKER_SYMBOL = "BTC-USD"


# ══════════════════════════════════════════════
#  Funções de persistência
# ══════════════════════════════════════════════

def load_history() -> dict[str, Any]:
    """Carrega o histórico salvo em disco. Retorna dict vazio se não existir."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(data: dict[str, Any]) -> None:
    """Persiste o histórico em disco."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    log.info("Histórico salvo em: %s  (%d registros)", HISTORY_FILE, len(data))


def save_info(info: dict[str, Any]) -> None:
    """Persiste as informações gerais do ticker."""
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    log.info("Informações gerais salvas em: %s", INFO_FILE)


# ══════════════════════════════════════════════
#  Funções de coleta
# ══════════════════════════════════════════════

def fetch_history(existing: dict[str, Any]) -> dict[str, Any]:
    """
    Busca o histórico de preços do BTC-USD.

    Se já houver dados locais, busca apenas a partir do último registro
    (atualização incremental). Caso contrário, baixa o histórico completo.
    """
    ticker = yf.Ticker(TICKER_SYMBOL)

    if existing:
        last_date_str = max(existing.keys())
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d %H:%M:%S")
        days_since = (datetime.now(timezone.utc).replace(tzinfo=None) - last_date).days

        if days_since <= 0:
            log.info("Histórico já está atualizado. Nenhuma busca necessária.")
            return existing

        period = f"{days_since + 5}d"  # margem de segurança
        log.info(
            "Histórico existente até %s. Buscando últimos %s dias...",
            last_date_str,
            days_since,
        )
    else:
        period = "max"
        log.info("Nenhum histórico local encontrado. Baixando histórico completo...")

    hist = ticker.history(period=period)

    if hist.empty:
        log.warning("Nenhum dado retornado pelo yfinance.")
        return existing

    hist.index = hist.index.strftime("%Y-%m-%d %H:%M:%S")
    new_data = hist.to_dict(orient="index")

    # Merge: dados existentes + novos (novos sobrescrevem em caso de sobreposição)
    merged = {**existing, **new_data}
    log.info("Novos registros adicionados: %d", len(new_data))
    return merged


def fetch_info() -> dict[str, Any]:
    """Retorna as informações gerais do ticker (campos .info do yfinance)."""
    ticker = yf.Ticker(TICKER_SYMBOL)
    info = ticker.info
    log.info("Informações gerais obtidas (%d campos).", len(info))
    return info


def fetch_fast_info() -> dict[str, Any]:
    """Retorna os principais indicadores de mercado em tempo real."""
    ticker = yf.Ticker(TICKER_SYMBOL)
    fi = ticker.fast_info

    attrs = [
        "currency", "day_high", "day_low", "exchange",
        "fifty_day_average", "last_price", "last_volume",
        "market_cap", "open", "previous_close",
        "regular_market_previous_close", "ten_day_average_volume",
        "three_month_average_volume", "timezone",
        "two_hundred_day_average", "year_change", "year_high", "year_low",
    ]

    result: dict[str, Any] = {}
    for attr in attrs:
        try:
            result[attr] = getattr(fi, attr)
        except AttributeError:
            pass

    return result


# ══════════════════════════════════════════════
#  Funções de exibição
# ══════════════════════════════════════════════

def display_fast_info(fast_info: dict[str, Any]) -> None:
    """Exibe os principais indicadores de mercado no terminal."""
    print("=" * 60)
    print("  INDICADORES DE MERCADO — BTC-USD (tempo real)")
    print("=" * 60)
    for key, value in fast_info.items():
        if isinstance(value, float) and value > 1_000:
            print(f"  {key}: ${value:,.2f}")
        else:
            print(f"  {key}: {value}")
    print()


# ══════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════

def main() -> None:
    try:
        # 1. Carregar histórico existente
        existing = load_history()

        # 2. Atualização incremental
        updated = fetch_history(existing)
        save_history(updated)

        # 3. Informações gerais
        info = fetch_info()
        save_info(info)

        # 4. Indicadores em tempo real
        fast_info = fetch_fast_info()
        display_fast_info(fast_info)

    except Exception as exc:
        log.error("Erro durante a coleta de dados: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
