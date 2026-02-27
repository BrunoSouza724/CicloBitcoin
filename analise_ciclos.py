"""
analise_ciclos.py - Analise de ciclos BTC-USD com deteccao dinamica de picos.

Logica de projecao:
  - O pico de alta ocorre historicamente 12-18 meses APOS o halving.
  - A projecao ancora no PROXIMO halving + offset historico (halving -> pico).
  - A janela de compra e calculada com base em:
      * Preco abaixo da SMA 200 semanas (fundo de ciclo)
      * RSI < 50 (momentum ainda nao revertido)
      * 6-12 meses antes do proximo halving (acumulacao pre-halving)
      * Apos confirmacao de vale

Requisitos:
    pip install scipy numpy plotly
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# -- Forca UTF-8 no stdout (Windows) --
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# -- Tentativa de importar plotly (opcional) --
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -- Configuracao de log --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HISTORY_FILE = Path("btc_usd_history.json")

# -- Halvings historicos e projetados do BTC --
HALVINGS: list[datetime] = [
    datetime(2012, 11, 28),
    datetime(2016, 7, 9),
    datetime(2020, 5, 11),
    datetime(2024, 4, 20),
    datetime(2028, 4, 20),  # projetado (~4 anos apos o anterior)
]

# -- Parametros de ciclo (ajustaveis) --
# Janela de acumulacao pre-halving: comprar entre X e Y meses antes do halving
BUY_WINDOW_START_MONTHS_BEFORE_HALVING = 12
BUY_WINDOW_END_MONTHS_BEFORE_HALVING = 3


# =============================================
#  Estruturas de dados
# =============================================

@dataclass
class CyclePoint:
    kind: str        # "peak" ou "valley"
    date: datetime
    price: float
    index: int


@dataclass
class CycleInterval:
    label: str
    start: CyclePoint
    end: CyclePoint
    days: int


@dataclass
class HalvingCycle:
    """Dados de um ciclo completo ancorado em um halving."""
    halving_date: datetime
    peak_date: datetime | None
    peak_price: float | None
    days_halving_to_peak: int | None


@dataclass
class Projection:
    label: str
    date_min: datetime
    date_center: datetime
    date_max: datetime
    basis: str


@dataclass
class BuySignal:
    active: bool
    score: int                    # 0-4 (quantos criterios atendem)
    reasons: list[str] = field(default_factory=list)
    window_start: datetime | None = None
    window_end: datetime | None = None
    next_halving: datetime | None = None


# =============================================
#  Carregamento e preparacao dos dados
# =============================================

def load_prices() -> tuple[list[datetime], np.ndarray]:
    """Carrega e ordena o historico de precos."""
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo '{HISTORY_FILE}' nao encontrado. "
            "Execute API.py primeiro para baixar o historico."
        )

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    entries = []
    for date_str, values in raw.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        close = float(values["Close"])
        entries.append((dt, close))

    entries.sort(key=lambda x: x[0])
    dates = [e[0] for e in entries]
    closes = np.array([e[1] for e in entries])
    log.info(
        "Historico carregado: %d registros (%s -> %s)",
        len(dates), dates[0].date(), dates[-1].date()
    )
    return dates, closes


# =============================================
#  Indicadores tecnicos
# =============================================

def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calcula o RSI para a serie de precos."""
    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0.0)
    loss = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.convolve(gain, np.ones(period) / period, mode="full")[:len(gain)]
    avg_loss = np.convolve(loss, np.ones(period) / period, mode="full")[:len(loss)]

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.concatenate([[np.nan], rsi])


def compute_sma(closes: np.ndarray, window: int) -> np.ndarray:
    """Calcula a Media Movel Simples."""
    sma = np.full(len(closes), np.nan)
    for i in range(window - 1, len(closes)):
        sma[i] = closes[i - window + 1: i + 1].mean()
    return sma


# =============================================
#  Deteccao de ciclos
# =============================================

def detect_cycles(
    dates: list[datetime],
    closes: np.ndarray,
    min_distance_days: int = 180,
) -> tuple[list[CyclePoint], list[CyclePoint]]:
    """Detecta picos e vales significativos na serie de precos."""
    distance = max(1, min_distance_days)
    prominence = closes.std() * 0.5

    peak_idx, _ = find_peaks(closes, distance=distance, prominence=prominence)
    valley_idx, _ = find_peaks(-closes, distance=distance, prominence=prominence)

    peaks = [CyclePoint("peak", dates[i], closes[i], i) for i in peak_idx]
    valleys = [CyclePoint("valley", dates[i], closes[i], i) for i in valley_idx]

    log.info("Picos detectados: %d  |  Vales detectados: %d", len(peaks), len(valleys))
    return peaks, valleys


def compute_intervals(points: list[CyclePoint]) -> list[CycleInterval]:
    """Calcula os intervalos (em dias) entre pontos consecutivos do mesmo tipo."""
    intervals = []
    for i in range(1, len(points)):
        days = (points[i].date - points[i - 1].date).days
        label = f"{points[i-1].kind.upper()} {i-1} -> {i}"
        intervals.append(CycleInterval(label, points[i - 1], points[i], days))
    return intervals


def next_halving_after(date: datetime) -> datetime:
    """Retorna o proximo halving apos uma data."""
    future = [h for h in HALVINGS if h > date]
    return future[0] if future else HALVINGS[-1]


def prev_halving_before(date: datetime) -> datetime:
    """Retorna o halving imediatamente anterior a uma data."""
    before = [h for h in HALVINGS if h <= date]
    return before[-1] if before else HALVINGS[0]


# =============================================
#  Analise de ciclos ancorados no halving
# =============================================

def analyze_halving_cycles(
    peaks: list[CyclePoint],
) -> list[HalvingCycle]:
    """
    Para cada halving historico, encontra o pico de alta que ocorreu depois
    e calcula o offset em dias (halving -> pico).
    """
    cycles: list[HalvingCycle] = []

    for halving in HALVINGS:
        # Picos que ocorreram apos este halving e antes do proximo
        next_h = next_halving_after(halving)
        picos_apos = [
            p for p in peaks
            if p.date > halving and p.date < next_h + timedelta(days=365)
        ]

        if picos_apos:
            # O pico relevante e o de maior preco apos o halving
            pico = max(picos_apos, key=lambda x: x.price)
            days_offset = (pico.date - halving).days
            cycles.append(HalvingCycle(halving, pico.date, pico.price, days_offset))
        else:
            cycles.append(HalvingCycle(halving, None, None, None))

    return cycles


# =============================================
#  Projecoes ancoradas no halving
# =============================================

def build_halving_projection(
    cycles: list[HalvingCycle],
    today: datetime,
) -> Projection:
    """
    Projeta o proximo pico de alta com base no offset historico
    (halving -> pico) aplicado ao PROXIMO halving.
    """
    # Usa apenas ciclos com dados reais
    completed = [c for c in cycles if c.days_halving_to_peak is not None]
    offsets = [c.days_halving_to_peak for c in completed]

    avg_offset = int(np.mean(offsets))
    std_offset = int(np.std(offsets))

    next_h = next_halving_after(today)
    proj_center = next_h + timedelta(days=avg_offset)
    proj_min = next_h + timedelta(days=avg_offset - std_offset)
    proj_max = next_h + timedelta(days=avg_offset + std_offset)

    basis = (
        f"Offset medio historico halving->pico: {avg_offset} +/- {std_offset} dias "
        f"| Proximo halving: {next_h.date()} "
        f"| Baseado em {len(completed)} ciclos completos"
    )

    return Projection(
        label="Proximo PICO de alta (ancoragem no halving)",
        date_min=proj_min,
        date_center=proj_center,
        date_max=proj_max,
        basis=basis,
    )


# =============================================
#  Sinal de compra
# =============================================

def evaluate_buy_signal(
    today: datetime,
    last_price: float,
    last_rsi: float | None,
    last_sma_200w: float | None,
    valleys: list[CyclePoint],
) -> BuySignal:
    """
    Avalia se o momento atual e uma boa janela de compra para o proximo ciclo.

    Criterios (score 0-4):
      1. Dentro da janela pre-halving (3-12 meses antes do proximo halving)
      2. Preco abaixo ou proximo da SMA 200 semanas (fundo de ciclo)
      3. RSI semanal < 50 (momentum ainda fraco = entrada antecipada)
      4. Vale recente confirmado (ultimo vale a menos de 18 meses)
    """
    next_h = next_halving_after(today)
    days_to_halving = (next_h - today).days
    months_to_halving = days_to_halving / 30.44

    window_start = next_h - timedelta(days=BUY_WINDOW_START_MONTHS_BEFORE_HALVING * 30)
    window_end = next_h - timedelta(days=BUY_WINDOW_END_MONTHS_BEFORE_HALVING * 30)

    score = 0
    reasons: list[str] = []

    # Criterio 1: dentro da janela pre-halving
    if window_start <= today <= window_end:
        score += 1
        reasons.append(
            f"[OK] Dentro da janela pre-halving "
            f"({months_to_halving:.1f} meses antes de {next_h.date()})"
        )
    elif today < window_start:
        reasons.append(
            f"[--] Ainda cedo para a janela pre-halving "
            f"(faltam {int(months_to_halving - BUY_WINDOW_START_MONTHS_BEFORE_HALVING)} "
            f"meses para ela iniciar)"
        )
    else:
        reasons.append(
            f"[--] Janela pre-halving ja encerrada "
            f"({int(months_to_halving)} meses para o halving)"
        )

    # Criterio 2: preco vs SMA 200 semanas
    if last_sma_200w is not None and not np.isnan(last_sma_200w):
        ratio = last_price / last_sma_200w
        if ratio < 1.2:
            score += 1
            reasons.append(
                f"[OK] Preco proximo/abaixo da SMA 200w "
                f"(preco = {ratio:.2f}x a media)"
            )
        else:
            reasons.append(
                f"[--] Preco elevado vs SMA 200w "
                f"(preco = {ratio:.2f}x a media - risco maior de entrada)"
            )

    # Criterio 3: RSI < 50
    if last_rsi is not None and not np.isnan(last_rsi):
        if last_rsi < 50:
            score += 1
            reasons.append(f"[OK] RSI em {last_rsi:.1f} (abaixo de 50 = momentum fraco, zona de acumulacao)")
        elif last_rsi < 60:
            reasons.append(f"[~] RSI em {last_rsi:.1f} (neutro, monitorar)")
        else:
            reasons.append(f"[--] RSI em {last_rsi:.1f} (acima de 60 = entrada mais arriscada)")

    # Criterio 4: vale recente confirmado (< 18 meses)
    if valleys:
        last_valley = valleys[-1]
        days_since_valley = (today - last_valley.date).days
        if days_since_valley <= 548:  # 18 meses
            score += 1
            reasons.append(
                f"[OK] Vale recente confirmado em {last_valley.date.date()} "
                f"(${last_valley.price:,.0f}) - ha {days_since_valley} dias"
            )
        else:
            reasons.append(
                f"[--] Ultimo vale ha {days_since_valley} dias "
                f"({last_valley.date.date()}) - ciclo pode estar avancado"
            )

    active = score >= 2

    return BuySignal(
        active=active,
        score=score,
        reasons=reasons,
        window_start=window_start,
        window_end=window_end,
        next_halving=next_h,
    )


# =============================================
#  Saida textual
# =============================================

def print_report(
    peaks: list[CyclePoint],
    valleys: list[CyclePoint],
    halving_cycles: list[HalvingCycle],
    projection: Projection,
    buy_signal: BuySignal,
    rsi: np.ndarray,
    sma_200w: np.ndarray,
    dates: list[datetime],
    closes: np.ndarray,
) -> None:
    SEP = "=" * 65
    DIV = "-" * 50

    print("\n" + SEP)
    print("  ANALISE DE CICLOS BTC-USD")
    print(SEP)

    # -- Ciclos historicos ancorados no halving --
    print(f"\n  {DIV}")
    print("  CICLOS HISTORICOS (halving -> pico)")
    print(f"  {DIV}")
    print(f"  {'Halving':<14} {'Pico de Alta':<14} {'Preco':<15} {'Offset (dias)'}")
    print(f"  {'-'*12:<14} {'-'*12:<14} {'-'*13:<15} {'-'*13}")
    for c in halving_cycles:
        if c.peak_date:
            print(
                f"  {str(c.halving_date.date()):<14} "
                f"{str(c.peak_date.date()):<14} "
                f"${c.peak_price:>12,.0f}   "
                f"{c.days_halving_to_peak} dias"
            )
        else:
            print(f"  {str(c.halving_date.date()):<14} {'(projetado)':<14} {'---':<15} ---")

    # -- Picos e vales detectados --
    print(f"\n  {DIV}")
    print("  PICOS DETECTADOS")
    print(f"  {DIV}")
    for i, p in enumerate(peaks):
        print(f"  [{i}] {p.date.strftime('%Y-%m-%d')}  |  ${p.price:>14,.2f}")

    print(f"\n  {DIV}")
    print("  VALES DETECTADOS")
    print(f"  {DIV}")
    for i, v in enumerate(valleys):
        print(f"  [{i}] {v.date.strftime('%Y-%m-%d')}  |  ${v.price:>14,.2f}")

    # -- Indicadores tecnicos --
    last_rsi = next((v for v in reversed(rsi) if not np.isnan(v)), None)
    last_sma = next((v for v in reversed(sma_200w) if not np.isnan(v)), None)
    last_price = closes[-1]

    print(f"\n  {DIV}")
    print("  INDICADORES TECNICOS (hoje)")
    print(f"  {DIV}")
    print(f"  Preco atual:    ${last_price:>14,.2f}")
    if last_rsi is not None:
        if last_rsi > 70:
            zone = "SOBRECOMPRADO (!)"
        elif last_rsi < 30:
            zone = "SOBREVENDIDO (v)"
        elif last_rsi < 50:
            zone = "Fraco - zona de acumulacao"
        else:
            zone = "Neutro / Topping"
        print(f"  RSI (14):       {last_rsi:>8.1f}  ->  {zone}")
    if last_sma is not None:
        ratio = last_price / last_sma
        rel = "acima" if last_price > last_sma else "abaixo"
        print(f"  SMA 200 sem:   ${last_sma:>14,.2f}  ->  Preco {rel} ({ratio:.2f}x)")

    # -- Halvings --
    today = dates[-1]
    print(f"\n  {DIV}")
    print("  HALVINGS")
    print(f"  {DIV}")
    for h in HALVINGS:
        delta = (h - today).days
        status = f"em {delta} dias ({delta//30} meses)" if delta > 0 else f"ha {-delta} dias"
        print(f"  {h.date()}  ->  {status}")

    # -- Projecao ancorada no halving --
    print(f"\n  {DIV}")
    print("  PROJECAO DO PROXIMO PICO (ancoragem no halving)")
    print(f"  {DIV}")
    print(f"\n  >> {projection.label}")
    print(f"     Centro:    {projection.date_center.strftime('%Y-%m-%d')}")
    print(f"     Intervalo: {projection.date_min.strftime('%Y-%m-%d')} -> {projection.date_max.strftime('%Y-%m-%d')}")
    print(f"     Base: {projection.basis}")

    # -- Sinal de compra --
    print(f"\n  {DIV}")
    status_str = "*** ATIVO ***" if buy_signal.active else "INATIVO"
    print(f"  SINAL DE COMPRA: {status_str}  (score: {buy_signal.score}/4)")
    print(f"  {DIV}")
    if buy_signal.window_start and buy_signal.window_end:
        print(f"  Janela ideal de acumulacao:")
        print(f"    Inicio: {buy_signal.window_start.strftime('%Y-%m-%d')}  "
              f"({BUY_WINDOW_START_MONTHS_BEFORE_HALVING} meses antes do halving)")
        print(f"    Fim:    {buy_signal.window_end.strftime('%Y-%m-%d')}  "
              f"({BUY_WINDOW_END_MONTHS_BEFORE_HALVING} meses antes do halving)")
        print(f"  Proximo halving: {buy_signal.next_halving.date()}")
    print()
    print("  Criterios avaliados:")
    for r in buy_signal.reasons:
        print(f"    {r}")

    print("\n" + SEP + "\n")


# =============================================
#  Visualizacao
# =============================================

def plot_chart(
    dates: list[datetime],
    closes: np.ndarray,
    peaks: list[CyclePoint],
    valleys: list[CyclePoint],
    projection: Projection,
    buy_signal: BuySignal,
    rsi: np.ndarray,
    sma_200w: np.ndarray,
) -> None:
    """Gera grafico interativo com Plotly e salva em HTML."""
    if not PLOTLY_AVAILABLE:
        log.warning("Plotly nao instalado. Grafico nao gerado. (pip install plotly)")
        return

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        subplot_titles=("BTC-USD - Preco & Ciclos", "RSI (14)"),
        vertical_spacing=0.05,
    )

    # -- Preco --
    fig.add_trace(
        go.Scatter(x=dates, y=closes, name="BTC-USD",
                   line=dict(color="#f7931a", width=1.5)),
        row=1, col=1,
    )

    # -- SMA 200 semanas --
    fig.add_trace(
        go.Scatter(x=dates, y=sma_200w, name="SMA 200w",
                   line=dict(color="#00bcd4", width=1.5, dash="dot")),
        row=1, col=1,
    )

    # -- Picos --
    fig.add_trace(
        go.Scatter(
            x=[p.date for p in peaks],
            y=[p.price for p in peaks],
            mode="markers+text",
            name="Pico",
            marker=dict(color="red", size=12, symbol="triangle-up"),
            text=[f"${p.price/1e3:.0f}K" for p in peaks],
            textposition="top center",
        ),
        row=1, col=1,
    )

    # -- Vales --
    fig.add_trace(
        go.Scatter(
            x=[v.date for v in valleys],
            y=[v.price for v in valleys],
            mode="markers+text",
            name="Vale",
            marker=dict(color="lime", size=12, symbol="triangle-down"),
            text=[f"${v.price/1e3:.1f}K" for v in valleys],
            textposition="bottom center",
        ),
        row=1, col=1,
    )

    # -- Halvings --
    for h in HALVINGS:
        fig.add_vline(
            x=h.timestamp() * 1000,
            line_width=1.5,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Halving {h.year}",
            annotation_position="top right",
            row=1, col=1,
        )

    # -- Janela de compra --
    if buy_signal.window_start and buy_signal.window_end:
        buy_color = "#00e676" if buy_signal.active else "#78909c"
        fig.add_vrect(
            x0=buy_signal.window_start, x1=buy_signal.window_end,
            fillcolor=buy_color, opacity=0.18,
            line_width=1.5, line_color=buy_color,
            annotation_text=f"Janela de Compra (score {buy_signal.score}/4)",
            annotation_position="top left",
            row=1, col=1,
        )

    # -- Projecao do proximo pico --
    fig.add_vrect(
        x0=projection.date_min, x1=projection.date_max,
        fillcolor="#ff6b6b", opacity=0.15,
        line_width=0,
        annotation_text="Proximo Pico Projetado",
        annotation_position="top left",
        row=1, col=1,
    )
    fig.add_vline(
        x=projection.date_center.timestamp() * 1000,
        line_width=2, line_dash="dot", line_color="#ff6b6b",
        row=1, col=1,
    )

    # -- RSI --
    fig.add_trace(
        go.Scatter(x=dates, y=rsi, name="RSI",
                   line=dict(color="#9b59b6", width=1.2)),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="lime", row=2, col=1)

    # -- Layout --
    fig.update_layout(
        title="Analise de Ciclos BTC-USD - Ancoragem no Halving",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        height=850,
    )
    fig.update_yaxes(title_text="Preco (USD)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    output_path = Path("btc_ciclos.html")
    fig.write_html(str(output_path))
    log.info("Grafico salvo em: %s", output_path)


# =============================================
#  Entry point
# =============================================

def main() -> None:
    try:
        dates, closes = load_prices()
        today = dates[-1]
        last_price = closes[-1]

        # Indicadores
        rsi = compute_rsi(closes, period=14)
        sma_200w = compute_sma(closes, window=200 * 7)
        last_rsi = next((v for v in reversed(rsi) if not np.isnan(v)), None)
        last_sma = next((v for v in reversed(sma_200w) if not np.isnan(v)), None)

        # Deteccao de ciclos
        peaks, valleys = detect_cycles(dates, closes)

        # Analise de ciclos ancorados no halving
        halving_cycles = analyze_halving_cycles(peaks)

        # Projecao ancorada no proximo halving
        projection = build_halving_projection(halving_cycles, today)

        # Sinal de compra
        buy_signal = evaluate_buy_signal(today, last_price, last_rsi, last_sma, valleys)

        # Relatorio
        print_report(
            peaks, valleys, halving_cycles, projection, buy_signal,
            rsi, sma_200w, dates, closes,
        )

        # Grafico
        plot_chart(dates, closes, peaks, valleys, projection, buy_signal, rsi, sma_200w)

    except FileNotFoundError as e:
        log.error(str(e))
    except Exception as e:
        log.error("Erro inesperado: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
