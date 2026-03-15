# ==========================================================
# OMEGA Ω — EDGE ENGINE (SPOT) v3.2 PRO ACCOUNTING
# - BTC filter: shock hard-block, chop = penalty / soft-block
# - SL dynamic: higher activation + breathing buffer (pre-TP)
# - SL trailing post-TP: volatility-aware trailing (real)
# - TP dynamic: heuristic + rolling stats from DB (symbol-aware)
# - Alerts: anti-spam (no-trade throttling per symbol/reason)
# - Capital guardrails: daily & global DD limits
# - Accounting: daily PnL in USDT + fee reserve suggestion (BNB gasolina)
# ==========================================================

import os
import time
import sqlite3
import requests
import math
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.enums import *
from ta import add_all_ta_features
from ta.volatility import AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytz
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# CONFIG — PRODUCCIÓN (CREDENCIALES POR ENV VARS)
# ==========================================================
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise RuntimeError("Faltan BINANCE_API_KEY / BINANCE_API_SECRET en variables de entorno.")

TZ_COLOMBIA = pytz.timezone('America/Bogota')

SYMBOLS_SURVIVAL = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SYMBOLS_GROWTH = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]
SYMBOLS_EXPANSION = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "FETUSDT", "INJUSDT"]

FEE_SINGLE = 0.00075          # 0.075% usando BNB
FEE_ROUNDTRIP = 0.0015        # ida y vuelta
# Gestión de “gasolina” (fees)
FEE_RESERVE_DAYS = 5          # objetivo: tener fees para ~5 días de trading
FEE_PROFIT_ALLOCATION = 0.35  # proporción de la ganancia diaria sugerida para recargar BNB

WARMUP_MINUTES = 5
SYMBOL_COOLDOWN_SEC = 300
MIN_CAPITAL_PER_TRADE = 15
MIN_BNB_FOR_FEES = 0.002

DAILY_REPORT_HOUR = 23
DAILY_REPORT_MINUTE = 59
ALERT_FREQUENCY_SEC = 60

BASE_DIR = "/home/ubuntu/bot"
DATA_DIR = f"{BASE_DIR}/data"
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = f"{DATA_DIR}/omega_real.db"
CSV_PATH = f"{DATA_DIR}/omega_trades.csv"

# ===========================
# WINNER-GRADE TUNING
# ===========================
BTC_SHOCK_BLOCK = True

# Shock thresholds (duro)
BTC_SHOCK_LAST_CHANGE_PCT = 0.65
BTC_SHOCK_LAST_RANGE_PCT = 1.00
BTC_SHOCK_DRIFT_5M_PCT = 0.95

# Chop scoring (suave): mayor score = peor (más chop)
BTC_CHOP_VOL_MIN = 1.60
BTC_CHOP_DRIFT_ABS_MAX = 0.12

# Soft-block only when chop score is high and edge not strong
BTC_CHOP_SOFTBLOCK_SCORE = 1.35   # >= bloquea LOW/MED
BTC_CHOP_EDGE_OVERRIDE = 8.2      # >= esto se permite aunque haya chop (edge manda)

# SL dynamics - "dejar respirar"
PROTECT_ACTIVATION_NET = 0.75      # antes ~0.40 -> ahora 0.75
PROTECT_BUFFER_BASE = 0.85         # buffer general (más amplio)
PROTECT_BUFFER_TIGHT = 0.70        # BTC/ETH
PROTECT_BUFFER_WIDE = 1.05         # SOL / alta vol
PROTECT_MAX_LOCK = 0.35            # nunca dejar SL pegado al precio (evita micro-stop)

# Trailing post-TP: sensibilidad por volatilidad
POST_TP_TRAIL_MIN = 0.85
POST_TP_TRAIL_MAX = 1.35

# No-trade spam control
NO_TRADE_MIN_INTERVAL_SEC = 3 * 60 * 60  # 3 horas


# TP stats
TP_STATS_MIN_EXITS = 18            # si no hay suficientes exits, usa heurística
TP_STATS_LOOKBACK = 160            # ventana de exits para stats
TP_STATS_CACHE_TTL = 10 * 60

import sys
import threading
import logging

LOG_DIR = f"{DATA_DIR}/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_global_error_logger():
    logger = logging.getLogger("OMEGA_GLOBAL_ERRORS")
    logger.setLevel(logging.ERROR)
    if logger.handlers:
        return logger
    today = datetime.now(TZ_COLOMBIA).strftime("%Y-%m-%d")
    log_file = f"{LOG_DIR}/errors_{today}.log"
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(fmt="%(asctime)s [ERROR] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

ERROR_LOGGER = setup_global_error_logger()

def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    ERROR_LOGGER.error("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler

def thread_exception_handler(args):
    ERROR_LOGGER.error(f"THREAD EXCEPTION: {args.thread.name}", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

threading.excepthook = thread_exception_handler

import csv

CSV_HEADERS = [
    "timestamp_colombia", "event_type", "symbol", "side", "price", "qty",
    "usdt_value", "gross_pnl_pct", "net_pnl_pct", "fees_pct", "hold_minutes",
    "tp_pct", "sl_pct", "tp_extensions", "strategies", "strategy_score",
    "ml_prediction", "market_regime", "mode", "capital_usdt", "bnb_balance",
    "order_id", "reason", "exit_subtype", "trade_result", "sl_at_exit_pct",
    "edge_score", "edge_bucket", "pos_fraction", "risk_multiplier",
    "no_trade_reason"
]

def write_csv_event(row: dict):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        for k in CSV_HEADERS:
            row.setdefault(k, "")
        writer.writerow(row)

STRATEGY_WEIGHTS = {
    "Momentum Breakout": 2.8,
    "Volume Spike Confirmed": 2.2,
    "MACD Bullish": 2.0,
    "Volatility Expansion Confirmed": 1.9,
    "EMA Crossover": 1.6,
    "Multi-Timeframe Confirmation": 2.5,
    "RSI Recovery": 1.4,
    "Early Momentum": 0.9,
    "Overbought Reversal Caution": -1.5,
    "Trend Follower (ADX)": 2.0
}

SYMBOL_PROFILES = {
    "BTCUSDT": {"trailing_buffer": 0.85, "max_extensions": 4, "volatility_class": "LOW"},
    "ETHUSDT": {"trailing_buffer": 0.90, "max_extensions": 4, "volatility_class": "LOW"},
    "SOLUSDT": {"trailing_buffer": 1.10, "max_extensions": 3, "volatility_class": "MEDIUM"},
    "BNBUSDT": {"trailing_buffer": 0.95, "max_extensions": 3, "volatility_class": "LOW"},
    "XRPUSDT": {"trailing_buffer": 1.05, "max_extensions": 3, "volatility_class": "MEDIUM"},
    "AVAXUSDT": {"trailing_buffer": 1.10, "max_extensions": 3, "volatility_class": "MEDIUM"},
    "LINKUSDT": {"trailing_buffer": 1.05, "max_extensions": 3, "volatility_class": "MEDIUM"},
    "FETUSDT": {"trailing_buffer": 1.25, "max_extensions": 2, "volatility_class": "HIGH"},
    "INJUSDT": {"trailing_buffer": 1.25, "max_extensions": 2, "volatility_class": "HIGH"}
}

# ==========================================================
# TELEGRAM REPORTER (anti-spam)
# ==========================================================
# ==========================================================
# TELEGRAM REPORTER (SALA DE CONTROL)
# ==========================================================
class Reporter:

    last_alert_time = 0
    last_bnb_alert_time = 0
    _no_trade_last = {}

    @staticmethod
    def send(msg, is_critical=False):
        if not TG_TOKEN or not TG_CHAT_ID: return
        now = time.time()
        if not is_critical and (now - Reporter.last_alert_time < ALERT_FREQUENCY_SEC): return
        try:
            requests.post(
                f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=8
            )
            Reporter.last_alert_time = now
        except: pass

    @staticmethod
    def online(mode, symbols_count, capital, bnb, risk_mult=1.0):
        msg = (
            f"🟢 <b>SISTEMA OMEGA INICIADO (v4.0)</b>\n\n"
            f"💰 <b>Capital Operativo:</b> ${capital:.2f} USDT\n"
            f"⛽ <b>Reserva BNB:</b> {bnb:.6f}\n"
            f"⚙️ <b>Modo Actual:</b> {mode}\n"
            f"📊 <b>Pares Activos:</b> {symbols_count}\n"
            f"🛡️ <b>Risk Multiplier:</b> x{risk_mult:.2f}\n\n"
            f"<i>Escaneo de mercado en curso...</i>"
        )
        Reporter.send(msg, is_critical=True)

    @staticmethod
    def entry(symbol, entry, strategy_score, strategies, usdt, tp, sl, order_id, ml_pred, regime, edge_score, edge_bucket, pos_fraction, risk_mult):
        sl_price = entry * (1 + sl/100)
        tp_price = entry * (1 + tp/100)
        breakeven_price = entry * (1 + FEE_ROUNDTRIP)

        # Cálculos predictivos de PnL en USDT
        qty = usdt / entry
        potential_win_usdt = (tp_price - entry) * qty - (FEE_ROUNDTRIP * usdt)
        potential_loss_usdt = (entry - sl_price) * qty + (FEE_ROUNDTRIP * usdt)

        strat_list = "\n".join([f" ├ {s}" for s in strategies[:4]])

        msg = (
            f"🚀 <b>NUEVA ENTRADA | {symbol}</b>\n\n"
            f"💵 <b>Capital Invertido:</b> ${usdt:.2f} (Size: {pos_fraction*100:.0f}%)\n"
            f"📌 <b>Precio Entrada:</b> {entry:.6g}\n\n"
            f"🎯 <b>TAKE PROFIT (TP):</b>\n"
            f" ├ Precio: <b>{tp_price:.6g}</b> (+{tp:.2f}%)\n"
            f" └ Ganancia Est.: <b>+${potential_win_usdt:.2f}</b>\n\n"
            f"🛑 <b>STOP LOSS (SL):</b>\n"
            f" ├ Precio: <b>{sl_price:.6g}</b> ({sl:.2f}%)\n"
            f" └ Pérdida Est.: <b>-${potential_loss_usdt:.2f}</b>\n\n"
            f"⚖️ <b>Breakeven (Cero Pérdida):</b> {breakeven_price:.6g}\n\n"
            f"🧠 <b>ANÁLISIS DEL MOTOR:</b>\n"
            f" ├ Régimen: {regime}\n"
            f" ├ Edge Score: {edge_score:.1f}/10 ({edge_bucket})\n"
            f" └ Risk Multiplier: x{risk_mult:.2f}\n"
            f"⚙️ <b>Estrategias:</b>\n{strat_list}\n\n"
            f"🆔 <code>{order_id}</code>"
        )
        Reporter.send(msg, is_critical=True)

    @staticmethod
    def protection_activated(symbol, entry, current, sl_pct, sl_price, margin):
        msg = (
            f"🛡️ <b>PISO DE CRISTAL ACTIVADO | {symbol}</b>\n\n"
            f"📈 <b>Precio Actual:</b> {current:.6g}\n"
            f"🔒 <b>Nuevo Stop Loss:</b> {sl_price:.6g}\n"
            f"💰 <b>Beneficio Asegurado:</b> {sl_pct:+.2f}%\n"
            f"📏 <b>Margen de Respiro:</b> {margin:.2f}%\n\n"
            f"<i>El capital está blindado. Este trade cerrará en verde.</i>"
        )
        Reporter.send(msg, is_critical=True)

    @staticmethod
    def tp_extension(symbol, net, old_tp, new_tp, new_sl, extension_num, max_ext):
        msg = (
            f"🔥 <b>TENDENCIA EXPRIMIDA | {symbol}</b>\n\n"
            f"🚀 <b>Rendimiento Actual:</b> +{net:.2f}%\n"
            f"🎯 <b>TP Extendido a:</b> +{new_tp:.2f}%\n"
            f"🔒 <b>SL Actualizado a:</b> +{new_sl:.2f}%\n"
            f"🔄 <b>Extensión:</b> {extension_num} de {max_ext}\n\n"
            f"<i>Dejando correr las ganancias...</i>"
        )
        Reporter.send(msg)

    @staticmethod
    def exit(symbol, net, gross, pnl_usdt, reason, hold, tp_reached, order_id, risk_mult, streak_w, streak_l):
        is_win = net > 0
        emoji = "✅" if is_win else "❌"
        header = "GANANCIA" if is_win else "PÉRDIDA"

        msg = (
            f"{emoji} <b>TRADE CERRADO: {header} | {symbol}</b>\n\n"
            f"💵 <b>PnL Neto (USDT):</b> {pnl_usdt:+.2f} USDT\n"
            f"📊 <b>Rentabilidad:</b> {net:+.2f}%\n"
            f"⏱️ <b>Duración:</b> {hold:.1f} min\n"
            f"📋 <b>Motivo:</b> {reason}\n\n"
            f"📈 <b>ESTADO DEL MOTOR:</b>\n"
            f" ├ Racha Actual: W{streak_w} / L{streak_l}\n"
            f" └ Nuevo Risk Mult: x{risk_mult:.2f}\n"
        )
        Reporter.send(msg, is_critical=True)

    @staticmethod
    def daily(report):
        pnl_usdt = report.get("pnl_usdt", 0.0)
        is_positive = report['pnl'] >= 0
        emoji = "🎉" if is_positive else "📉"

        msg = (
            f"📊 <b>CIERRE DIARIO OMEGA Ω | {report['date']}</b>\n\n"
            f"💰 <b>RESUMEN DE CAPITAL:</b>\n"
            f" ├ Inicio: ${report['cap_start']:.2f}\n"
            f" ├ Final: <b>${report['cap_end']:.2f}</b>\n"
            f" └ <b>PnL del Día: {report['pnl']:+.2f}% ({pnl_usdt:+.2f} USDT)</b> {emoji}\n\n"
            f"📈 <b>RENDIMIENTO OPERATIVO:</b>\n"
            f" ├ Total Trades: {report['trades']}\n"
            f" ├ Aciertos / Fallos: ✅ {report['wins']} | ❌ {report['losses']}\n"
            f" └ Win Rate: {report['win_rate']:.1f}%\n\n"
            f"⛽ <b>RESERVA DE GASOLINA (BNB):</b>\n"
            f" ├ BNB Disponible: {report['bnb']:.6f}\n"
            f" └ Días de Reserva: ~{report.get('fee_reserve_days', 5)}\n\n"
            f"⚙️ <b>ESTADO DEL SISTEMA:</b>\n"
            f" ├ Modo Actual: {report['mode']}\n"
            f" ├ Risk Multiplier: x{report['risk_mult']:.2f}\n"
            f" └ Cerebro ML: {report['ml_status']}\n"
        )
        Reporter.send(msg, is_critical=True)

    @staticmethod
    def mode_change(old_mode, new_mode, capital, symbols_count):
            msg = (
                f"🔄 <b>CAMBIO DE MODO OPERATIVO</b>\n\n"
                f"📈 <b>Capital Actual:</b> ${capital:.2f}\n"
                f"⚙️ <b>Transición:</b> {old_mode} ➡️ {new_mode}\n"
                f"📊 <b>Pares Habilitados:</b> {symbols_count}\n\n"
                f"<i>El motor ha reajustado los límites de exposición al riesgo.</i>"
            )
            Reporter.send(msg, is_critical=True)

    @staticmethod
    def emergency_stop(pnl, limit, scope="daily"):
        scope_txt = "EQUITY GLOBAL" if scope == "equity" else "DÍA ACTUAL"
        msg = (
            f"🚨 <b>DETENCIÓN DE EMERGENCIA</b> 🚨\n\n"
            f"Ámbito: {scope_txt}\n"
            f"Caída Registrada (Drawdown): {pnl:.2f}%\n"
            f"Límite Máximo Permitido: {limit:.2f}%\n\n"
            f"⚠️ <i>Entradas congeladas. Capital protegido por protocolo de seguridad.</i>"
        )
        Reporter.send(msg, is_critical=True)

    @staticmethod
    def low_bnb(bnb_balance, estimated_trades_left):
        msg = (
            f"⛽ <b>ALERTA: GASOLINA BAJA</b>\n\n"
            f"BNB disponible: {bnb_balance:.6f}\n"
            f"Trades restantes aprox: {estimated_trades_left}\n\n"
            f"<i>Recomendación: Recargar BNB pronto para evitar pagar comisiones completas en USDT.</i>"
        )
        Reporter.send(msg, is_critical=True)

    @staticmethod
    def ml_phase_change(phase, precision, trades_analyzed):
        msg = (
            f"🧠 <b>EVOLUCIÓN DEL CEREBRO ML</b>\n\n"
            f"Fase Alcanzada: <b>{phase}</b>\n"
            f"Trades Analizados: {trades_analyzed}\n"
            f"Precisión Predictiva: {precision:.1f}%\n\n"
            f"<i>El modelo HistGradientBoosting se ha calibrado con la nueva data.</i>"
        )
        Reporter.send(msg)

    @staticmethod
    def warning(msg):
        Reporter.send(f"⚠️ <b>ALERTA DEL SISTEMA:</b>\n\n{msg}", is_critical=True)

    @staticmethod
    def no_trade(symbol, reason, edge_score=None):
        critical = ["shock", "Daily DD", "emergency", "SURVIVAL"]
        if not any(k.lower() in reason.lower() for k in critical): return
        
        now = time.time()
        last = Reporter._no_trade_last.get(symbol, {"t": 0})
        if now - last["t"] < NO_TRADE_MIN_INTERVAL_SEC: return
        Reporter._no_trade_last[symbol] = {"t": now}

        msg = (
            f"⛔ <b>ENTRADA RECHAZADA (SEGURIDAD) | {symbol}</b>\n\n"
            f"Razón: {reason}\n"
            f"Edge Score abortado: {edge_score:.1f}/10" if edge_score else f"Razón: {reason}"
        )
        Reporter.send(msg, is_critical=True)



# ==========================================================
# OMEGA Ω — EDGE ENGINE v3.2
# ==========================================================
class OmegaEvolutionary:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

        self.positions = {}
        self.start_time = time.time()
        self.last_daily = None
        self.trade_history = []
        self.ml_models = {}
        self.ml_phase = "OBSERVATION"
        self.mode = "SURVIVAL"
        self.symbols = SYMBOLS_SURVIVAL
        self.symbol_ready = {s: False for s in SYMBOLS_EXPANSION}
        self.cooldown_until = {s: 0 for s in SYMBOLS_EXPANSION}

        self.daily_start_capital = 0
        self.trades_today = 0
        self.daily_loss_limit = -3.0
        self.emergency_stop = False

        self.risk_multiplier = 1.0
        self.win_streak = 0
        self.loss_streak = 0
        self.daily_entry_freeze = False
        self.last_dust_check = 0

        # Guardrail global de equity
        self.global_peak_capital = 0.0
        self.global_dd_limit = -12.0  # se ajusta según modo

        self._capital_cache = {"value": 0, "timestamp": 0}
        self._capital_cache_ttl = 25

        # TP stats cache
        self._tp_stats_cache = {"t": 0, "data": {}}

        self._init_db()
        self._load_historical_data()
        self._train_ml()
        self._recover_trades_today()
        self._initialize_daily_capital()
        self._init_global_peak_capital()
        self._determine_mode()
        self._recover_open_positions()
        self.initial_bnb = self._get_bnb_balance()

        Reporter.online(self.mode, len(self.symbols), self.daily_start_capital, self.initial_bnb, self.risk_multiplier)



    def _get_usdt_total(self):
        """
        Retorna SOLO USDT libre + USDT bloqueado.
        No incluye BNB ni otros assets.
        """
        try:
            bal = self.client.get_asset_balance(asset="USDT")
            free = float(bal.get("free", 0))
            locked = float(bal.get("locked", 0))
            return free + locked
        except:
            return 0.0


    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    ts TEXT, symbol TEXT, action TEXT, price REAL, qty REAL,
                    usdt REAL, net REAL, gross REAL, hold REAL, order_id TEXT,
                    strategies TEXT, score REAL, tp REAL, sl REAL, reason TEXT,
                    momentum REAL, volatility REAL, rsi REAL, macd REAL,
                    ml_prediction REAL, regime TEXT, tp_extensions INTEGER,
                    edge_score REAL, edge_bucket TEXT, pos_fraction REAL,
                    risk_multiplier REAL, no_trade_reason TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_snapshots (
                    date TEXT PRIMARY KEY, capital_start REAL, capital_end REAL,
                    bnb_start REAL, bnb_end REAL, mode TEXT,
                    risk_multiplier REAL, win_streak INTEGER, loss_streak INTEGER
                )
            """)

    def _load_historical_data(self):
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT symbol, net, gross, hold, strategies, momentum, volatility, rsi, macd "
                "FROM trades WHERE action = 'exit' ORDER BY ts DESC LIMIT 700"
            ).fetchall()

        self.trade_history = []
        for r in rows:
            self.trade_history.append({
                'symbol': r[0],
                'net': float(r[1]) if r[1] is not None else 0.0,
                'gross': float(r[2]) if r[2] is not None else 0.0,
                'hold': float(r[3]) if r[3] is not None else 0.0,
                'strategies': (r[4].split(',') if r[4] else []),
                'momentum': float(r[5]) if r[5] is not None else 0.0,
                'volatility': float(r[6]) if r[6] is not None else 0.0,
                'rsi': float(r[7]) if r[7] is not None else 50.0,
                'macd': float(r[8]) if r[8] is not None else 0.0
            })

    def _recover_trades_today(self):
        today = datetime.now(TZ_COLOMBIA).strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(DB_PATH) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE ts LIKE ? AND action = 'entry'",
                    (f"{today}%",)
                ).fetchone()[0]
            self.trades_today = int(count or 0)
            if self.trades_today > 0:
                Reporter.warning(f"Trades recuperados del día: {self.trades_today}")
        except Exception as e:
            ERROR_LOGGER.error(f"Error recuperando trades del día: {str(e)}")
            self.trades_today = 0

    def _initialize_daily_capital(self):
        today = datetime.now(TZ_COLOMBIA).strftime("%Y-%m-%d")
        cap_real = self._get_usdt_total()
        bnb_real = self._get_bnb_balance()

        try:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    "SELECT capital_start, risk_multiplier, win_streak, loss_streak "
                    "FROM daily_snapshots WHERE date = ?",
                    (today,)
                ).fetchone()

                if row:
                    self.daily_start_capital = float(row[0] or cap_real)
                    self.risk_multiplier = float(row[1] or 1.0)
                    self.win_streak = int(row[2] or 0)
                    self.loss_streak = int(row[3] or 0)
                else:
                    self.daily_start_capital = cap_real
                    conn.execute("""
                        INSERT INTO daily_snapshots
                        (date, capital_start, capital_end, bnb_start, bnb_end, mode, risk_multiplier, win_streak, loss_streak)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (today, cap_real, cap_real, bnb_real, bnb_real,
                        self.mode, self.risk_multiplier,
                        self.win_streak, self.loss_streak))

        except Exception as e:
            ERROR_LOGGER.error(f"Error inicializando snapshot diario: {str(e)}")
            self.daily_start_capital = cap_real

        if self.daily_start_capital < 10:
            self.daily_start_capital = cap_real

        # 🔥 AQUÍ es donde debe ir
        self.global_peak_capital = self.daily_start_capital


    def _init_global_peak_capital(self):
        """
        Inicializa el peak global SOLO con USDT real.
        Evita inconsistencias históricas de snapshots antiguos.
        """
        current_usdt = self._get_capital_robust()

        if current_usdt <= 0:
            current_usdt = self.daily_start_capital or 0.0

        self.global_peak_capital = float(current_usdt)


    # ---------------- API helpers ----------------
    def _get_usdt_free(self):
        try:
            return float(self.client.get_asset_balance(asset="USDT")["free"])
        except:
            return 0.0

    def _get_bnb_balance(self):
        try:
            return float(self.client.get_asset_balance(asset="BNB")["free"])
        except:
            return 0.0

    def _get_capital_robust(self):
        try:
            account = self.client.get_account()
            total_usdt = 0.0

            for balance in account["balances"]:
                asset = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                qty = free + locked

                if qty <= 0:
                    continue

                if asset == "USDT":
                    total_usdt += qty
                else:
                    symbol = asset + "USDT"
                    try:
                        price = float(self.client.get_symbol_ticker(symbol=symbol)["price"])
                        total_usdt += qty * price
                    except:
                        # si no hay par contra USDT, ignoramos
                        pass

            return total_usdt

        except Exception as e:
            ERROR_LOGGER.error(f"Error calculando capital robusto: {str(e)}")
            return self.daily_start_capital or 0.0


    def _get_capital(self):
        now = time.time()
        if now - self._capital_cache["timestamp"] < self._capital_cache_ttl:
            return self._capital_cache["value"]
        cap = self._get_capital_robust()
        self._capital_cache = {"value": cap, "timestamp": now}
        return cap

    # ---------------- Mode ----------------
    def _determine_mode(self):
        cap = self._get_usdt_total()
        old_mode = self.mode
        if cap < 100:
            self.mode = "SURVIVAL"
            self.symbols = SYMBOLS_SURVIVAL
            self.daily_loss_limit = -3.0
            self.global_dd_limit = -12.0
        elif cap < 500:
            self.mode = "GROWTH"
            self.symbols = SYMBOLS_GROWTH
            self.daily_loss_limit = -5.0
            self.global_dd_limit = -15.0
        else:
            self.mode = "EXPANSION"
            self.symbols = SYMBOLS_EXPANSION
            self.daily_loss_limit = -7.0
            self.global_dd_limit = -20.0

        if old_mode != self.mode:
            Reporter.mode_change(old_mode, self.mode, cap, len(self.symbols))

    def _get_mode_params(self):
        if self.mode == "SURVIVAL":
            return {'max_positions': 1, 'min_strategies': 2, 'min_score': 5.0,
                    'sl_base': -1.05, 'ml_threshold': 0.80, 'max_trades_day': 10}
        elif self.mode == "GROWTH":
            return {'max_positions': 2, 'min_strategies': 2, 'min_score': 4.8,
                    'sl_base': -1.00, 'ml_threshold': 0.75, 'max_trades_day': 15}
        else:
            return {'max_positions': self._dynamic_max_positions(), 'min_strategies': 2, 'min_score': 4.5,
                    'sl_base': -0.90, 'ml_threshold': 0.60, 'max_trades_day': 22}

    def _calculate_recent_win_rate(self):
        recent = self.trade_history[-40:]
        if len(recent) < 12:
            return 0.50
        return sum(1 for t in recent if t['net'] > 0) / len(recent)

    def _dynamic_max_positions(self):
        wr = self._calculate_recent_win_rate()
        if wr > 0.75:
            return 4
        elif wr > 0.65:
            return 3
        return 2

    # ---------------- Recovery positions ----------------
    def _recover_open_positions(self):
        try:
            account = self.client.get_account()
            for b in account["balances"]:
                asset = b["asset"]
                free = float(b["free"])
                if asset in ["USDT", "BNB"] or free <= 0:
                    continue
                sym = f"{asset}USDT"
                if sym not in SYMBOLS_EXPANSION:
                    continue
                try:
                    p = float(self.client.get_symbol_ticker(symbol=sym)["price"])
                except:
                    continue
                value = free * p
                if value < 3.0:
                    continue

                orders = self.client.get_all_orders(symbol=sym, limit=20)
                buys = [o for o in orders if o.get("side") == "BUY" and o.get("status") == "FILLED"]
                if not buys:
                    continue
                last_buy = buys[-1]
                qty_bought = float(last_buy.get("executedQty", 0))
                cost = float(last_buy.get("cummulativeQuoteQty", 0))
                avg_price = (cost / qty_bought) if qty_bought > 0 else p

                profile = SYMBOL_PROFILES.get(sym, {"trailing_buffer": 1.0, "max_extensions": 3, "volatility_class": "MEDIUM"})

                self.positions[sym] = {
                    "qty": min(qty_bought, free) if qty_bought > 0 else free,
                    "price": avg_price,
                    "ts": time.time(),
                    "tp": 1.8,
                    "sl": -1.2,
                    "order_id": last_buy.get("orderId"),
                    "strategies": ["Recovered"],
                    "ml_prediction": 0,
                    "tp_extensions": 0,
                    "max_price": avg_price,
                    "trailing_buffer": profile["trailing_buffer"],
                    "max_extensions": profile["max_extensions"],
                    "tp_original": 1.8,
                    "protection_alerted": False,
                    "edge_score": 5.0,
                    "edge_bucket": "MEDIUM",
                    "pos_fraction": 0.4,
                    # por compatibilidad
                    "momentum": 0.0,
                    "volatility": 0.0,
                    "rsi": 50.0,
                    "macd": 0.0,
                    "regime": "",
                    "strategy_score": 0.0
                }
                Reporter.warning(f"✅ Recuperado: {sym} | {self.positions[sym]['qty']:.8f} @ {avg_price:.8g}")
        except Exception as e:
            ERROR_LOGGER.error(f"Error recuperación posiciones: {str(e)}")

    # ---------------- Binance constraints ----------------
    def adjust_qty_to_step(self, symbol, qty):
        try:
            if qty is None or qty <= 0:
                return "0"
            info = self.client.get_symbol_info(symbol)
            for f in info["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    step = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                    precision = int(round(-math.log(step, 10), 0))
                    qty = math.floor(qty / step) * step
                    if qty < min_qty:
                        return "0"
                    qty_str = f"{qty:.{precision}f}".rstrip("0").rstrip(".")
                    return qty_str if qty_str not in ["", "0", "0.0"] else "0"
            return "0"
        except Exception as e:
            ERROR_LOGGER.error(f"Error ajustando qty {symbol}: {str(e)}")
            return "0"

    def _min_notional_ok(self, symbol, usdt):
        try:
            info = self.client.get_symbol_info(symbol)
            for f in info["filters"]:
                if f["filterType"] == "MIN_NOTIONAL":
                    return usdt >= float(f["minNotional"])
            return True
        except:
            return False

    def _market_buy(self, symbol, usdt):
        for attempt in range(3):
            try:
                return self.client.create_order(
                    symbol=symbol, side=SIDE_BUY, type=ORDER_TYPE_MARKET,
                    quoteOrderQty=round(usdt, 2)
                )
            except Exception as e:
                ERROR_LOGGER.error(f"Intento {attempt+1} compra {symbol}: {str(e)}")
                time.sleep(1.5)
        raise Exception(f"Fallo compra {symbol}")

    def _market_sell_qty(self, symbol, qty):
        for attempt in range(3):
            try:
                return self.client.create_order(
                    symbol=symbol, side=SIDE_SELL, type=ORDER_TYPE_MARKET,
                    quantity=str(qty)
                )
            except Exception as e:
                ERROR_LOGGER.error(f"Intento {attempt+1} venta {symbol}: {str(e)}")
                time.sleep(1.5)
        raise Exception(f"Fallo venta {symbol}")

    # ==========================================================
    # METRICS + STRATEGIES
    # ==========================================================
    def _metrics(self, symbol):
        try:
            k = self.client.get_klines(symbol=symbol, interval="1m", limit=220)[:-1]
            df = pd.DataFrame(k, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'taker_base',
                'taker_quote', 'ignore'
            ])
            df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
            df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)

            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            opens = df['open'].values
            vols = df['volume'].values
            returns = np.diff(closes) / closes[:-1]

            momentum = (closes[-1] / closes[0] - 1) * 100
            volatility = np.std(returns) * 100 * np.sqrt(1440)

            last_open = float(opens[-1])
            last_close = float(closes[-1])
            last_high = float(highs[-1])
            last_low = float(lows[-1])
            candle_change = (last_close / last_open - 1) * 100
            candle_range = (last_high / last_low - 1) * 100

            vol_spike = vols[-1] > vols.mean() * 1.5

            gains = np.maximum(returns, 0)
            losses = np.abs(np.minimum(returns, 0))
            rs = gains.mean() / (losses.mean() + 1e-6)
            rsi = 100 - (100 / (1 + rs))

            macd = float(df['trend_macd'].iloc[-1])
            ema_cross = bool(df['trend_ema_fast'].iloc[-1] > df['trend_ema_slow'].iloc[-1])
            adx = float(df['trend_adx'].iloc[-1])

            # Nuevas métricas dinámicas de volatilidad
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100

            k5m = self.client.get_klines(symbol=symbol, interval="5m", limit=20)
            c5 = [float(x[4]) for x in k5m]
            mom_5m = (c5[-1] / c5[0] - 1) * 100

            k15m = self.client.get_klines(symbol=symbol, interval="15m", limit=12)
            c15 = [float(x[4]) for x in k15m]
            mom_15m = (c15[-1] / c15[0] - 1) * 100

            multi_tf = momentum > 0 and mom_5m > 0 and mom_15m > 0

            drift_30 = (closes[-1] / closes[-31] - 1) * 100 if len(closes) >= 31 else momentum
            vol_30 = np.std(returns[-30:]) * 100 * np.sqrt(1440) if len(returns) >= 30 else volatility
            chop = (abs(drift_30) < 0.20) and (vol_30 > 1.80)

            return {
                            'momentum': float(momentum),
                            'volatility': float(volatility),
                            'vol_spike': bool(vol_spike),
                            'rsi': float(rsi),
                            'macd': float(macd),
                            'ema_cross': bool(ema_cross),
                            'multi_tf': bool(multi_tf),
                            'closes': closes,
                            'candle_change': float(candle_change),
                            'candle_range': float(candle_range),
                            'drift_30': float(drift_30),
                            'vol_30': float(vol_30),
                            'chop': bool(chop),
                            'atr_pct': float(atr_pct),
                            'adx': float(adx)
                        }
        except Exception as e:
            ERROR_LOGGER.error(f"Error métricas {symbol}: {str(e)}")
            return None

    def _strategies(self, metrics):
        if not metrics:
            return []
        s = []
        m = metrics['momentum']
        v = metrics['volatility']
        vs = metrics['vol_spike']
        rsi = metrics['rsi']
        macd = metrics['macd']
        ema_cross = metrics['ema_cross']
        multi_tf = metrics['multi_tf']
        adx = metrics.get('adx', 0.0)
        if m > 1.0:
            s.append("Momentum Breakout")
        if 0.4 < m <= 1.0:
            s.append("Early Momentum")
        if v > 1.5 and ema_cross and macd > 0:
            s.append("Volatility Expansion Confirmed")
        if rsi < 30 and m > 0:
            s.append("RSI Recovery")
        if vs and m > 0:
            s.append("Volume Spike Confirmed")
        if macd > 0:
            s.append("MACD Bullish")
        if ema_cross:
            s.append("EMA Crossover")
        if multi_tf:
            s.append("Multi-Timeframe Confirmation")
        if rsi > 80 and m < -0.7:
            s.append("Overbought Reversal Caution")
        if adx > 25 and m > 0:
            s.append("Trend Follower (ADX)")
        return list(set(s))

    def _calculate_strategy_score(self, strategies):
        score = 0.0
        for s in strategies:
            score += STRATEGY_WEIGHTS.get(s, 0.0)
        return float(max(0.0, score))

    # ==========================================================
    # ML
    # ==========================================================
    def _train_ml(self):
            if len(self.trade_history) < 80: # Reducido de 120 a 80 para aprender más rápido
                return

            X, y_class, y_reg = [], [], []
            for t in self.trade_history:
                X.append([t['momentum'], t['volatility'], t['rsi'], t['macd']])
                y_class.append(1 if t['net'] > 0 else 0)
                y_reg.append(t['net'])

            X = np.array(X)
            y_class = np.array(y_class)
            y_reg = np.array(y_reg)

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.3, random_state=42)

                # Modelo superior para no-linealidad en datos cripto
                rf_class = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=5, random_state=42)
                rf_class.fit(X_train, y_train)
                precision = accuracy_score(y_test, rf_class.predict(X_test)) * 100

                rf_reg = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.05, max_depth=5, random_state=42)
                rf_reg.fit(X, y_reg)

                self.ml_models = {'classifier': rf_class, 'regressor': rf_reg, 'precision': float(precision)}

                old_phase = self.ml_phase
                # Umbrales ajustados para el nuevo modelo
                if precision > 65:
                    self.ml_phase = "EXPERT"
                elif precision > 55:
                    self.ml_phase = "TRAINED"
                else:
                    self.ml_phase = "OBSERVATION"

                if old_phase != self.ml_phase:
                    Reporter.ml_phase_change(self.ml_phase, precision, len(self.trade_history))

            except Exception as e:
                ERROR_LOGGER.error(f"Error entrenando ML: {str(e)}")

    def _predict_ml(self, metrics):
            if not self.ml_models or self.ml_phase == "OBSERVATION":
                return 0.0
            try:
                X = np.array([[metrics['momentum'], metrics['volatility'], metrics['rsi'], metrics['macd']]])
                return float(self.ml_models['regressor'].predict(X)[0])
            except:
                return 0.0

    # ==========================================================
    # MARKET REGIME + BTC STATE (WINNER-GRADE)
    # ==========================================================
    def _get_market_regime(self):
            try:
                k = self.client.get_klines(symbol="BTCUSDT", interval="15m", limit=30)
                df = pd.DataFrame(k, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
                df = df.astype(float)
                df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
                
                adx = float(df['trend_adx'].iloc[-1])
                ema_fast = float(df['trend_ema_fast'].iloc[-1])
                ema_slow = float(df['trend_ema_slow'].iloc[-1])
                
                if adx < 20.0:  # Por debajo de esto, el mercado es lateral puro (chop)
                    return "LATERAL"
                return "ALCISTA" if ema_fast > ema_slow else "BAJISTA"
            except Exception as e:
                ERROR_LOGGER.error(f"Error regime: {str(e)}")
                return "LATERAL"

    def _btc_state(self):
        """
        Retorna:
        - shock: bloqueo duro si extreme move
        - chop_score: score continuo (0..~2)
        """
        try:
            k = self.client.get_klines(symbol="BTCUSDT", interval="1m", limit=90)[:-1]
            closes = np.array([float(x[4]) for x in k])
            opens = np.array([float(x[1]) for x in k])
            highs = np.array([float(x[2]) for x in k])
            lows = np.array([float(x[3]) for x in k])

            last_change = (closes[-1] / opens[-1] - 1) * 100
            last_range = (highs[-1] / lows[-1] - 1) * 100

            drift_5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else 0.0
            rets = np.diff(closes) / closes[:-1]
            vol = np.std(rets[-30:]) * 100 * np.sqrt(1440) if len(rets) >= 30 else np.std(rets) * 100 * np.sqrt(1440)

            shock = (abs(last_change) >= BTC_SHOCK_LAST_CHANGE_PCT) or (last_range >= BTC_SHOCK_LAST_RANGE_PCT) or (abs(drift_5) >= BTC_SHOCK_DRIFT_5M_PCT)

            # Chop score: alto si vol alto + drift bajo
            chop_score = 0.0
            if vol > BTC_CHOP_VOL_MIN and abs(drift_5) < BTC_CHOP_DRIFT_ABS_MAX:
                chop_score = float(np.clip((vol - BTC_CHOP_VOL_MIN) / 1.6 + 1.0, 0.0, 2.2))

            return {
                "shock": bool(shock),
                "chop_score": float(chop_score),
                "last_change": float(last_change),
                "last_range": float(last_range),
                "vol": float(vol),
                "drift_5": float(drift_5)
            }
        except:
            return {"shock": False, "chop_score": 0.0, "last_change": 0.0, "last_range": 0.0, "vol": 0.0, "drift_5": 0.0}

    # ==========================================================
    # EDGE ENGINE
    # ==========================================================
    def _symbol_recent_performance_penalty(self, sym):
        recent = [t for t in self.trade_history[-50:] if t['symbol'] == sym]
        if len(recent) < 6:
            return 0.0
        avg = float(np.mean([t['net'] for t in recent]))
        wr = sum(1 for t in recent if t['net'] > 0) / len(recent)

        penalty = 0.0
        if avg < -0.25 and wr < 0.45:
            penalty -= 1.2
        elif avg < 0 and wr < 0.50:
            penalty -= 0.7
        elif avg > 0.20 and wr > 0.55:
            penalty += 0.3
        return float(penalty)

    def _compute_edge_score(self, sym, strategy_score, ml_pred, regime, metrics):
        s_norm = max(0.0, min(10.0, (strategy_score / 8.5) * 10.0))

        ml_adj = 0.0
        if self.ml_phase != "OBSERVATION":
            ml_adj = float(np.clip(ml_pred, -1.0, 1.5))
            ml_adj = float(np.clip(ml_adj / 1.0, -1.5, 1.5))

        reg_adj = 0.0
        if regime == "ALCISTA":
            reg_adj += 0.6
        elif regime == "BAJISTA":
            reg_adj -= 1.0
        else:
            reg_adj -= 0.1

        v = float(metrics['volatility'])
        vol_adj = 0.0
        if v >= 3.2:
            vol_adj -= 1.0
        elif v >= 2.4:
            vol_adj -= 0.6
        elif v <= 0.9:
            vol_adj -= 0.2

        sym_adj = self._symbol_recent_performance_penalty(sym)

        btc = self._btc_state()
        chop_penalty = 0.0
        if btc["chop_score"] > 0:
            chop_penalty = -float(np.clip(0.35 * btc["chop_score"] + 0.05, 0.0, 0.95))

        edge = s_norm + ml_adj + reg_adj + vol_adj + sym_adj + chop_penalty
        edge = float(np.clip(edge, 0.0, 10.0))

        if edge < 4.8:
            bucket = "LOW"
        elif edge < 7.2:
            bucket = "MEDIUM"
        else:
            bucket = "HIGH"

        return edge, bucket, btc

    def _position_fraction_from_edge(self, edge, bucket, regime, metrics):
            if bucket == "LOW":
                base = 0.20 + (edge / 4.8) * 0.10
            elif bucket == "MEDIUM":
                base = 0.40 + ((edge - 4.8) / (7.2 - 4.8)) * 0.20
            else:
                base = 0.70 + ((edge - 7.2) / (10.0 - 7.2)) * 0.20

            # Soft-Scaling: Reducir posición a la mitad si el mercado está lateral
            if regime == "LATERAL":
                base *= 0.50

            frac = float(base * self.risk_multiplier)
            frac = float(np.clip(frac, 0.10, 0.90)) # Permitimos bajar hasta el 10% del capital

            if self.loss_streak >= 2:
                frac *= 0.50
            if self.win_streak >= 3 and self.loss_streak == 0:
                frac *= 1.25

            return float(np.clip(frac, 0.10, 0.90))

    # ==========================================================
    # TP DYNAMIC: Heuristic + rolling stats
    # ==========================================================
    def _tp_sl_from_atr(self, atr_pct, bucket):
            # Multiplicadores base sobre la volatilidad real
            ATR_SL_MULTIPLIER = 1.8
            ATR_TP_MULTIPLIER = 2.5

            base_sl = -abs(atr_pct * ATR_SL_MULTIPLIER)
            base_tp = abs(atr_pct * ATR_TP_MULTIPLIER)

            # Ajuste por Edge
            if bucket == "HIGH":
                base_sl *= 0.85 # SL más ajustado (menos riesgo) si tenemos alta confianza
                base_tp *= 1.20 # TP más ambicioso
                max_ext = 3
            elif bucket == "MEDIUM":
                max_ext = 2
            else:
                base_tp *= 0.80 # Salir rápido si el edge es bajo
                max_ext = 1

            # Limites sanos absolutos para evitar excesos
            sl = float(np.clip(base_sl, -3.5, -0.6))
            tp = float(np.clip(base_tp, 1.0, 5.0))
            
            return tp, sl, max_ext

    # ==========================================================
    # NO-TRADE ZONES (hard vs soft)
    # ==========================================================
    def _no_trade_zone(self, sym, metrics, regime, edge_score, edge_bucket, btc_state):
            if self.daily_entry_freeze or self.emergency_stop:
                return True, "Daily DD freeze / emergency"

            if (time.time() - self.start_time) < WARMUP_MINUTES * 60:
                return True, "Warm-up"

            # Hard blocks reducidos al mínimo (Solo anomalías extremas)
            if abs(metrics.get("candle_change", 0.0)) >= 3.5:
                return True, f"Candle extreme ({metrics['candle_change']:.2f}%)"
            if metrics.get("candle_range", 0.0) >= 4.0:
                return True, f"Candle range extreme ({metrics['candle_range']:.2f}%)"

            # Eliminamos el bloqueo duro por CHOP o LATERAL. Ahora lo gestionaremos
            # reduciendo el tamaño de la posición (Soft-Scaling) en la Fase 4.

            if self.mode == "SURVIVAL" and regime == "BAJISTA":
                return True, "SURVIVAL: mercado bajista"

            return False, ""

    # ==========================================================
    # DAILY LIMITS + DD ADAPTATIVO
    # ==========================================================
    def _check_daily_limits(self):
        if self.emergency_stop:
            self.daily_entry_freeze = True
            return False

        capital_total = self._get_capital_robust()
        if capital_total <= 0:
            return True

        # Protección de equity global (no descapitalizar la cuenta)
        if self.global_peak_capital <= 0:
            self.global_peak_capital = capital_total
        if capital_total > self.global_peak_capital:
            self.global_peak_capital = capital_total

        if self.global_peak_capital > 0:
            dd_global = ((capital_total / self.global_peak_capital) - 1) * 100
            if dd_global <= self.global_dd_limit:
                if not self.emergency_stop:
                    self.emergency_stop = True
                    self.daily_entry_freeze = True
                    Reporter.emergency_stop(dd_global, self.global_dd_limit, scope="equity")
                return False

        if self.daily_start_capital <= 0:
            self.daily_start_capital = capital_total
            return True

        daily_pnl = ((capital_total / self.daily_start_capital) - 1) * 100

        if daily_pnl < -50:
            time.sleep(2)
            cap2 = self._get_capital_robust()
            pnl2 = ((cap2 / self.daily_start_capital) - 1) * 100
            if pnl2 < -50:
                return True
            daily_pnl = pnl2

        emergency_limit = self.daily_loss_limit * 1.5
        if daily_pnl <= emergency_limit:
            if not self.emergency_stop:
                self.emergency_stop = True
                self.daily_entry_freeze = True
                Reporter.emergency_stop(daily_pnl, emergency_limit, scope="daily")
            return False

        if daily_pnl <= self.daily_loss_limit:
            self.daily_entry_freeze = True
            return False

        params = self._get_mode_params()
        if self.trades_today >= params['max_trades_day']:
            return False

        return True

    def _update_risk_multiplier_after_exit(self, net_pct):
        if net_pct > 0:
            self.win_streak += 1
            self.loss_streak = 0
            if self.win_streak >= 2:
                self.risk_multiplier *= 1.05
        else:
            self.loss_streak += 1
            self.win_streak = 0
            if self.loss_streak == 1:
                self.risk_multiplier *= 0.92
            elif self.loss_streak >= 2:
                self.risk_multiplier *= 0.85

        self.risk_multiplier = float(np.clip(self.risk_multiplier, 0.35, 1.30))

    # ==========================================================
    # BNB check
    # ==========================================================
    def _check_bnb_balance(self):
        """
        Verifica BNB y sugiere recarga dinámica basada en:
        - Trades reales del día (no estimados)
        - Fees realmente gastados
        - Runway proyectado
        - % de ganancia diaria a reinvertir
        """
        try:
            bnb = self._get_bnb_balance()
            bnb_price = float(self.client.get_symbol_ticker(symbol="BNBUSDT")["price"])
            cap = self._get_usdt_total()

            params = self._get_mode_params()

            # ==============================================
            # 1. CALCULAR FEE PROMEDIO REAL DEL DÍA
            # ==============================================
            today = datetime.now(TZ_COLOMBIA).strftime("%Y-%m-%d")
            
            with sqlite3.connect(DB_PATH) as conn:
                # Obtener trades de entrada del día para calcular fee real
                rows = conn.execute("""
                    SELECT usdt FROM trades 
                    WHERE action = 'entry' AND ts LIKE ?
                """, (f"{today}%",)).fetchall()
            
            if rows and len(rows) >= 2:
                # Usar promedio real de los trades del día
                avg_trade_usdt = sum(float(r[0]) for r in rows) / len(rows)
                fee_per_trade_usdt = avg_trade_usdt * FEE_ROUNDTRIP
            else:
                # Fallback: estimar basado en capital y sizing típico
                typical_frac = 0.45
                usdt_per_trade = (cap / max(params['max_positions'], 1)) * typical_frac
                fee_per_trade_usdt = usdt_per_trade * FEE_ROUNDTRIP

            if fee_per_trade_usdt <= 0 or bnb_price <= 0:
                return

            # ==============================================
            # 2. CALCULAR RUNWAY (días de operación)
            # ==============================================
            bnb_value_usdt = bnb * bnb_price
            trades_left = bnb_value_usdt / fee_per_trade_usdt
            
            # Trades reales por día (últimos 3 días)
            with sqlite3.connect(DB_PATH) as conn:
                last_3_days = (datetime.now(TZ_COLOMBIA) - timedelta(days=3)).strftime("%Y-%m-%d")
                count = conn.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE action = 'entry' AND ts >= ?
                """, (f"{last_3_days}%",)).fetchone()[0]
            
            avg_trades_per_day = count / 3.0 if count > 0 else params['max_trades_day'] * 0.6
            avg_trades_per_day = max(3, min(avg_trades_per_day, params['max_trades_day']))
            
            days_runway = trades_left / avg_trades_per_day if avg_trades_per_day > 0 else 0

            # ==============================================
            # 3. DECISIÓN DE ALERTA DINÁMICA
            # ==============================================
            # Umbral: alertar cuando quedan menos de 2 días
            if days_runway >= 2:
                return  # Todo bien, no alertar

            # ==============================================
            # 4. CALCULAR CUÁNTO BNB RECARGAR
            # ==============================================
            # Objetivo: tener fees para N días
            target_days = FEE_RESERVE_DAYS  # 5 días por defecto
            target_fees_usdt = avg_trades_per_day * target_days * fee_per_trade_usdt
            missing_fees_usdt = max(0, target_fees_usdt - bnb_value_usdt)
            
            # Ganancia neta del día en USDT
            with sqlite3.connect(DB_PATH) as conn:
                exits_today = conn.execute("""
                    SELECT usdt, net FROM trades 
                    WHERE action = 'exit' AND ts LIKE ?
                """, (f"{today}%",)).fetchall()
            
            daily_pnl_usdt = sum(
                float(usdt) * (float(net) / 100.0) 
                for usdt, net in exits_today
            ) if exits_today else 0.0
            
            # Solo reinvertir si hubo ganancia
            if daily_pnl_usdt > 0:
                # Sugerir % de la ganancia según urgencia
                if days_runway < 0.5:
                    # CRÍTICO: < 12 horas
                    alloc_pct = 0.60  # 60% de ganancia
                elif days_runway < 1:
                    # URGENTE: < 1 día
                    alloc_pct = 0.45
                else:
                    # PREVENTIVO: 1-2 días
                    alloc_pct = FEE_PROFIT_ALLOCATION  # 35% default
                
                suggested_from_profit = daily_pnl_usdt * alloc_pct
                bnb_to_buy_usdt = min(missing_fees_usdt, suggested_from_profit)
            else:
                # Sin ganancia: sugerir el mínimo necesario
                bnb_to_buy_usdt = min(missing_fees_usdt, cap * 0.05)  # máx 5% del capital
            
            bnb_to_buy = bnb_to_buy_usdt / bnb_price if bnb_price > 0 else 0

            # ==============================================
            # 5. ENVIAR ALERTA CON RECOMENDACIÓN
            # ==============================================
            current_time = time.time()
            
            # Anti-spam: solo alertar cada 4 horas
            if current_time - Reporter.last_bnb_alert_time < 14400:
                return

            urgency = "🚨 CRÍTICO" if days_runway < 0.5 else ("⚠️ URGENTE" if days_runway < 1 else "🔔 PREVENTIVO")
            
            Reporter.send(
                f"{urgency} <b>BNB BAJO - RECARGA GASOLINA</b>\n\n"
                f"📊 <b>ESTADO ACTUAL</b>\n"
                f"BNB disponible: {bnb:.6f} (${bnb_value_usdt:.2f})\n"
                f"Trades restantes: ~{trades_left:.0f}\n"
                f"Runway: <b>{days_runway:.1f} días</b>\n\n"
                f"📈 <b>ACTIVIDAD</b>\n"
                f"Trades/día (promedio): {avg_trades_per_day:.1f}\n"
                f"Fee por trade: ${fee_per_trade_usdt:.3f}\n"
                f"Ganancia hoy: ${daily_pnl_usdt:+.2f}\n\n"
                f"⛽ <b>RECARGA SUGERIDA</b>\n"
                f"Objetivo: {target_days} días de fees\n"
                f"Faltante: ${missing_fees_usdt:.2f}\n"
                f"<b>Recargar: ~{bnb_to_buy:.6f} BNB (${bnb_to_buy_usdt:.2f})</b>\n\n"
                f"💡 <i>{'Usar ' + str(int(alloc_pct*100)) + '% de ganancia del día' if daily_pnl_usdt > 0 else 'Considerar recarga manual'}</i>",
                is_critical=True
            )
            
            Reporter.last_bnb_alert_time = current_time

        except Exception as e:
            ERROR_LOGGER.error(f"Error BNB check: {str(e)}")

    # ==========================================================
    # LOG TRADE
    # ==========================================================
    def _log_trade(self, data):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        data['ts'], data['symbol'], data['action'], data['price'], data['qty'],
                        data['usdt'], data['net'], data['gross'], data['hold'], data['order_id'],
                        ','.join(data['strategies']), data['score'], data['tp'], data['sl'], data['reason'],
                        data.get('momentum', 0), data.get('volatility', 0), data.get('rsi', 0), data.get('macd', 0),
                        data.get('ml_prediction', 0), data.get('regime', ''), data.get('tp_extensions', 0),
                        data.get('edge_score', 0), data.get('edge_bucket', ''), data.get('pos_fraction', 0),
                        data.get('risk_multiplier', 1.0), data.get('no_trade_reason', '')
                    )
                )
        except Exception as e:
            ERROR_LOGGER.error(f"Error DB log: {str(e)}")

        try:
            write_csv_event({
                "timestamp_colombia": data['ts'],
                "event_type": data['action'],
                "symbol": data['symbol'],
                "side": "BUY" if data['action'] == "entry" else "SELL",
                "price": data['price'],
                "qty": data['qty'],
                "usdt_value": data['usdt'],
                "gross_pnl_pct": data['gross'],
                "net_pnl_pct": data['net'],
                "fees_pct": (FEE_ROUNDTRIP * 100) if data['action'] == "exit" else (FEE_SINGLE * 100),
                "hold_minutes": data['hold'],
                "tp_pct": data['tp'],
                "sl_pct": data['sl'],
                "tp_extensions": data.get('tp_extensions', 0),
                "strategies": ",".join(data['strategies']),
                "strategy_score": data.get('score', 0),
                "ml_prediction": data.get('ml_prediction', 0),
                "market_regime": data.get('regime', ''),
                "mode": self.mode,
                "capital_usdt": self._get_usdt_total(),
                "bnb_balance": self._get_bnb_balance(),
                "order_id": data['order_id'],
                "reason": data['reason'],
                "exit_subtype": data.get("exit_subtype", ""),
                "trade_result": data.get("trade_result", ""),
                "sl_at_exit_pct": data.get("sl_at_exit_pct", ""),
                "edge_score": data.get("edge_score", ""),
                "edge_bucket": data.get("edge_bucket", ""),
                "pos_fraction": data.get("pos_fraction", ""),
                "risk_multiplier": data.get("risk_multiplier", ""),
                "no_trade_reason": data.get("no_trade_reason", "")
            })
        except Exception as e:
            ERROR_LOGGER.error(f"Error CSV log: {str(e)}")

        if data['action'] == 'exit':
            self.trade_history.append({
                'symbol': data['symbol'],
                'net': float(data['net']),
                'gross': float(data['gross']),
                'hold': float(data['hold']),
                'strategies': data['strategies'],
                'momentum': float(data.get('momentum', 0) or 0),
                'volatility': float(data.get('volatility', 0) or 0),
                'rsi': float(data.get('rsi', 50) or 50),
                'macd': float(data.get('macd', 0) or 0)
            })
            if len(self.trade_history) % 50 == 0:
                self._train_ml()

    # ==========================================================
    # ENTRY (EDGE)
    # ==========================================================
    def _check_entry(self, sym):
        if not self._check_daily_limits():
            return

        params = self._get_mode_params()
        if len(self.positions) >= params['max_positions']:
            return

        if time.time() < self.cooldown_until.get(sym, 0):
            return

        metrics = self._metrics(sym)
        if not metrics:
            return

        regime = self._get_market_regime()

        strategies = self._strategies(metrics)
        if len(strategies) < params['min_strategies']:
            return

        strategy_score = self._calculate_strategy_score(strategies)
        if strategy_score < params['min_score']:
            return

        ml_pred = self._predict_ml(metrics)

        if self.ml_phase != "OBSERVATION":
            if ml_pred < (params['ml_threshold'] - 0.25) and strategy_score < (params['min_score'] + 1.2):
                return

        edge_score, edge_bucket, btc_state = self._compute_edge_score(sym, strategy_score, ml_pred, regime, metrics)

        blocked, no_trade_reason = self._no_trade_zone(sym, metrics, regime, edge_score, edge_bucket, btc_state)
        if blocked:
            Reporter.no_trade(sym, no_trade_reason, edge_score=edge_score)
            return

        if edge_bucket == "LOW" and edge_score < 4.2:
            return

# Llamada actualizada con regime y metrics para Soft-Scaling
        pos_fraction = self._position_fraction_from_edge(edge_score, edge_bucket, regime, metrics)

        usdt_free = self._get_usdt_free()
        usdt_allocatable = max(0.0, usdt_free * 0.98)
        usdt_to_use = usdt_allocatable * pos_fraction

        slot_cap = (self._get_usdt_total() / max(params['max_positions'], 1)) * 0.95
        usdt_to_use = min(usdt_to_use, slot_cap)
        usdt_to_use = min(usdt_to_use, usdt_free * 0.90)
        usdt_to_use = float(round(usdt_to_use, 2))

        if usdt_to_use < MIN_CAPITAL_PER_TRADE:
            return
        if not self._min_notional_ok(sym, usdt_to_use):
            return

        # Nueva lógica de TP y SL basados en ATR
        tp_base, sl_base, max_ext = self._tp_sl_from_atr(metrics['atr_pct'], edge_bucket)

        try:
            order = self._market_buy(sym, usdt_to_use)
        except Exception as e:
            ERROR_LOGGER.error(f"Fallo entrada {sym}: {str(e)}")
            return

        fills = order.get("fills", [])
        qty = sum(float(f["qty"]) for f in fills)
        cost = sum(float(f["price"]) * float(f["qty"]) for f in fills)
        avg_price = cost / qty if qty > 0 else float(self.client.get_symbol_ticker(symbol=sym)["price"])

        profile = SYMBOL_PROFILES.get(sym, {"trailing_buffer": 1.0, "max_extensions": 3, "volatility_class": "MEDIUM"})

        self.positions[sym] = {
            "qty": qty,
            "price": avg_price,
            "ts": time.time(),
            "tp": float(tp_base),
            "sl": float(sl_base),
            "order_id": order["orderId"],
            "strategies": strategies,
            "ml_prediction": float(ml_pred),
            "tp_extensions": 0,
            "max_price": avg_price,
            "trailing_buffer": float(profile["trailing_buffer"]),
            "max_extensions": int(min(profile["max_extensions"], max_ext)),
            "tp_original": float(tp_base),
            "protection_alerted": False,
            "edge_score": float(edge_score),
            "edge_bucket": edge_bucket,
            "pos_fraction": float(pos_fraction),
            "momentum": float(metrics['momentum']),
            "volatility": float(metrics['volatility']),
            "rsi": float(metrics['rsi']),
            "macd": float(metrics['macd']),
            "regime": regime,
            "strategy_score": float(strategy_score)
        }

        self.cooldown_until[sym] = time.time() + SYMBOL_COOLDOWN_SEC
        self.trades_today += 1

        entry_data = {
            'ts': datetime.now(TZ_COLOMBIA).strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': sym,
            'action': 'entry',
            'price': avg_price,
            'qty': qty,
            'usdt': usdt_to_use,
            'net': 0,
            'gross': 0,
            'hold': 0,
            'order_id': order["orderId"],
            'strategies': strategies,
            'score': float(strategy_score),
            'tp': float(tp_base),
            'sl': float(sl_base),
            'reason': 'Signal(EDGE)',
            'momentum': float(metrics['momentum']),
            'volatility': float(metrics['volatility']),
            'rsi': float(metrics['rsi']),
            'macd': float(metrics['macd']),
            'ml_prediction': float(ml_pred),
            'regime': regime,
            'tp_extensions': 0,
            'edge_score': float(edge_score),
            'edge_bucket': edge_bucket,
            'pos_fraction': float(pos_fraction),
            'risk_multiplier': float(self.risk_multiplier),
            'no_trade_reason': ""
        }

        self._log_trade(entry_data)

        Reporter.entry(sym, avg_price, strategy_score, strategies, usdt_to_use,
                       tp_base, sl_base, order["orderId"], ml_pred, regime,
                       edge_score, edge_bucket, pos_fraction, self.risk_multiplier)

    # ==========================================================
    # EXIT — SL dinámico mejorado + trailing post-TP real + PnL USDT
    # ==========================================================
    def _check_exit(self, sym):
        pos = self.positions.get(sym)
        if not pos:
            return

        try:
            p = float(self.client.get_symbol_ticker(symbol=sym)["price"])
        except:
            return

        entry_price = float(pos["price"])
        gross = (p / entry_price - 1) * 100
        net = gross - (FEE_ROUNDTRIP * 100)
        hold = (time.time() - pos["ts"]) / 60

        current_max = float(pos.get("max_price", entry_price))
        if p > current_max:
            pos["max_price"] = p
            current_max = p

        max_gross = (current_max / entry_price - 1) * 100
        max_net = max_gross - (FEE_ROUNDTRIP * 100)

        current_sl = float(pos.get("sl", -1.2))
        tp_extensions = int(pos.get("tp_extensions", 0))

        if sym in ["BTCUSDT", "ETHUSDT"]:
            base_buffer = PROTECT_BUFFER_TIGHT
        elif sym in ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "LINKUSDT", "FETUSDT", "INJUSDT"]:
            base_buffer = PROTECT_BUFFER_WIDE
        else:
            base_buffer = PROTECT_BUFFER_BASE

        if tp_extensions == 0:

            # ==============================
            # SL dinámico base (pre-TP)
            # ==============================
            if max_net >= PROTECT_ACTIVATION_NET:
                proposed = max_net - base_buffer
                proposed = min(proposed, max_net - PROTECT_MAX_LOCK)
                current_sl = max(current_sl, proposed)

            # ==============================
            # PISO NETO REAL INTELIGENTE
            # ==============================
            try:
                vol_now = float(self._metrics(sym)['volatility'])
            except:
                vol_now = 2.0

            if vol_now < 1.2:
                breakeven_buffer = 0.03
            elif vol_now < 2.5:
                breakeven_buffer = 0.05
            else:
                breakeven_buffer = 0.08

            if pos.get("momentum", 0) > 1.0:
                breakeven_buffer += 0.02

            net_floor = breakeven_buffer

            if max_net >= PROTECT_ACTIVATION_NET:
                current_sl = max(current_sl, net_floor)

            # Refuerzo definitivo del piso
            if max_net >= PROTECT_ACTIVATION_NET:
                current_sl = max(current_sl, net_floor)

            # ==============================
            # Segundo ajuste dinámico
            # ==============================
            if max_net >= 1.60:
                proposed2 = max_net - max(base_buffer * 0.85, 0.70)
                proposed2 = min(proposed2, max_net - 0.30)
                current_sl = max(current_sl, proposed2)

            # 🔒 Garantía final: nunca por debajo del piso
            if max_net >= PROTECT_ACTIVATION_NET:
                current_sl = max(current_sl, net_floor)

        else:
            try:
                vol = float(self._metrics(sym)['volatility'])
            except:
                vol = 2.0

            trail_buffer = float(np.clip(
                0.65 + (vol / 6.0) + (base_buffer * 0.35),
                POST_TP_TRAIL_MIN,
                POST_TP_TRAIL_MAX
            ))

            proposed = max_net - trail_buffer
            current_sl = max(current_sl, proposed)

            tp_original = float(pos.get("tp_original", 1.8))
            last_tp_reached = tp_original * tp_extensions
            floor_sl = last_tp_reached - float(pos.get("trailing_buffer", 1.0))
            current_sl = max(current_sl, floor_sl)

        pos["sl"] = float(current_sl)

        if not pos.get("protection_alerted") and current_sl > 0:
            sl_price = entry_price * (1 + current_sl / 100)
            margin = net - current_sl
            Reporter.protection_activated(sym, entry_price, p, current_sl, sl_price, margin)
            pos["protection_alerted"] = True

        current_tp = float(pos.get("tp", 1.8))
        max_extensions = int(pos.get("max_extensions", 3))

        reason = None

        if net >= current_tp - 0.02:
            if tp_extensions < max_extensions:
                old_tp = current_tp
                tp_original = float(pos.get("tp_original", 1.8))
                new_tp = current_tp + tp_original
                new_extensions = tp_extensions + 1

                buffer = float(pos.get("trailing_buffer", 1.0))
                new_sl = max(current_sl, old_tp - buffer)

                pos["tp"] = new_tp
                pos["tp_extensions"] = new_extensions
                pos["sl"] = new_sl

                Reporter.tp_extension(sym, net, old_tp, new_tp, new_sl, new_extensions, max_extensions)
                return
            else:
                reason = f"TP Final ({tp_extensions}x)"

        elif net <= current_sl:
            reason = "Stop Loss"
        else:
            return

        asset = sym.replace("USDT", "")
        try:
            balance = float(self.client.get_asset_balance(asset=asset)["free"])
        except:
            Reporter.warning(f"Error obteniendo balance {asset}")
            return

        if balance <= 0:
            Reporter.warning(f"Cerrado manual: {sym}")
            del self.positions[sym]
            self.cooldown_until[sym] = time.time() + SYMBOL_COOLDOWN_SEC
            return

        raw_qty = min(float(pos["qty"]), balance)
        qty_to_sell = self.adjust_qty_to_step(sym, raw_qty)
        if not qty_to_sell or qty_to_sell == "0":
            Reporter.warning(f"Qty inválida para vender {sym}: {qty_to_sell}")
            return

        qty_float = float(qty_to_sell)
        try:
            order = self._market_sell_qty(sym, qty_to_sell)
        except Exception as e:
            ERROR_LOGGER.error(f"Fallo venta {sym}: {str(e)}")
            return

        notional_entry = entry_price * qty_float
        pnl_usdt = (p - entry_price) * qty_float - (FEE_ROUNDTRIP * notional_entry)

        trade_result = "WIN" if net > 0 else "LOSS"
        exit_subtype = "SL_TRAIL" if reason == "Stop Loss" and net > 0 else reason

        self._update_risk_multiplier_after_exit(net)

        exit_data = {
            'ts': datetime.now(TZ_COLOMBIA).strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': sym,
            'action': 'exit',
            'price': p,
            'qty': float(qty_to_sell),
            'usdt': float(qty_to_sell) * p,
            'net': float(net),
            'gross': float(gross),
            'hold': float(hold),
            'order_id': order["orderId"],
            'strategies': pos.get("strategies", []),
            'score': float(pos.get("strategy_score", 0.0)),
            'tp': float(current_tp),
            'sl': float(current_sl),
            'reason': reason,
            'momentum': float(pos.get("momentum", 0)),
            'volatility': float(pos.get("volatility", 0)),
            'rsi': float(pos.get("rsi", 0)),
            'macd': float(pos.get("macd", 0)),
            'ml_prediction': float(pos.get("ml_prediction", 0)),
            'regime': pos.get("regime", ''),
            'tp_extensions': int(tp_extensions),
            'exit_subtype': exit_subtype,
            'trade_result': trade_result,
            'sl_at_exit_pct': float(current_sl),
            'edge_score': float(pos.get("edge_score", 0)),
            'edge_bucket': pos.get("edge_bucket", ""),
            'pos_fraction': float(pos.get("pos_fraction", 0)),
            'risk_multiplier': float(self.risk_multiplier),
            'no_trade_reason': ""
        }

        self._log_trade(exit_data)

        Reporter.exit(sym, net, gross, pnl_usdt, reason, hold, tp_extensions,
                    order["orderId"], self.risk_multiplier,
                    self.win_streak, self.loss_streak)

        del self.positions[sym]
        self.cooldown_until[sym] = time.time() + SYMBOL_COOLDOWN_SEC

    # ==========================================================
    # ACCOUNTING DIARIO (PnL USDT + fees + recomendación BNB)
    # ==========================================================
    def _compute_daily_accounting(self, start_ts, end_ts, capital_start, capital_end, bnb_end):
        result = {
            "pnl_usdt": 0.0,
            "fees_today_usdt": 0.0,
            "fees_target_reserve_usdt": 0.0,
            "fees_current_reserve_usdt": 0.0,
            "fees_to_invest_usdt": 0.0,
            "fees_to_invest_bnb": 0.0,
            "avg_trade_usdt": 0.0,
            "trades_exits": 0
        }
        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT usdt, net
                    FROM trades
                    WHERE action = 'exit' AND ts BETWEEN ? AND ?
                """, (start_ts, end_ts)).fetchall()
        except Exception as e:
            ERROR_LOGGER.error(f"Error contabilidad diaria: {str(e)}")
            return result

        if not rows:
            return result

        total_notional = 0.0
        pnl_usdt = 0.0
        fees_today_usdt = 0.0
        for usdt, net in rows:
            usdt_val = abs(float(usdt) or 0.0)
            net_pct = float(net or 0.0)
            total_notional += usdt_val
            pnl_usdt += usdt_val * (net_pct / 100.0)
            fees_today_usdt += usdt_val * FEE_ROUNDTRIP

        avg_trade_usdt = total_notional / len(rows) if rows else 0.0

        try:
            bnb_price = float(self.client.get_symbol_ticker(symbol="BNBUSDT")["price"])
        except:
            bnb_price = 0.0

        mode_params = self._get_mode_params()
        expected_trades_per_day = min(max(len(rows), 4), mode_params['max_trades_day'])
        fee_per_trade_usdt = avg_trade_usdt * FEE_ROUNDTRIP if avg_trade_usdt > 0 else 0.0
        target_fee_reserve = expected_trades_per_day * FEE_RESERVE_DAYS * fee_per_trade_usdt

        current_fee_reserve = bnb_end * bnb_price if bnb_price > 0 else 0.0
        missing_fee_reserve = max(0.0, target_fee_reserve - current_fee_reserve)

        profit_usdt = max(0.0, pnl_usdt)
        max_alloc_from_profit = profit_usdt * FEE_PROFIT_ALLOCATION
        fees_to_invest_usdt = min(missing_fee_reserve, max_alloc_from_profit)
        fees_to_invest_bnb = fees_to_invest_usdt / bnb_price if bnb_price > 0 else 0.0

        result.update({
            "pnl_usdt": pnl_usdt,
            "fees_today_usdt": fees_today_usdt,
            "fees_target_reserve_usdt": target_fee_reserve,
            "fees_current_reserve_usdt": current_fee_reserve,
            "fees_to_invest_usdt": fees_to_invest_usdt,
            "fees_to_invest_bnb": fees_to_invest_bnb,
            "avg_trade_usdt": avg_trade_usdt,
            "trades_exits": len(rows)
        })
        return result

    # ==========================================================
    # DAILY REPORT
    # ==========================================================
    def _daily_report(self):
        now = datetime.now(TZ_COLOMBIA)
        report_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        if self.last_daily == report_date:
            return

        try:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute("SELECT capital_start FROM daily_snapshots WHERE date = ?", (report_date,)).fetchone()
        except:
            return
        if not row:
            return

        capital_start = float(row[0] or 0)
        capital_end = self._get_capital_robust()
        bnb_end = self._get_bnb_balance()

        start_day = f"{report_date} 00:00:00"
        end_day = f"{report_date} 23:59:59"

        try:
            with sqlite3.connect(DB_PATH) as conn:
                rows = conn.execute("""
                    SELECT net FROM trades
                    WHERE action = 'exit' AND ts BETWEEN ? AND ?
                """, (start_day, end_day)).fetchall()
        except:
            rows = []

        nets = [float(r[0]) for r in rows]
        wins = len([n for n in nets if n > 0])
        losses = len([n for n in nets if n <= 0])

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE daily_snapshots
                    SET capital_end = ?, bnb_end = ?, mode = ?,
                        risk_multiplier = ?, win_streak = ?, loss_streak = ?
                    WHERE date = ?
                """, (capital_end, bnb_end, self.mode, self.risk_multiplier, self.win_streak, self.loss_streak, report_date))
        except Exception as e:
            ERROR_LOGGER.error(f"Error actualizando snapshot: {str(e)}")

        pnl_pct = ((capital_end / capital_start) - 1) * 100 if capital_start > 0 else 0.0

        ml_status = f"{self.ml_phase} ({len(self.trade_history)} trades)"
        if self.ml_models:
            ml_status += f" - {self.ml_models.get('precision', 0):.1f}%"

        accounting = self._compute_daily_accounting(start_day, end_day, capital_start, capital_end, bnb_end)

        report = {
            "date": report_date,
            "cap_start": capital_start,
            "cap_end": capital_end,
            "pnl": pnl_pct,
            "trades": len(nets),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(nets) * 100) if nets else 0,
            "best": max(nets) if nets else 0,
            "worst": min(nets) if nets else 0,
            "mode": self.mode,
            "ml_status": ml_status,
            "bnb": bnb_end,
            "risk_mult": self.risk_multiplier
        }
        report.update(accounting)
        report["fee_reserve_days"] = FEE_RESERVE_DAYS

        Reporter.daily(report)
        self.last_daily = report_date

        today = now.strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(DB_PATH) as conn:
                exists = conn.execute("SELECT 1 FROM daily_snapshots WHERE date = ?", (today,)).fetchone()
                if not exists:
                    conn.execute("""
                        INSERT INTO daily_snapshots
                        (date, capital_start, capital_end, bnb_start, bnb_end, mode, risk_multiplier, win_streak, loss_streak)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (today, capital_end, capital_end, bnb_end, bnb_end, self.mode,
                          self.risk_multiplier, self.win_streak, self.loss_streak))
        except Exception as e:
            ERROR_LOGGER.error(f"Error creando snapshot nuevo día: {str(e)}")

        self.daily_start_capital = capital_end
        self.trades_today = 0
        self.emergency_stop = False
        self.daily_entry_freeze = False

        self._train_ml()

    # ==========================================================
    # MAIN LOOP
    # ==========================================================
    def run(self):
        ERROR_LOGGER.error("BOT STARTED v4.0 PRO ACCOUNTING")

        while True:
            try:
                self._daily_report()
                self._check_bnb_balance()
                self._determine_mode()

                for sym in self.symbols:
                    if not self.symbol_ready.get(sym, False):
                        self.symbol_ready[sym] = True
                        continue

                    if time.time() < self.cooldown_until.get(sym, 0):
                        continue

                    if sym in self.positions:
                        self._check_exit(sym)
                    else:
                        self._check_entry(sym)

                time.sleep(1)

            except Exception as e:
                ERROR_LOGGER.error(f"Error loop principal: {str(e)}")
                Reporter.warning(f"Error: {str(e)}")
                time.sleep(5)

if __name__ == "__main__":
    OmegaEvolutionary().run()
