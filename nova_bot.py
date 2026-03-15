import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from mt5linux import MetaTrader5
mt5 = MetaTrader5(host="localhost", port=18812)
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Variables environnement Railway
MT5_LOGIN    = int(os.environ.get("MT5_LOGIN", "0"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER   = os.environ.get("MT5_SERVER", "")

# ============================================================
#  NOVA IA V7.1 — SMC/ICT + FIBONACCI + HTF CASCADE
#  CASCADE : H4 → H1 → M30 → M15
#  SYMBOLES : XAUUSD | DE40 | US30 | USTEC | US500
#  V7.0 : BOS/CHoCH | Sweep | OB qualité | Div RSI | IPDA
#          SL structure | Trail OB | DD auto | Corrélation | CSV
#  V7.1 : Equal Highs/Lows | Stop Hunt EQH/EQL | Absorption
#          Wick disproportionné | Volume spike fake move
# ============================================================

class OrderType(Enum):
    BUY  = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL

class TradingStrategy(Enum):
    SMC_ICT       = "SMC_ICT"
    FIBONACCI_SMC = "FIBONACCI_SMC"

class MarketBias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class SymbolSpecs:
    name: str
    pip_value: float
    min_distance: float
    digits: int
    point: float
    spread: float
    tick_value: float
    tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float


@dataclass
class TradeSignal:
    symbol: str
    timeframe: str
    direction: str
    order_type: str
    entry: float
    sl: float
    tp: float
    confidence: float
    strength: int
    timestamp: datetime
    reason: str
    trend_aligned: bool
    strategy: TradingStrategy
    momentum_state: str
    num_reasons: int
    fib_level: Optional[float] = None
    htf_bias: MarketBias = MarketBias.NEUTRAL
    has_sweep: bool = False
    has_bos: bool = False


@dataclass
class ManagedPosition:
    ticket: int
    symbol: str
    direction: str
    volume: float
    entry_price: float
    sl: float
    tp: float
    open_time: datetime
    tp_levels: List[float]
    tp_hit: List[bool]
    breakeven_set: bool = False
    trailing_active: bool = False
    highest_profit: float = 0.0
    partial_closed: int = 0
    strategy: TradingStrategy = TradingStrategy.SMC_ICT
    last_ob_sl: float = 0.0


@dataclass
class TradingStats:
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    current_streak: int = 0
    best_streak: int = 0


# ============================================================
#  CLASSE PRINCIPALE
# ============================================================

class AutoTrader:
    """
    NOVA IA V7.0 — SMC/ICT + FIBONACCI
    CASCADE HTF : H4 → H1 → M30 → M15
    """

    AUTHORIZED_SYMBOLS = ["XAUUSD", "DE40", "US30", "USTEC", "US500"]

    SYMBOL_LIMITS = {
        "XAUUSD": {"min": 0.01, "max": 0.50},
        "DE40":   {"min": 0.10, "max": 5.0},
        "US30":   {"min": 0.10, "max": 5.0},
        "USTEC":  {"min": 0.10, "max": 5.0},
        "US500":  {"min": 0.10, "max": 5.0},
    }

    CORRELATED_PAIRS = [
        {"US30", "USTEC"},
        {"US30", "US500"},
        {"USTEC", "US500"},
    ]

    def __init__(self, capital: float = 10000, risk_per_trade: float = 0.01,
                 mt5_login: int = None, mt5_password: str = None,
                 mt5_server: str = None, use_momentum_filter: bool = False):

        self.capital             = capital
        self.initial_capital     = capital
        self.risk_per_trade      = risk_per_trade
        self.max_positions       = 5
        self.use_momentum_filter = use_momentum_filter

        self.mt5_login    = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server   = mt5_server

        # Critères relevés
        self.MIN_SCORE      = 55   # calibré backtest
        self.MIN_CONFIDENCE = 72   # confiance min
        self.MIN_REASONS    = 6    # raisons min

        self.use_4_tp  = capital >= 2000
        self.TP_RATIOS = [1.5, 2.0, 2.5, 3.0] if self.use_4_tp else [1.5, 2.5]

        # Protection capital
        self.MAX_DAILY_DRAWDOWN = 0.05
        self.day_start_balance  = 0.0
        self.trading_halted     = False

        self.symbol_specs: Dict[str, SymbolSpecs]          = {}
        self.active_positions: Dict[int, ManagedPosition]  = {}
        self._indicator_cache: Dict[str, pd.DataFrame]     = {}

        self.total_trades   = 0
        self.winning_trades = 0
        self.total_pnl      = 0.0
        self.stats          = TradingStats()

        self.monitoring        = False
        self.monitoring_thread = None

        self.log_file = "nova_v7_trades.csv"
        self._init_log()

        if not self.connect_mt5():
            raise Exception("❌ Échec connexion MT5")

        self.print_banner()

    # ── LOG CSV ───────────────────────────────────────────────

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "symbol", "direction", "strategy",
                             "entry", "sl", "tp", "lot", "score", "confidence",
                             "reasons", "has_sweep", "has_bos", "htf_bias", "result", "pnl"])

    def _log_trade(self, signal: TradeSignal, lot: float, result: str = "", pnl: float = 0.0):
        with open(self.log_file, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                signal.symbol, signal.direction, signal.strategy.value,
                signal.entry, signal.sl, signal.tp, lot,
                signal.strength, f"{signal.confidence:.1f}",
                signal.num_reasons, signal.has_sweep, signal.has_bos,
                signal.htf_bias.value, result, f"{pnl:.2f}"
            ])

    # ── CONNEXION MT5 ─────────────────────────────────────────

    def connect_mt5(self) -> bool:
        try:
            if self.mt5_login and self.mt5_password and self.mt5_server:
                print(f"🔐 Connexion au compte {self.mt5_login}...")
                if not mt5.initialize(login=self.mt5_login, password=self.mt5_password,
                                      server=self.mt5_server):
                    print("❌ Identifiants échoués, tentative défaut...")
                    if not mt5.initialize():
                        return False
            else:
                if not mt5.initialize():
                    return False
            account = mt5.account_info()
            if account is None:
                return False
            self.day_start_balance = account.balance
            print(f"✅ CONNECTÉ — Compte: {account.login} | Balance: ${account.balance:,.2f}")
            return True
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
            return False

    def ensure_connected(self) -> bool:
        if mt5.account_info() is not None:
            return True
        print("⚠️ Connexion perdue — reconnexion...")
        for attempt in range(3):
            time.sleep(5)
            if self.connect_mt5():
                print("✅ Reconnexion réussie")
                return True
            print(f"   Tentative {attempt+1}/3 échouée")
        print("❌ Reconnexion impossible")
        return False

    # ── BANNER ────────────────────────────────────────────────

    def print_banner(self):
        tp_ratios = ", ".join([f"1:{r}" for r in self.TP_RATIOS])
        mom = "✅ ACTIVÉ" if self.use_momentum_filter else "❌ DÉSACTIVÉ"
        print(f"\n{'='*100}")
        print(f"{'⚔️  NOVA IA V7.0 — SMC/ICT + FIBONACCI + HTF CASCADE ⚔️':^100}")
        print(f"{'='*100}")
        print(f"║ Capital: ${self.capital:,.2f} | Risk: {self.risk_per_trade*100:.1f}% | Max Pos: {self.max_positions}{' '*30} ║")
        print(f"║ CASCADE TF : H4 → H1 → M30 → M15{' '*62} ║")
        print(f"║ Symboles : XAUUSD | DE40 | US30 | USTEC | US500{' '*46} ║")
        print(f"║ TP: {tp_ratios} | BE: 70% | TRAIL sur OB | DD MAX: 5%{' '*28} ║")
        print(f"║ Score≥{self.MIN_SCORE} | Conf≥{self.MIN_CONFIDENCE}% | Min {self.MIN_REASONS} raisons | MOMENTUM: {mom}{' '*16} ║")
        print(f"║ BOS/CHoCH | Sweep | Inducement | OB Fresh | Div RSI | Log CSV{' '*30} ║")
        print(f"{'='*100}\n")

    # ── SPECS SYMBOLE ─────────────────────────────────────────

    def get_symbol_specs(self, symbol: str) -> Optional[SymbolSpecs]:
        if symbol not in self.AUTHORIZED_SYMBOLS:
            return None
        if symbol in self.symbol_specs:
            return self.symbol_specs[symbol]
        info = mt5.symbol_info(symbol)
        if info is None:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            return None
        pip_value = 0.1 if "XAU" in symbol else 1.0
        tick = mt5.symbol_info_tick(symbol)
        spread = (tick.ask - tick.bid) if tick else 0
        specs = SymbolSpecs(
            name=symbol, pip_value=pip_value,
            min_distance=max(info.trade_stops_level * info.point, pip_value * 2),
            digits=info.digits, point=info.point, spread=spread,
            tick_value=info.trade_tick_value, tick_size=info.trade_tick_size,
            volume_min=self.SYMBOL_LIMITS[symbol]["min"],
            volume_max=self.SYMBOL_LIMITS[symbol]["max"],
            volume_step=info.volume_step,
        )
        self.symbol_specs[symbol] = specs
        return specs

    # ── DONNÉES MT5 ───────────────────────────────────────────

    def get_live_data(self, symbol: str, timeframe: str, bars: int = 300) -> Optional[pd.DataFrame]:
        tf_map = {
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1':  mt5.TIMEFRAME_H1,
            'H4':  mt5.TIMEFRAME_H4,
        }
        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            return None
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    # ── INDICATEURS (avec cache) ──────────────────────────────

    def calculate_indicators(self, df: pd.DataFrame, tf_label: str = "") -> pd.DataFrame:
        cache_key = f"{tf_label}_{df['time'].iloc[-1]}"
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]

        df = df.copy()

        # ATR
        df['prev_close'] = df['close'].shift(1)
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['prev_close'])
        lc = abs(df['low']  - df['prev_close'])
        df['tr']  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR'] = df['tr'].rolling(14).mean()

        # Candle
        df['candle_size']     = abs(df['close'] - df['open'])
        df['candle_size_avg'] = df['candle_size'].rolling(20).mean()
        df['big_candle']      = df['candle_size'] > (df['candle_size_avg'] * 2.5)
        df['bullish_candle']  = df['close'] > df['open']
        df['bearish_candle']  = df['close'] < df['open']

        # EMAs
        for span in [9, 21, 50, 100, 200]:
            df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))

        # Divergence RSI
        df['rsi_div_bull'] = False
        df['rsi_div_bear'] = False
        for i in range(5, len(df)):
            if df['close'].iloc[i] < df['close'].iloc[i-5] and df['RSI'].iloc[i] > df['RSI'].iloc[i-5]:
                df.loc[df.index[i], 'rsi_div_bull'] = True
            if df['close'].iloc[i] > df['close'].iloc[i-5] and df['RSI'].iloc[i] < df['RSI'].iloc[i-5]:
                df.loc[df.index[i], 'rsi_div_bear'] = True

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD']        = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

        # Stochastic
        low14  = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        df['Stoch_K'] = 100 * (df['close'] - low14) / (high14 - low14)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        # ADX
        dmp = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                       np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        dmm = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                       np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        tr14 = df['tr'].rolling(14).mean()
        df['DIplus']  = 100 * (pd.Series(dmp, index=df.index).rolling(14).mean() / tr14)
        df['DIminus'] = 100 * (pd.Series(dmm, index=df.index).rolling(14).mean() / tr14)
        dx = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
        df['ADX'] = dx.rolling(14).mean()

        # Order Blocks (qualité pondérée — OB frais vs déjà touché)
        df['bullish_ob']       = False
        df['bearish_ob']       = False
        df['bullish_ob_fresh'] = False
        df['bearish_ob_fresh'] = False
        for i in range(3, len(df)):
            if (df['close'].iloc[i]   > df['open'].iloc[i] and
                    df['close'].iloc[i-1] < df['open'].iloc[i-1] and
                    df['close'].iloc[i]   > df['high'].iloc[i-1]):
                df.loc[df.index[i-1], 'bullish_ob'] = True
                ob_low = df['low'].iloc[i-1]
                if len(df) > i and df['low'].iloc[i:].min() > ob_low * 0.9995:
                    df.loc[df.index[i-1], 'bullish_ob_fresh'] = True
            if (df['close'].iloc[i]   < df['open'].iloc[i] and
                    df['close'].iloc[i-1] > df['open'].iloc[i-1] and
                    df['close'].iloc[i]   < df['low'].iloc[i-1]):
                df.loc[df.index[i-1], 'bearish_ob'] = True
                ob_high = df['high'].iloc[i-1]
                if len(df) > i and df['high'].iloc[i:].max() < ob_high * 1.0005:
                    df.loc[df.index[i-1], 'bearish_ob_fresh'] = True

        # FVG
        df['fvg_bullish'] = False
        df['fvg_bearish'] = False
        for i in range(1, len(df) - 1):
            if df['low'].iloc[i+1]  > df['high'].iloc[i-1]:
                df.loc[df.index[i], 'fvg_bullish'] = True
            if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                df.loc[df.index[i], 'fvg_bearish'] = True

        # BOS — Break of Structure
        df['bos_bull'] = False
        df['bos_bear'] = False
        for i in range(10, len(df)):
            prev_high = df['high'].iloc[i-10:i].max()
            prev_low  = df['low'].iloc[i-10:i].min()
            if df['close'].iloc[i] > prev_high:
                df.loc[df.index[i], 'bos_bull'] = True
            if df['close'].iloc[i] < prev_low:
                df.loc[df.index[i], 'bos_bear'] = True

        # CHoCH — Change of Character
        df['choch_bull'] = False
        df['choch_bear'] = False
        for i in range(5, len(df)):
            if df['bos_bear'].iloc[i-5:i].any() and df['close'].iloc[i] > df['high'].iloc[i-5:i].max():
                df.loc[df.index[i], 'choch_bull'] = True
            if df['bos_bull'].iloc[i-5:i].any() and df['close'].iloc[i] < df['low'].iloc[i-5:i].min():
                df.loc[df.index[i], 'choch_bear'] = True

        # Liquidity Sweep
        df['sweep_high'] = False
        df['sweep_low']  = False
        for i in range(20, len(df)):
            key_high = df['high'].iloc[i-20:i-1].max()
            key_low  = df['low'].iloc[i-20:i-1].min()
            if df['high'].iloc[i] > key_high and df['close'].iloc[i] < key_high:
                df.loc[df.index[i], 'sweep_high'] = True
            if df['low'].iloc[i] < key_low and df['close'].iloc[i] > key_low:
                df.loc[df.index[i], 'sweep_low'] = True

        # ── EQUAL HIGHS / EQUAL LOWS ─────────────────────────
        # Deux highs ou lows quasi-identiques (±0.05%) = pool de liquidité
        df['eqh'] = False   # Equal High — liquidité au-dessus
        df['eql'] = False   # Equal Low  — liquidité en-dessous
        eq_tolerance = 0.0005   # 0.05%
        for i in range(3, len(df)):
            for j in range(1, 4):
                if i - j < 0:
                    break
                # Equal High : deux highs dans la même zone
                hi, hj = df['high'].iloc[i], df['high'].iloc[i-j]
                if hi > 0 and abs(hi - hj) / hi <= eq_tolerance:
                    df.loc[df.index[i], 'eqh'] = True
                # Equal Low : deux lows dans la même zone
                li, lj = df['low'].iloc[i], df['low'].iloc[i-j]
                if li > 0 and abs(li - lj) / li <= eq_tolerance:
                    df.loc[df.index[i], 'eql'] = True

        # ── DÉTECTION MANIPULATION ────────────────────────────

        # 1. Wick disproportionné sur OB (wick ≥ 3× corps)
        df['wick_up']   = df['high'] - df[['open', 'close']].max(axis=1)
        df['wick_down'] = df[['open', 'close']].min(axis=1) - df['low']
        body            = df['candle_size'].replace(0, np.nan)
        df['manip_wick_bull'] = (df['wick_down'] >= body * 3) & (df['bullish_candle'])
        df['manip_wick_bear'] = (df['wick_up']   >= body * 3) & (df['bearish_candle'])

        # 2. Volume spike sans suivi de prix (fake move)
        # Gros volume mais prix ne bouge pas → manipulation / absorption
        if 'tick_volume' in df.columns:
            vol_ma   = df['tick_volume'].rolling(20).mean()
            big_vol  = df['tick_volume'] > vol_ma * 2.0
            small_move = df['candle_size'] < df['candle_size_avg'] * 0.5
            df['manip_fake_move'] = big_vol & small_move
        else:
            df['manip_fake_move'] = False

        # 3. Absorption institutionnelle
        # Volume très fort + clôture dans le milieu de la bougie = absorption
        if 'tick_volume' in df.columns:
            vol_ma      = df['tick_volume'].rolling(20).mean()
            strong_vol  = df['tick_volume'] > vol_ma * 1.8
            candle_mid  = (df['high'] + df['low']) / 2
            close_near_mid = abs(df['close'] - candle_mid) < (df['high'] - df['low']) * 0.2
            df['manip_absorption_bull'] = strong_vol & close_near_mid & (df['low'] < df['low'].shift(1))
            df['manip_absorption_bear'] = strong_vol & close_near_mid & (df['high'] > df['high'].shift(1))
        else:
            df['manip_absorption_bull'] = False
            df['manip_absorption_bear'] = False

        # 4. Stop hunt sur Equal Highs / Equal Lows
        # Wick dépasse EQH ou EQL puis revient → stop hunt confirmé
        df['stophunt_bull'] = False   # stop hunt baissier sur EQL → long opportunity
        df['stophunt_bear'] = False   # stop hunt haussier sur EQH → short opportunity
        for i in range(5, len(df)):
            # Stop hunt sur EQL (wick passe sous equal low puis remonte)
            if any(df['eql'].iloc[i-5:i]):
                eq_low_zone = df.loc[df['eql'].iloc[i-5:i][df['eql'].iloc[i-5:i]].index, 'low'].min() if df['eql'].iloc[i-5:i].any() else None
                if eq_low_zone is not None:
                    if df['low'].iloc[i] < eq_low_zone * 0.9998 and df['close'].iloc[i] > eq_low_zone:
                        df.loc[df.index[i], 'stophunt_bull'] = True
            # Stop hunt sur EQH (wick passe au dessus equal high puis redescend)
            if any(df['eqh'].iloc[i-5:i]):
                eq_high_zone = df.loc[df['eqh'].iloc[i-5:i][df['eqh'].iloc[i-5:i]].index, 'high'].max() if df['eqh'].iloc[i-5:i].any() else None
                if eq_high_zone is not None:
                    if df['high'].iloc[i] > eq_high_zone * 1.0002 and df['close'].iloc[i] < eq_high_zone:
                        df.loc[df.index[i], 'stophunt_bear'] = True

        df.drop(['wick_up', 'wick_down'], axis=1, inplace=True, errors='ignore')

        # ── MARKET STRUCTURE SHIFT (MSS) ──────────────────────
        # Plus précis que BOS : swing high cassé + nouveau higher low confirmé
        # Bullish MSS : prix casse un swing high précédent ET a formé un higher low
        # Bearish MSS : prix casse un swing low précédent ET a formé un lower high
        df['mss_bull'] = False
        df['mss_bear'] = False
        for i in range(15, len(df)):
            swing_high = df['high'].iloc[i-15:i-3].max()
            swing_low  = df['low'].iloc[i-15:i-3].min()
            # Bullish MSS : clôture au-dessus du swing high + higher low récent
            if df['close'].iloc[i] > swing_high:
                recent_low  = df['low'].iloc[i-5:i].min()
                prior_low   = df['low'].iloc[i-15:i-5].min()
                if recent_low > prior_low:   # higher low confirmé
                    df.loc[df.index[i], 'mss_bull'] = True
            # Bearish MSS : clôture en-dessous du swing low + lower high récent
            if df['close'].iloc[i] < swing_low:
                recent_high = df['high'].iloc[i-5:i].max()
                prior_high  = df['high'].iloc[i-15:i-5].max()
                if recent_high < prior_high:  # lower high confirmé
                    df.loc[df.index[i], 'mss_bear'] = True

        # ── MITIGATION BLOCK ──────────────────────────────────
        # OB déjà touché une fois → deuxième toucher souvent plus fiable
        # car les positions faibles ont été éliminées au premier passage
        df['mitigation_bull'] = False
        df['mitigation_bear'] = False
        ob_bull_zones = []
        ob_bear_zones = []
        for i in range(3, len(df)):
            # Enregistre les zones OB
            if df['bullish_ob'].iloc[i-1]:
                ob_bull_zones.append({
                    'idx': i-1, 'low': df['low'].iloc[i-1],
                    'high': df['high'].iloc[i-1], 'touches': 0
                })
            if df['bearish_ob'].iloc[i-1]:
                ob_bear_zones.append({
                    'idx': i-1, 'low': df['low'].iloc[i-1],
                    'high': df['high'].iloc[i-1], 'touches': 0
                })
            # Vérifie retour sur zone OB bull
            for zone in ob_bull_zones:
                if (df['low'].iloc[i] <= zone['high'] and
                        df['close'].iloc[i] >= zone['low']):
                    zone['touches'] += 1
                    if zone['touches'] == 2:   # deuxième toucher
                        df.loc[df.index[i], 'mitigation_bull'] = True
            # Vérifie retour sur zone OB bear
            for zone in ob_bear_zones:
                if (df['high'].iloc[i] >= zone['low'] and
                        df['close'].iloc[i] <= zone['high']):
                    zone['touches'] += 1
                    if zone['touches'] == 2:
                        df.loc[df.index[i], 'mitigation_bear'] = True
            # Limite mémoire
            ob_bull_zones = ob_bull_zones[-20:]
            ob_bear_zones = ob_bear_zones[-20:]

        # ── BREAKER BLOCK ─────────────────────────────────────
        # Ancien OB bearish cassé à la hausse → devient support (breaker bull)
        # Ancien OB bullish cassé à la baisse → devient résistance (breaker bear)
        df['breaker_bull'] = False
        df['breaker_bear'] = False
        for i in range(10, len(df)):
            # Breaker bull : ancien OB bearish dans les 20 dernières bougies
            # dont le prix a cassé le high à la hausse
            for j in range(max(0, i-20), i-3):
                if df['bearish_ob'].iloc[j]:
                    ob_high = df['high'].iloc[j]
                    # Prix revient sur cet ancien OB bearish par le dessus
                    if (df['close'].iloc[i-1] > ob_high and
                            df['low'].iloc[i] <= ob_high and
                            df['close'].iloc[i] > ob_high):
                        df.loc[df.index[i], 'breaker_bull'] = True
                        break
            # Breaker bear : ancien OB bullish dont le prix a cassé le low
            for j in range(max(0, i-20), i-3):
                if df['bullish_ob'].iloc[j]:
                    ob_low = df['low'].iloc[j]
                    if (df['close'].iloc[i-1] < ob_low and
                            df['high'].iloc[i] >= ob_low and
                            df['close'].iloc[i] < ob_low):
                        df.loc[df.index[i], 'breaker_bear'] = True
                        break

        # ── PROPULSION BLOCK ──────────────────────────────────
        # OB + FVG sur la même zone = confluence ultra-haute probabilité
        # Zone où OB et FVG se chevauchent → zone institutionnelle majeure
        df['propulsion_bull'] = False
        df['propulsion_bear'] = False
        for i in range(3, len(df)):
            # Propulsion bull : OB bullish ET FVG bullish dans la même zone
            if df['bullish_ob'].iloc[i] or df['fvg_bullish'].iloc[i]:
                window = range(max(0, i-3), min(len(df), i+3))
                has_ob  = any(df['bullish_ob'].iloc[j]  for j in window)
                has_fvg = any(df['fvg_bullish'].iloc[j] for j in window)
                if has_ob and has_fvg:
                    df.loc[df.index[i], 'propulsion_bull'] = True
            # Propulsion bear
            if df['bearish_ob'].iloc[i] or df['fvg_bearish'].iloc[i]:
                window = range(max(0, i-3), min(len(df), i+3))
                has_ob  = any(df['bearish_ob'].iloc[j]  for j in window)
                has_fvg = any(df['fvg_bearish'].iloc[j] for j in window)
                if has_ob and has_fvg:
                    df.loc[df.index[i], 'propulsion_bear'] = True

        # ── REJECTION BLOCK ───────────────────────────────────
        # Wick extrême sur une zone HTF clé = rejet institutionnel confirmé
        # Différent du wick manip : doit coïncider avec OB ou EQH/EQL
        df['rejection_bull'] = False
        df['rejection_bear'] = False
        for i in range(3, len(df)):
            wick_d = df['low'].iloc[i]
            wick_u = df['high'].iloc[i]
            body_s = abs(df['close'].iloc[i] - df['open'].iloc[i])
            if body_s == 0:
                continue
            full_range = df['high'].iloc[i] - df['low'].iloc[i]
            # Rejection bull : wick bas > 60% du range ET proche EQL ou OB bull
            lower_wick = df[['open','close']].iloc[i].min() - df['low'].iloc[i]
            if lower_wick / full_range > 0.60 and df['bullish_candle'].iloc[i]:
                near_eql = any(df['eql'].iloc[max(0,i-5):i])
                near_ob  = any(df['bullish_ob'].iloc[max(0,i-5):i])
                if near_eql or near_ob:
                    df.loc[df.index[i], 'rejection_bull'] = True
            # Rejection bear : wick haut > 60% du range ET proche EQH ou OB bear
            upper_wick = df['high'].iloc[i] - df[['open','close']].iloc[i].max()
            if upper_wick / full_range > 0.60 and df['bearish_candle'].iloc[i]:
                near_eqh = any(df['eqh'].iloc[max(0,i-5):i])
                near_ob  = any(df['bearish_ob'].iloc[max(0,i-5):i])
                if near_eqh or near_ob:
                    df.loc[df.index[i], 'rejection_bear'] = True

        # ── VOLUME PROFILE simplifié (POC / VAH / VAL) ────────
        # POC  = Point of Control  — prix avec le plus de volume échangé
        # VAH  = Value Area High   — 70% du volume se situe en-dessous
        # VAL  = Value Area Low    — 70% du volume se situe au-dessus
        # Basé sur les 50 dernières bougies avec tick_volume comme proxy
        if 'tick_volume' in df.columns:
            lookback_vp = 50
            price_levels = 20   # nombre de niveaux de prix
            df['poc']       = np.nan
            df['vah']       = np.nan
            df['val']       = np.nan
            df['near_poc']  = False
            df['above_vah'] = False
            df['below_val'] = False

            for i in range(lookback_vp, len(df)):
                window   = df.iloc[i-lookback_vp:i]
                hi       = window['high'].max()
                lo       = window['low'].min()
                if hi == lo:
                    continue
                step     = (hi - lo) / price_levels
                bins     = np.arange(lo, hi + step, step)
                vol_profile = np.zeros(len(bins) - 1)

                for _, row in window.iterrows():
                    bar_lo  = row['low']
                    bar_hi  = row['high']
                    bar_vol = row['tick_volume']
                    for b in range(len(bins) - 1):
                        overlap = min(bar_hi, bins[b+1]) - max(bar_lo, bins[b])
                        if overlap > 0:
                            vol_profile[b] += bar_vol * (overlap / (bar_hi - bar_lo + 1e-10))

                if vol_profile.sum() == 0:
                    continue

                poc_idx  = np.argmax(vol_profile)
                poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2

                # Value Area : 70% du volume total
                total_vol = vol_profile.sum()
                target    = total_vol * 0.70
                va_vol    = vol_profile[poc_idx]
                va_lo_idx = poc_idx
                va_hi_idx = poc_idx

                while va_vol < target:
                    expand_hi = vol_profile[va_hi_idx+1] if va_hi_idx+1 < len(vol_profile) else 0
                    expand_lo = vol_profile[va_lo_idx-1] if va_lo_idx-1 >= 0 else 0
                    if expand_hi >= expand_lo and va_hi_idx+1 < len(vol_profile):
                        va_hi_idx += 1; va_vol += expand_hi
                    elif va_lo_idx-1 >= 0:
                        va_lo_idx -= 1; va_vol += expand_lo
                    else:
                        break

                vah_price = (bins[va_hi_idx] + bins[va_hi_idx+1]) / 2
                val_price = (bins[va_lo_idx] + bins[va_lo_idx+1]) / 2
                cur_price = df['close'].iloc[i]

                df.loc[df.index[i], 'poc'] = poc_price
                df.loc[df.index[i], 'vah'] = vah_price
                df.loc[df.index[i], 'val'] = val_price

                # Price near POC (±0.3%)
                if abs(cur_price - poc_price) / poc_price < 0.003:
                    df.loc[df.index[i], 'near_poc'] = True
                # Price above VAH = premium zone (short bias)
                if cur_price > vah_price:
                    df.loc[df.index[i], 'above_vah'] = True
                # Price below VAL = discount zone (long bias)
                if cur_price < val_price:
                    df.loc[df.index[i], 'below_val'] = True
        else:
            df['poc'] = np.nan; df['vah'] = np.nan; df['val'] = np.nan
            df['near_poc'] = False; df['above_vah'] = False; df['below_val'] = False

        # ── DIVERGENCE CACHÉE (Hidden Divergence) ─────────────
        # Confirme la CONTINUATION du trend (opposé de la div classique)
        # Hidden Bull : prix fait un higher low MAIS RSI fait un lower low
        #               → continuation haussière probable
        # Hidden Bear : prix fait un lower high MAIS RSI fait un higher high
        #               → continuation baissière probable
        df['hidden_div_bull'] = False
        df['hidden_div_bear'] = False
        for i in range(10, len(df)):
            # Hidden Bull divergence
            price_hl = df['low'].iloc[i]  > df['low'].iloc[i-10:i-3].min()
            rsi_ll   = df['RSI'].iloc[i]  < df['RSI'].iloc[i-10:i-3].min()
            if price_hl and rsi_ll and df['trend_bull'].iloc[i]:
                df.loc[df.index[i], 'hidden_div_bull'] = True
            # Hidden Bear divergence
            price_lh = df['high'].iloc[i] < df['high'].iloc[i-10:i-3].max()
            rsi_hh   = df['RSI'].iloc[i]  > df['RSI'].iloc[i-10:i-3].max()
            if price_lh and rsi_hh and df['trend_bear'].iloc[i]:
                df.loc[df.index[i], 'hidden_div_bear'] = True

        # Premium / Discount (IPDA 20 bougies)
        df['range_high']           = df['high'].rolling(20).max()
        df['range_low']            = df['low'].rolling(20).min()
        df['equilibrium']          = (df['range_high'] + df['range_low']) / 2
        df['in_premium']           = df['close'] > df['equilibrium']
        df['in_discount']          = df['close'] < df['equilibrium']
        range_size                 = df['range_high'] - df['range_low']
        df['premium_discount_pct'] = ((df['close'] - df['range_low']) / range_size * 100).fillna(50)

        # Volume
        if 'tick_volume' in df.columns:
            df['Volume_MA']    = df['tick_volume'].rolling(20).mean()
            df['Volume_surge'] = df['tick_volume'] > df['Volume_MA'] * 1.5

        # Trend
        df['trend_bull'] = ((df['EMA_9']  > df['EMA_21']) &
                            (df['EMA_21'] > df['EMA_50']) &
                            (df['EMA_50'] > df['EMA_100']))
        df['trend_bear'] = ((df['EMA_9']  < df['EMA_21']) &
                            (df['EMA_21'] < df['EMA_50']) &
                            (df['EMA_50'] < df['EMA_100']))

        # Momentum
        df['momentum_state'] = 'NORMAL'
        df.loc[df['big_candle'], 'momentum_state'] = 'HIGH_MOMENTUM'

        df.drop(['prev_close', 'tr'], axis=1, inplace=True, errors='ignore')
        self._indicator_cache[cache_key] = df
        return df

    # ── BIAIS H4 ──────────────────────────────────────────────

    def get_htf_bias(self, symbol: str) -> MarketBias:
        df = self.get_live_data(symbol, "H4", 100)
        if df is None or len(df) < 50:
            return MarketBias.NEUTRAL
        df   = self.calculate_indicators(df, f"{symbol}_H4")
        last = df.iloc[-1]
        bull = bear = 0
        if last['trend_bull']:        bull += 3
        if last['trend_bear']:        bear += 3
        if last['EMA_50'] > last['EMA_200']: bull += 2
        else:                         bear += 2
        if last['in_discount']:       bull += 1
        if last['in_premium']:        bear += 1
        if last['bos_bull']:          bull += 2
        if last['bos_bear']:          bear += 2
        if last['choch_bull']:        bull += 2
        if last['choch_bear']:        bear += 2
        if bull >= 5 and bull > bear: return MarketBias.BULLISH
        if bear >= 5 and bear > bull: return MarketBias.BEARISH
        return MarketBias.NEUTRAL

    # ── FIBONACCI ─────────────────────────────────────────────

    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        high = df['high'].tail(lookback).max()
        low  = df['low'].tail(lookback).min()
        diff = high - low
        return {
            'fib_0':   high,
            'fib_236': high - diff * 0.236,
            'fib_382': high - diff * 0.382,
            'fib_50':  high - diff * 0.500,
            'fib_618': high - diff * 0.618,
            'fib_786': high - diff * 0.786,
            'fib_100': low,
        }

    def is_near_fib_level(self, price: float, levels: Dict[str, float],
                           tolerance: float = 0.002) -> Optional[Tuple[str, float]]:
        for name, lvl in levels.items():
            if abs(price - lvl) / price <= tolerance:
                return (name, lvl)
        return None

    # ── FILTRE CORRÉLATION ────────────────────────────────────

    def is_correlated_blocked(self, symbol: str) -> bool:
        open_symbols = {pos.symbol for pos in self.active_positions.values()}
        for group in self.CORRELATED_PAIRS:
            if symbol in group:
                overlap = group & open_symbols - {symbol}
                if overlap:
                    print(f"⚠️ Corrélation bloquée : {symbol} ↔ {overlap}")
                    return True
        return False

    # ── PROTECTION DRAWDOWN ───────────────────────────────────

    def check_drawdown(self) -> bool:
        if self.trading_halted:
            return False
        account = mt5.account_info()
        if account is None:
            return False
        if self.day_start_balance <= 0:
            return True
        dd = (self.day_start_balance - account.equity) / self.day_start_balance
        if dd >= self.MAX_DAILY_DRAWDOWN:
            print(f"\n🛑 DRAWDOWN MAX : {dd*100:.1f}% — Trading stoppé aujourd'hui")
            self.trading_halted = True
            return False
        return True

    # ── DÉTECTION SIGNAUX ─────────────────────────────────────

    def detect_signals(self, symbol: str) -> List[TradeSignal]:
        signals = []

        # 1. Biais H4 — filtre principal
        htf_bias = self.get_htf_bias(symbol)
        if htf_bias == MarketBias.NEUTRAL:
            return signals

        # 2. Données H1 / M30 / M15
        df_h1  = self.get_live_data(symbol, "H1",  150)
        df_m30 = self.get_live_data(symbol, "M30", 200)
        df_m15 = self.get_live_data(symbol, "M15", 300)
        if df_h1 is None or df_m30 is None or df_m15 is None:
            return signals
        if len(df_h1) < 50 or len(df_m30) < 50 or len(df_m15) < 50:
            return signals

        df_h1  = self.calculate_indicators(df_h1,  f"{symbol}_H1")
        df_m30 = self.calculate_indicators(df_m30, f"{symbol}_M30")
        df_m15 = self.calculate_indicators(df_m15, f"{symbol}_M15")

        cur_h1  = df_h1.iloc[-1]
        cur_m30 = df_m30.iloc[-1]
        cur_m15 = df_m15.iloc[-1]

        price          = cur_m15['close']
        atr            = cur_m15['ATR']
        momentum_state = cur_m15['momentum_state']

        fib_levels = self.calculate_fibonacci_levels(df_m30, lookback=50)
        near_fib   = self.is_near_fib_level(price, fib_levels)

        support    = df_m30['low'].tail(30).min()
        resistance = df_m30['high'].tail(30).max()

        recent_sweep_low  = any(df_m15['sweep_low'].tail(5))  or any(df_m30['sweep_low'].tail(5))
        recent_sweep_high = any(df_m15['sweep_high'].tail(5)) or any(df_m30['sweep_high'].tail(5))

        # ── SCORING UNIFIÉ PAR CONFLUENCE MTF ─────────────────
        # Principe : +2 pts par TF où le signal est présent
        # Si le même signal apparaît sur H1 + M30 + M15 → +6 pts
        # La confluence naturelle fait monter le score — pas de favoritisme TF
        # ──────────────────────────────────────────────────────

        PTS = 2   # points par TF par signal — égal pour tous les TF

        def score_mtf_bull(score, reasons):
            """Scoring LONG — chaque signal checké sur chaque TF."""

            # ── BOS (Break of Structure) ──────────────────────
            bos_tfs = []
            if cur_h1['bos_bull']  or any(df_h1['bos_bull'].tail(3)):   bos_tfs.append("H1")
            if cur_m30['bos_bull'] or any(df_m30['bos_bull'].tail(3)):  bos_tfs.append("M30")
            if cur_m15['bos_bull'] or any(df_m15['bos_bull'].tail(3)):  bos_tfs.append("M15")
            if bos_tfs:
                score += PTS * len(bos_tfs)
                reasons.append(f"BOS({'|'.join(bos_tfs)})")

            # ── CHoCH (Change of Character) ───────────────────
            choch_tfs = []
            if cur_h1['choch_bull']  or any(df_h1['choch_bull'].tail(3)):  choch_tfs.append("H1")
            if cur_m30['choch_bull'] or any(df_m30['choch_bull'].tail(3)): choch_tfs.append("M30")
            if cur_m15['choch_bull'] or any(df_m15['choch_bull'].tail(3)): choch_tfs.append("M15")
            if choch_tfs:
                score += PTS * len(choch_tfs)
                reasons.append(f"CHoCH({'|'.join(choch_tfs)})")

            # ── ORDER BLOCK frais (+1 bonus si frais) ─────────
            ob_tfs = []
            ob_fresh_tfs = []
            if any(df_h1['bullish_ob_fresh'].tail(5)):  ob_fresh_tfs.append("H1")
            elif cur_h1['bullish_ob'] or any(df_h1['bullish_ob'].tail(5)):  ob_tfs.append("H1")
            if any(df_m30['bullish_ob_fresh'].tail(5)): ob_fresh_tfs.append("M30")
            elif cur_m30['bullish_ob'] or any(df_m30['bullish_ob'].tail(5)): ob_tfs.append("M30")
            if any(df_m15['bullish_ob_fresh'].tail(5)): ob_fresh_tfs.append("M15")
            elif cur_m15['bullish_ob'] or any(df_m15['bullish_ob'].tail(5)): ob_tfs.append("M15")
            if ob_fresh_tfs:
                score += (PTS + 1) * len(ob_fresh_tfs)
                reasons.append(f"OB_Fresh({'|'.join(ob_fresh_tfs)})")
            if ob_tfs:
                score += PTS * len(ob_tfs)
                reasons.append(f"OB({'|'.join(ob_tfs)})")

            # ── FVG (Fair Value Gap) ───────────────────────────
            fvg_tfs = []
            if cur_h1['fvg_bullish']  or any(df_h1['fvg_bullish'].tail(3)):  fvg_tfs.append("H1")
            if cur_m30['fvg_bullish'] or any(df_m30['fvg_bullish'].tail(3)): fvg_tfs.append("M30")
            if cur_m15['fvg_bullish'] or any(df_m15['fvg_bullish'].tail(3)): fvg_tfs.append("M15")
            if fvg_tfs:
                score += PTS * len(fvg_tfs)
                reasons.append(f"FVG({'|'.join(fvg_tfs)})")

            # ── EQUAL LOWS — pool de liquidité ────────────────
            eql_tfs = []
            if any(df_h1['eql'].tail(10)):  eql_tfs.append("H1")
            if any(df_m30['eql'].tail(5)):  eql_tfs.append("M30")
            if any(df_m15['eql'].tail(3)):  eql_tfs.append("M15")
            if eql_tfs:
                score += PTS * len(eql_tfs)
                reasons.append(f"EQL({'|'.join(eql_tfs)})")

            # ── STOP HUNT sur EQL ─────────────────────────────
            sh_tfs = []
            if any(df_h1['stophunt_bull'].tail(5)):  sh_tfs.append("H1")
            if any(df_m30['stophunt_bull'].tail(5)): sh_tfs.append("M30")
            if any(df_m15['stophunt_bull'].tail(3)): sh_tfs.append("M15")
            if sh_tfs:
                score += (PTS + 1) * len(sh_tfs)   # +3 par TF car signal fort
                reasons.append(f"StopHunt({'|'.join(sh_tfs)})")

            # ── SWEEP ─────────────────────────────────────────
            if recent_sweep_low:
                score += PTS + 1; reasons.append("Sweep")

            # ── ABSORPTION institutionnelle ────────────────────
            abs_tfs = []
            if any(df_h1['manip_absorption_bull'].tail(5)):  abs_tfs.append("H1")
            if any(df_m30['manip_absorption_bull'].tail(5)): abs_tfs.append("M30")
            if any(df_m15['manip_absorption_bull'].tail(3)): abs_tfs.append("M15")
            if abs_tfs:
                score += PTS * len(abs_tfs)
                reasons.append(f"Absorption({'|'.join(abs_tfs)})")

            # ── WICK disproportionné ───────────────────────────
            wick_tfs = []
            if any(df_h1['manip_wick_bull'].tail(5)):  wick_tfs.append("H1")
            if any(df_m30['manip_wick_bull'].tail(5)): wick_tfs.append("M30")
            if any(df_m15['manip_wick_bull'].tail(3)): wick_tfs.append("M15")
            if wick_tfs:
                score += PTS * len(wick_tfs)
                reasons.append(f"Wick({'|'.join(wick_tfs)})")

            # ── FAKE MOVE en zone Discount ─────────────────────
            fm_tfs = []
            if any(df_h1['manip_fake_move'].tail(5))  and cur_h1['in_discount']:  fm_tfs.append("H1")
            if any(df_m30['manip_fake_move'].tail(5)) and cur_m30['in_discount']: fm_tfs.append("M30")
            if any(df_m15['manip_fake_move'].tail(3)) and cur_m15['in_discount']: fm_tfs.append("M15")
            if fm_tfs:
                score += PTS * len(fm_tfs)
                reasons.append(f"FakeMove({'|'.join(fm_tfs)})")

            # ── TREND EMA alignment ────────────────────────────
            trend_tfs = []
            if cur_h1['trend_bull']:  trend_tfs.append("H1")
            if cur_m30['trend_bull']: trend_tfs.append("M30")
            if cur_m15['trend_bull']: trend_tfs.append("M15")
            if trend_tfs:
                score += PTS * len(trend_tfs)
                reasons.append(f"Trend({'|'.join(trend_tfs)})")

            # ── PREMIUM/DISCOUNT ───────────────────────────────
            disc_tfs = []
            if cur_h1['in_discount']  and cur_h1['premium_discount_pct']  < 40: disc_tfs.append("H1")
            if cur_m30['in_discount'] and cur_m30['premium_discount_pct'] < 40: disc_tfs.append("M30")
            if cur_m15['in_discount'] and cur_m15['premium_discount_pct'] < 40: disc_tfs.append("M15")
            if disc_tfs:
                score += PTS * len(disc_tfs)
                reasons.append(f"Discount({'|'.join(disc_tfs)})")

            # ── MOMENTUM — RSI / MACD / Stoch / ADX / DivRSI ─
            # Ces indicateurs sont vérifiés sur M15 (déclencheur)
            # + bonus si confirmés sur M30 aussi
            adx_tfs = []
            if cur_h1['ADX']  > 20 and cur_h1['DIplus']  > cur_h1['DIminus']:  adx_tfs.append("H1")
            if cur_m30['ADX'] > 20 and cur_m30['DIplus'] > cur_m30['DIminus']: adx_tfs.append("M30")
            if cur_m15['ADX'] > 20 and cur_m15['DIplus'] > cur_m15['DIminus']: adx_tfs.append("M15")
            if adx_tfs:
                score += PTS * len(adx_tfs)
                reasons.append(f"ADX({'|'.join(adx_tfs)})")

            rsi_tfs = []
            if 25 < cur_h1['RSI']  < 50: rsi_tfs.append("H1")
            if 25 < cur_m30['RSI'] < 50: rsi_tfs.append("M30")
            if 25 < cur_m15['RSI'] < 50: rsi_tfs.append("M15")
            if rsi_tfs:
                score += 1 * len(rsi_tfs)
                reasons.append(f"RSI({'|'.join(rsi_tfs)})")

            div_tfs = []
            if cur_h1['rsi_div_bull']:  div_tfs.append("H1")
            if cur_m30['rsi_div_bull']: div_tfs.append("M30")
            if cur_m15['rsi_div_bull']: div_tfs.append("M15")
            if div_tfs:
                score += PTS * len(div_tfs)
                reasons.append(f"DivRSI({'|'.join(div_tfs)})")

            macd_tfs = []
            if cur_h1['MACD_Hist']  > 0: macd_tfs.append("H1")
            if cur_m30['MACD_Hist'] > 0: macd_tfs.append("M30")
            if cur_m15['MACD_Hist'] > 0: macd_tfs.append("M15")
            if macd_tfs:
                score += 1 * len(macd_tfs)
                reasons.append(f"MACD+({'|'.join(macd_tfs)})")

            stoch_tfs = []
            if cur_h1['Stoch_K']  < 30 and cur_h1['Stoch_K']  > cur_h1['Stoch_D']:  stoch_tfs.append("H1")
            if cur_m30['Stoch_K'] < 30 and cur_m30['Stoch_K'] > cur_m30['Stoch_D']: stoch_tfs.append("M30")
            if cur_m15['Stoch_K'] < 30 and cur_m15['Stoch_K'] > cur_m15['Stoch_D']: stoch_tfs.append("M15")
            if stoch_tfs:
                score += 1 * len(stoch_tfs)
                reasons.append(f"Stoch({'|'.join(stoch_tfs)})")

            if 'Volume_surge' in cur_m15 and cur_m15['Volume_surge']:
                score += 1; reasons.append("Vol")


            # ── NOUVEAUX SIGNAUX V7.3 ─────────────────────────

            # MSS — Market Structure Shift (plus précis que BOS)
            mss_tfs = []
            if cur_h1['mss_bull']  or any(df_h1['mss_bull'].tail(3)):  mss_tfs.append("H1")
            if cur_m30['mss_bull'] or any(df_m30['mss_bull'].tail(3)): mss_tfs.append("M30")
            if cur_m15['mss_bull'] or any(df_m15['mss_bull'].tail(3)): mss_tfs.append("M15")
            if mss_tfs:
                score += (PTS + 1) * len(mss_tfs)
                reasons.append(f"MSS({'|'.join(mss_tfs)})")

            # Mitigation Block — 2ème toucher OB
            mit_tfs = []
            if any(df_h1['mitigation_bull'].tail(5)):  mit_tfs.append("H1")
            if any(df_m30['mitigation_bull'].tail(5)): mit_tfs.append("M30")
            if any(df_m15['mitigation_bull'].tail(3)): mit_tfs.append("M15")
            if mit_tfs:
                score += PTS * len(mit_tfs)
                reasons.append(f"Mitigation({'|'.join(mit_tfs)})")

            # Breaker Block — ancien OB bear devenu support
            brk_tfs = []
            if any(df_h1['breaker_bull'].tail(5)):  brk_tfs.append("H1")
            if any(df_m30['breaker_bull'].tail(5)): brk_tfs.append("M30")
            if any(df_m15['breaker_bull'].tail(3)): brk_tfs.append("M15")
            if brk_tfs:
                score += PTS * len(brk_tfs)
                reasons.append(f"Breaker({'|'.join(brk_tfs)})")

            # Propulsion Block — OB + FVG colocalisés
            prop_tfs = []
            if any(df_h1['propulsion_bull'].tail(5)):  prop_tfs.append("H1")
            if any(df_m30['propulsion_bull'].tail(5)): prop_tfs.append("M30")
            if any(df_m15['propulsion_bull'].tail(3)): prop_tfs.append("M15")
            if prop_tfs:
                score += (PTS + 1) * len(prop_tfs)
                reasons.append(f"Propulsion({'|'.join(prop_tfs)})")

            # Rejection Block — wick fort sur zone clé
            rej_tfs = []
            if any(df_h1['rejection_bull'].tail(5)):  rej_tfs.append("H1")
            if any(df_m30['rejection_bull'].tail(5)): rej_tfs.append("M30")
            if any(df_m15['rejection_bull'].tail(3)): rej_tfs.append("M15")
            if rej_tfs:
                score += PTS * len(rej_tfs)
                reasons.append(f"Rejection({'|'.join(rej_tfs)})")

            # Volume Profile — below VAL = zone discount profonde
            vp_tfs = []
            if cur_h1.get('below_val',  False): vp_tfs.append("H1")
            if cur_m30.get('below_val', False): vp_tfs.append("M30")
            if cur_m15.get('below_val', False): vp_tfs.append("M15")
            if vp_tfs:
                score += PTS * len(vp_tfs)
                reasons.append(f"BelowVAL({'|'.join(vp_tfs)})")
            poc_tfs = []
            if cur_h1.get('near_poc',  False): poc_tfs.append("H1")
            if cur_m30.get('near_poc', False): poc_tfs.append("M30")
            if poc_tfs:
                score += 1 * len(poc_tfs)
                reasons.append(f"NearPOC({'|'.join(poc_tfs)})")

            # Divergence cachée — continuation haussière
            hdiv_tfs = []
            if cur_h1['hidden_div_bull']:  hdiv_tfs.append("H1")
            if cur_m30['hidden_div_bull']: hdiv_tfs.append("M30")
            if cur_m15['hidden_div_bull']: hdiv_tfs.append("M15")
            if hdiv_tfs:
                score += PTS * len(hdiv_tfs)
                reasons.append(f"HiddenDiv({'|'.join(hdiv_tfs)})")

            return score, reasons

        def score_mtf_bear(score, reasons):
            """Scoring SHORT — symétrique du LONG."""

            bos_tfs = []
            if cur_h1['bos_bear']  or any(df_h1['bos_bear'].tail(3)):   bos_tfs.append("H1")
            if cur_m30['bos_bear'] or any(df_m30['bos_bear'].tail(3)):  bos_tfs.append("M30")
            if cur_m15['bos_bear'] or any(df_m15['bos_bear'].tail(3)):  bos_tfs.append("M15")
            if bos_tfs:
                score += PTS * len(bos_tfs)
                reasons.append(f"BOS({'|'.join(bos_tfs)})")

            choch_tfs = []
            if cur_h1['choch_bear']  or any(df_h1['choch_bear'].tail(3)):  choch_tfs.append("H1")
            if cur_m30['choch_bear'] or any(df_m30['choch_bear'].tail(3)): choch_tfs.append("M30")
            if cur_m15['choch_bear'] or any(df_m15['choch_bear'].tail(3)): choch_tfs.append("M15")
            if choch_tfs:
                score += PTS * len(choch_tfs)
                reasons.append(f"CHoCH({'|'.join(choch_tfs)})")

            ob_tfs = []
            ob_fresh_tfs = []
            if any(df_h1['bearish_ob_fresh'].tail(5)):  ob_fresh_tfs.append("H1")
            elif cur_h1['bearish_ob'] or any(df_h1['bearish_ob'].tail(5)):  ob_tfs.append("H1")
            if any(df_m30['bearish_ob_fresh'].tail(5)): ob_fresh_tfs.append("M30")
            elif cur_m30['bearish_ob'] or any(df_m30['bearish_ob'].tail(5)): ob_tfs.append("M30")
            if any(df_m15['bearish_ob_fresh'].tail(5)): ob_fresh_tfs.append("M15")
            elif cur_m15['bearish_ob'] or any(df_m15['bearish_ob'].tail(5)): ob_tfs.append("M15")
            if ob_fresh_tfs:
                score += (PTS + 1) * len(ob_fresh_tfs)
                reasons.append(f"OB_Fresh({'|'.join(ob_fresh_tfs)})")
            if ob_tfs:
                score += PTS * len(ob_tfs)
                reasons.append(f"OB({'|'.join(ob_tfs)})")

            fvg_tfs = []
            if cur_h1['fvg_bearish']  or any(df_h1['fvg_bearish'].tail(3)):  fvg_tfs.append("H1")
            if cur_m30['fvg_bearish'] or any(df_m30['fvg_bearish'].tail(3)): fvg_tfs.append("M30")
            if cur_m15['fvg_bearish'] or any(df_m15['fvg_bearish'].tail(3)): fvg_tfs.append("M15")
            if fvg_tfs:
                score += PTS * len(fvg_tfs)
                reasons.append(f"FVG({'|'.join(fvg_tfs)})")

            eqh_tfs = []
            if any(df_h1['eqh'].tail(10)):  eqh_tfs.append("H1")
            if any(df_m30['eqh'].tail(5)):  eqh_tfs.append("M30")
            if any(df_m15['eqh'].tail(3)):  eqh_tfs.append("M15")
            if eqh_tfs:
                score += PTS * len(eqh_tfs)
                reasons.append(f"EQH({'|'.join(eqh_tfs)})")

            sh_tfs = []
            if any(df_h1['stophunt_bear'].tail(5)):  sh_tfs.append("H1")
            if any(df_m30['stophunt_bear'].tail(5)): sh_tfs.append("M30")
            if any(df_m15['stophunt_bear'].tail(3)): sh_tfs.append("M15")
            if sh_tfs:
                score += (PTS + 1) * len(sh_tfs)
                reasons.append(f"StopHunt({'|'.join(sh_tfs)})")

            if recent_sweep_high:
                score += PTS + 1; reasons.append("Sweep")

            abs_tfs = []
            if any(df_h1['manip_absorption_bear'].tail(5)):  abs_tfs.append("H1")
            if any(df_m30['manip_absorption_bear'].tail(5)): abs_tfs.append("M30")
            if any(df_m15['manip_absorption_bear'].tail(3)): abs_tfs.append("M15")
            if abs_tfs:
                score += PTS * len(abs_tfs)
                reasons.append(f"Absorption({'|'.join(abs_tfs)})")

            wick_tfs = []
            if any(df_h1['manip_wick_bear'].tail(5)):  wick_tfs.append("H1")
            if any(df_m30['manip_wick_bear'].tail(5)): wick_tfs.append("M30")
            if any(df_m15['manip_wick_bear'].tail(3)): wick_tfs.append("M15")
            if wick_tfs:
                score += PTS * len(wick_tfs)
                reasons.append(f"Wick({'|'.join(wick_tfs)})")

            fm_tfs = []
            if any(df_h1['manip_fake_move'].tail(5))  and cur_h1['in_premium']:  fm_tfs.append("H1")
            if any(df_m30['manip_fake_move'].tail(5)) and cur_m30['in_premium']: fm_tfs.append("M30")
            if any(df_m15['manip_fake_move'].tail(3)) and cur_m15['in_premium']: fm_tfs.append("M15")
            if fm_tfs:
                score += PTS * len(fm_tfs)
                reasons.append(f"FakeMove({'|'.join(fm_tfs)})")

            trend_tfs = []
            if cur_h1['trend_bear']:  trend_tfs.append("H1")
            if cur_m30['trend_bear']: trend_tfs.append("M30")
            if cur_m15['trend_bear']: trend_tfs.append("M15")
            if trend_tfs:
                score += PTS * len(trend_tfs)
                reasons.append(f"Trend({'|'.join(trend_tfs)})")

            prem_tfs = []
            if cur_h1['in_premium']  and cur_h1['premium_discount_pct']  > 60: prem_tfs.append("H1")
            if cur_m30['in_premium'] and cur_m30['premium_discount_pct'] > 60: prem_tfs.append("M30")
            if cur_m15['in_premium'] and cur_m15['premium_discount_pct'] > 60: prem_tfs.append("M15")
            if prem_tfs:
                score += PTS * len(prem_tfs)
                reasons.append(f"Premium({'|'.join(prem_tfs)})")

            adx_tfs = []
            if cur_h1['ADX']  > 20 and cur_h1['DIminus']  > cur_h1['DIplus']:  adx_tfs.append("H1")
            if cur_m30['ADX'] > 20 and cur_m30['DIminus'] > cur_m30['DIplus']: adx_tfs.append("M30")
            if cur_m15['ADX'] > 20 and cur_m15['DIminus'] > cur_m15['DIplus']: adx_tfs.append("M15")
            if adx_tfs:
                score += PTS * len(adx_tfs)
                reasons.append(f"ADX({'|'.join(adx_tfs)})")

            rsi_tfs = []
            if 50 < cur_h1['RSI']  < 75: rsi_tfs.append("H1")
            if 50 < cur_m30['RSI'] < 75: rsi_tfs.append("M30")
            if 50 < cur_m15['RSI'] < 75: rsi_tfs.append("M15")
            if rsi_tfs:
                score += 1 * len(rsi_tfs)
                reasons.append(f"RSI({'|'.join(rsi_tfs)})")

            div_tfs = []
            if cur_h1['rsi_div_bear']:  div_tfs.append("H1")
            if cur_m30['rsi_div_bear']: div_tfs.append("M30")
            if cur_m15['rsi_div_bear']: div_tfs.append("M15")
            if div_tfs:
                score += PTS * len(div_tfs)
                reasons.append(f"DivRSI({'|'.join(div_tfs)})")

            macd_tfs = []
            if cur_h1['MACD_Hist']  < 0: macd_tfs.append("H1")
            if cur_m30['MACD_Hist'] < 0: macd_tfs.append("M30")
            if cur_m15['MACD_Hist'] < 0: macd_tfs.append("M15")
            if macd_tfs:
                score += 1 * len(macd_tfs)
                reasons.append(f"MACD-({'|'.join(macd_tfs)})")

            stoch_tfs = []
            if cur_h1['Stoch_K']  > 70 and cur_h1['Stoch_K']  < cur_h1['Stoch_D']:  stoch_tfs.append("H1")
            if cur_m30['Stoch_K'] > 70 and cur_m30['Stoch_K'] < cur_m30['Stoch_D']: stoch_tfs.append("M30")
            if cur_m15['Stoch_K'] > 70 and cur_m15['Stoch_K'] < cur_m15['Stoch_D']: stoch_tfs.append("M15")
            if stoch_tfs:
                score += 1 * len(stoch_tfs)
                reasons.append(f"Stoch({'|'.join(stoch_tfs)})")

            if 'Volume_surge' in cur_m15 and cur_m15['Volume_surge']:
                score += 1; reasons.append("Vol")


            # ── NOUVEAUX SIGNAUX V7.3 ─────────────────────────

            # MSS bear
            mss_tfs = []
            if cur_h1['mss_bear']  or any(df_h1['mss_bear'].tail(3)):  mss_tfs.append("H1")
            if cur_m30['mss_bear'] or any(df_m30['mss_bear'].tail(3)): mss_tfs.append("M30")
            if cur_m15['mss_bear'] or any(df_m15['mss_bear'].tail(3)): mss_tfs.append("M15")
            if mss_tfs:
                score += (PTS + 1) * len(mss_tfs)
                reasons.append(f"MSS({'|'.join(mss_tfs)})")

            # Mitigation Block bear
            mit_tfs = []
            if any(df_h1['mitigation_bear'].tail(5)):  mit_tfs.append("H1")
            if any(df_m30['mitigation_bear'].tail(5)): mit_tfs.append("M30")
            if any(df_m15['mitigation_bear'].tail(3)): mit_tfs.append("M15")
            if mit_tfs:
                score += PTS * len(mit_tfs)
                reasons.append(f"Mitigation({'|'.join(mit_tfs)})")

            # Breaker Block bear — ancien OB bull devenu résistance
            brk_tfs = []
            if any(df_h1['breaker_bear'].tail(5)):  brk_tfs.append("H1")
            if any(df_m30['breaker_bear'].tail(5)): brk_tfs.append("M30")
            if any(df_m15['breaker_bear'].tail(3)): brk_tfs.append("M15")
            if brk_tfs:
                score += PTS * len(brk_tfs)
                reasons.append(f"Breaker({'|'.join(brk_tfs)})")

            # Propulsion Block bear
            prop_tfs = []
            if any(df_h1['propulsion_bear'].tail(5)):  prop_tfs.append("H1")
            if any(df_m30['propulsion_bear'].tail(5)): prop_tfs.append("M30")
            if any(df_m15['propulsion_bear'].tail(3)): prop_tfs.append("M15")
            if prop_tfs:
                score += (PTS + 1) * len(prop_tfs)
                reasons.append(f"Propulsion({'|'.join(prop_tfs)})")

            # Rejection Block bear
            rej_tfs = []
            if any(df_h1['rejection_bear'].tail(5)):  rej_tfs.append("H1")
            if any(df_m30['rejection_bear'].tail(5)): rej_tfs.append("M30")
            if any(df_m15['rejection_bear'].tail(3)): rej_tfs.append("M15")
            if rej_tfs:
                score += PTS * len(rej_tfs)
                reasons.append(f"Rejection({'|'.join(rej_tfs)})")

            # Volume Profile — above VAH = zone premium
            vp_tfs = []
            if cur_h1.get('above_vah',  False): vp_tfs.append("H1")
            if cur_m30.get('above_vah', False): vp_tfs.append("M30")
            if cur_m15.get('above_vah', False): vp_tfs.append("M15")
            if vp_tfs:
                score += PTS * len(vp_tfs)
                reasons.append(f"AboveVAH({'|'.join(vp_tfs)})")
            poc_tfs = []
            if cur_h1.get('near_poc',  False): poc_tfs.append("H1")
            if cur_m30.get('near_poc', False): poc_tfs.append("M30")
            if poc_tfs:
                score += 1 * len(poc_tfs)
                reasons.append(f"NearPOC({'|'.join(poc_tfs)})")

            # Divergence cachée bear — continuation baissière
            hdiv_tfs = []
            if cur_h1['hidden_div_bear']:  hdiv_tfs.append("H1")
            if cur_m30['hidden_div_bear']: hdiv_tfs.append("M30")
            if cur_m15['hidden_div_bear']: hdiv_tfs.append("M15")
            if hdiv_tfs:
                score += PTS * len(hdiv_tfs)
                reasons.append(f"HiddenDiv({'|'.join(hdiv_tfs)})")

            return score, reasons

        # ── VALIDATION LONG ───────────────────────────────────
        if htf_bias == MarketBias.BULLISH:
            score, reasons = score_mtf_bull(0, [])
            if score >= self.MIN_SCORE and len(reasons) >= self.MIN_REASONS:
                sig = self._build_signal(
                    symbol=symbol, direction="LONG", price=price, atr=atr,
                    support=support, resistance=resistance,
                    score=score, reasons=reasons,
                    momentum_state=momentum_state, strategy=TradingStrategy.SMC_ICT,
                    htf_bias=htf_bias, has_sweep=recent_sweep_low,
                    has_bos=bool(cur_h1['bos_bull'] or cur_h1['choch_bull'] or
                                 cur_m30['bos_bull'] or cur_m30['choch_bull']),
                    cur_m5=cur_m15, cur_m15=cur_m30
                )
                if sig:
                    signals.append(sig)

        # ── VALIDATION SHORT ──────────────────────────────────
        if htf_bias == MarketBias.BEARISH:
            score, reasons = score_mtf_bear(0, [])
            if score >= self.MIN_SCORE and len(reasons) >= self.MIN_REASONS:
                sig = self._build_signal(
                    symbol=symbol, direction="SHORT", price=price, atr=atr,
                    support=support, resistance=resistance,
                    score=score, reasons=reasons,
                    momentum_state=momentum_state, strategy=TradingStrategy.SMC_ICT,
                    htf_bias=htf_bias, has_sweep=recent_sweep_high,
                    has_bos=bool(cur_h1['bos_bear'] or cur_h1['choch_bear'] or
                                 cur_m30['bos_bear'] or cur_m30['choch_bear']),
                    cur_m5=cur_m15, cur_m15=cur_m30
                )
                if sig:
                    signals.append(sig)

        # ── FIBONACCI + SMC ───────────────────────────────────
        if near_fib:
            fib_name, fib_price = near_fib
            if fib_name not in ['fib_618', 'fib_50', 'fib_382']:
                return signals

            if htf_bias == MarketBias.BULLISH and (
                    cur_m15['bullish_ob'] or any(df_m15['bullish_ob'].tail(3)) or
                    cur_m15['fvg_bullish'] or any(df_m15['fvg_bullish'].tail(2))):
                fs = 10; fr = [f"Fib_{fib_name}", "OB/FVG_M15"]
                if cur_m30['trend_bull']:   fs += 2; fr.append("TrendM30")
                if cur_m15['RSI'] < 50:      fs += 1; fr.append("RSI")
                if cur_m15['MACD_Hist'] > 0: fs += 1; fr.append("MACD+")
                if cur_m15['ADX'] > 20:      fs += 1; fr.append("ADX")
                if recent_sweep_low:        fs += 2; fr.append("Sweep")
                if fs >= self.MIN_SCORE and len(fr) >= self.MIN_REASONS:
                    sig = self._build_signal(
                        symbol=symbol, direction="LONG", price=price, atr=atr,
                        support=fib_price, resistance=resistance,
                        score=fs, reasons=fr, momentum_state=momentum_state,
                        strategy=TradingStrategy.FIBONACCI_SMC, htf_bias=htf_bias,
                        has_sweep=recent_sweep_low, has_bos=False,
                        cur_m5=cur_m15, cur_m15=cur_m30, fib_level=fib_price
                    )
                    if sig: signals.append(sig)

            if htf_bias == MarketBias.BEARISH and (
                    cur_m15['bearish_ob'] or any(df_m15['bearish_ob'].tail(3)) or
                    cur_m15['fvg_bearish'] or any(df_m15['fvg_bearish'].tail(2))):
                fs = 10; fr = [f"Fib_{fib_name}", "OB/FVG_M15"]
                if cur_m30['trend_bear']:   fs += 2; fr.append("TrendM30")
                if cur_m15['RSI'] > 50:      fs += 1; fr.append("RSI")
                if cur_m15['MACD_Hist'] < 0: fs += 1; fr.append("MACD-")
                if cur_m15['ADX'] > 20:      fs += 1; fr.append("ADX")
                if recent_sweep_high:       fs += 2; fr.append("Sweep")
                if fs >= self.MIN_SCORE and len(fr) >= self.MIN_REASONS:
                    sig = self._build_signal(
                        symbol=symbol, direction="SHORT", price=price, atr=atr,
                        support=support, resistance=fib_price,
                        score=fs, reasons=fr, momentum_state=momentum_state,
                        strategy=TradingStrategy.FIBONACCI_SMC, htf_bias=htf_bias,
                        has_sweep=recent_sweep_high, has_bos=False,
                        cur_m5=cur_m15, cur_m15=cur_m30, fib_level=fib_price
                    )
                    if sig: signals.append(sig)

        return signals

    # ── BUILDER SIGNAL (unifié — zéro doublon) ───────────────

    def _build_signal(self, symbol, direction, price, atr,
                      support, resistance, score, reasons,
                      momentum_state, strategy, htf_bias,
                      has_sweep, has_bos, cur_m5=None, cur_m15=None,
                      fib_level=None) -> Optional[TradeSignal]:

        if self.use_momentum_filter and momentum_state != 'HIGH_MOMENTUM':
            return None

        # SL sur structure réelle (sous OB / FVG), pas ATR fixe
        if direction == "LONG":
            sl = support - (atr * 0.5)
        else:
            sl = resistance + (atr * 0.5)

        risk = abs(price - sl)
        if risk == 0:
            return None

        # TP adaptatif (RR minimum 1.5)
        tp_mult = 3.0
        tp = price + risk * tp_mult if direction == "LONG" else price - risk * tp_mult
        if abs(price - tp) / risk < 1.5:
            return None

        confidence = min(score * 5.5, 100)
        if confidence < self.MIN_CONFIDENCE:
            return None

        label     = "SMC" if strategy == TradingStrategy.SMC_ICT else "FIB"
        sweep_tag = "[SWEEP]" if has_sweep else ""
        bos_tag   = "[BOS]"   if has_bos   else ""

        return TradeSignal(
            symbol=symbol, timeframe="M15", direction=direction,
            order_type="MARKET", entry=price, sl=sl, tp=tp,
            confidence=confidence, strength=score,
            timestamp=datetime.now(),
            reason=f"{label}{score}{sweep_tag}{bos_tag}: {', '.join(reasons[:6])}",
            trend_aligned=(
                (direction == "LONG"  and htf_bias == MarketBias.BULLISH) or
                (direction == "SHORT" and htf_bias == MarketBias.BEARISH)
            ),
            strategy=strategy, momentum_state=momentum_state,
            num_reasons=len(reasons), fib_level=fib_level,
            htf_bias=htf_bias, has_sweep=has_sweep, has_bos=has_bos
        )

    # ── LOT DYNAMIQUE ─────────────────────────────────────────

    def calculate_dynamic_lot(self, specs: SymbolSpecs, signal: TradeSignal,
                               entry: float, sl: float) -> float:
        risk_amount   = self.capital * self.risk_per_trade
        risk_distance = abs(entry - sl)
        if risk_distance == 0:
            return specs.volume_min
        base_lot = risk_amount / (risk_distance * (specs.tick_value / specs.tick_size))
        base_lot = max(specs.volume_min, min(base_lot, specs.volume_max))
        mult = 1.0
        if signal.strength >= 18:   mult = 1.30
        elif signal.strength >= 16: mult = 1.20
        elif signal.strength >= 14: mult = 1.10
        if signal.trend_aligned:    mult *= 1.10
        if signal.has_sweep:        mult *= 1.10
        if signal.has_bos:          mult *= 1.05
        if signal.confidence >= 85: mult *= 1.10
        if signal.strategy == TradingStrategy.FIBONACCI_SMC: mult *= 1.05
        lot = base_lot * mult
        lot = max(specs.volume_min, min(lot, specs.volume_max))
        lot = round(lot / specs.volume_step) * specs.volume_step
        return round(lot, 2)

    # ── EXÉCUTION TRADE ───────────────────────────────────────

    def execute_trade(self, signal: TradeSignal) -> Optional[int]:
        if signal.confidence  < self.MIN_CONFIDENCE:  return None
        if signal.strength    < self.MIN_SCORE:        return None
        if signal.num_reasons < self.MIN_REASONS:      return None
        if len(self.active_positions) >= self.max_positions: return None
        if self.is_correlated_blocked(signal.symbol):  return None
        if not self.check_drawdown():                  return None

        specs = self.get_symbol_specs(signal.symbol)
        if specs is None: return None

        entry = round(signal.entry, specs.digits)
        sl    = round(signal.sl,    specs.digits)

        total_lot = self.calculate_dynamic_lot(specs, signal, entry, sl)
        if total_lot < specs.volume_min: return None

        risk_distance = abs(entry - sl)
        tp_levels = []
        for ratio in self.TP_RATIOS:
            tp = entry + risk_distance * ratio if signal.direction == "LONG" \
                 else entry - risk_distance * ratio
            tp_levels.append(round(tp, specs.digits))

        num_tp     = len(self.TP_RATIOS)
        lot_per_tp = round(total_lot / num_tp / specs.volume_step) * specs.volume_step
        if lot_per_tp < specs.volume_min:
            lot_per_tp = specs.volume_min
        lots = [lot_per_tp] * num_tp

        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "LONG" else mt5.ORDER_TYPE_SELL

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       signal.symbol,
            "volume":       lots[0],
            "type":         order_type,
            "price":        entry,
            "sl":           sl,
            "tp":           tp_levels[0],
            "deviation":    20,
            "magic":        999777,
            "comment":      f"NOVA_V7_{signal.strategy.value}_S{signal.strength}",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Ordre échoué {signal.symbol}: retcode={getattr(result,'retcode','None')}")
            return None

        position = ManagedPosition(
            ticket=result.order, symbol=signal.symbol,
            direction=signal.direction, volume=lots[0],
            entry_price=entry, sl=sl, tp=tp_levels[-1],
            open_time=datetime.now(),
            tp_levels=tp_levels, tp_hit=[False]*len(tp_levels),
            strategy=signal.strategy, last_ob_sl=sl
        )
        self.active_positions[result.order] = position
        self.total_trades += 1
        self.stats.trades_today += 1

        emoji = "📐" if signal.strategy == TradingStrategy.FIBONACCI_SMC else "⚔️"
        sweep = " 🌊" if signal.has_sweep else ""
        bos   = " 🔄" if signal.has_bos   else ""

        print(f"\n✅ TRADE #{result.order} | {signal.symbol} {signal.direction} | H4:{signal.htf_bias.value}")
        print(f"   {emoji}{sweep}{bos} {signal.strategy.value} | Lot:{lots[0]} | Score:{signal.strength} | Conf:{signal.confidence:.0f}%")
        print(f"   Entry:{entry} | SL:{sl} | TP1:{tp_levels[0]} | Raisons:{signal.num_reasons}")
        print(f"   {signal.reason}")
        if signal.fib_level:
            print(f"   📐 Fibo: {signal.fib_level}")

        self._log_trade(signal, lots[0])
        time.sleep(0.5)

        for i in range(1, len(lots)):
            if lots[i] >= specs.volume_min:
                req = request.copy()
                req['volume']  = lots[i]
                req['tp']      = tp_levels[i]
                req['comment'] = f"NOVA_TP{i+1}_1:{self.TP_RATIOS[i]}"
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    pos2 = ManagedPosition(
                        ticket=res.order, symbol=signal.symbol,
                        direction=signal.direction, volume=lots[i],
                        entry_price=entry, sl=sl, tp=tp_levels[-1],
                        open_time=datetime.now(),
                        tp_levels=tp_levels, tp_hit=[False]*len(tp_levels),
                        strategy=signal.strategy, last_ob_sl=sl
                    )
                    self.active_positions[res.order] = pos2
                    print(f"   + #{res.order} TP{i+1} 1:{self.TP_RATIOS[i]} Lot:{lots[i]}")
                time.sleep(0.5)

        print(f"{'─'*80}\n")
        return result.order

    # ── GESTION POSITIONS — BE + TRAIL SUR OB ────────────────

    def manage_positions(self):
        mt5_positions = mt5.positions_get(magic=999777)
        if not mt5_positions:
            self.active_positions.clear()
            return

        active_tickets = {p.ticket for p in mt5_positions}

        for ticket, pos in list(self.active_positions.items()):
            if ticket not in active_tickets:
                del self.active_positions[ticket]
                continue
            mt5_pos = next((p for p in mt5_positions if p.ticket == ticket), None)
            if mt5_pos is None:
                continue

            current_price = mt5_pos.price_current
            specs = self.get_symbol_specs(pos.symbol)
            if specs is None:
                continue

            profit_pips = ((current_price - pos.entry_price) / specs.pip_value
                           if pos.direction == "LONG"
                           else (pos.entry_price - current_price) / specs.pip_value)

            tp1      = pos.tp_levels[0] if pos.tp_levels else pos.tp
            tp_dist  = abs(tp1 - pos.entry_price) / specs.pip_value

            # Breakeven à 70% du TP1
            if not pos.breakeven_set and profit_pips >= tp_dist * 0.7:
                new_sl = (pos.entry_price + 3 * specs.pip_value if pos.direction == "LONG"
                          else pos.entry_price - 3 * specs.pip_value)
                new_sl = round(new_sl, specs.digits)
                req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol,
                       "position": ticket, "sl": new_sl, "tp": mt5_pos.tp, "magic": 999777}
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    pos.sl = new_sl; pos.breakeven_set = True
                    print(f"🔒 BE #{ticket} → SL: {new_sl}")

            # Trail sur OB — activé à 80% du TP1
            if not pos.trailing_active and profit_pips >= tp_dist * 0.8:
                pos.trailing_active = True
                print(f"📈 Trail OB activé #{ticket}")

            if pos.trailing_active:
                df_m15 = self.get_live_data(pos.symbol, "M15", 50)
                if df_m15 is not None:
                    df_m15 = self.calculate_indicators(df_m15, f"{pos.symbol}_M15_trail")
                    if pos.direction == "LONG":
                        ob_rows = df_m15[df_m15['bullish_ob']].tail(3)
                        if not ob_rows.empty:
                            ob_sl = round(ob_rows['low'].min() - specs.pip_value, specs.digits)
                            if ob_sl > pos.sl:
                                req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol,
                                       "position": ticket, "sl": ob_sl, "tp": mt5_pos.tp, "magic": 999777}
                                res = mt5.order_send(req)
                                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                    pos.sl = ob_sl; pos.last_ob_sl = ob_sl
                    else:
                        ob_rows = df_m15[df_m15['bearish_ob']].tail(3)
                        if not ob_rows.empty:
                            ob_sl = round(ob_rows['high'].max() + specs.pip_value, specs.digits)
                            if ob_sl < pos.sl or pos.sl == pos.entry_price:
                                req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": pos.symbol,
                                       "position": ticket, "sl": ob_sl, "tp": mt5_pos.tp, "magic": 999777}
                                res = mt5.order_send(req)
                                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                    pos.sl = ob_sl; pos.last_ob_sl = ob_sl

    # ── BOUCLE DE TRADING ─────────────────────────────────────

    def _trading_loop(self):
        scan_count = 0
        while self.monitoring:
            try:
                if not self.ensure_connected():
                    time.sleep(30); continue
                if not self.check_drawdown():
                    time.sleep(300); continue

                self.manage_positions()

                for symbol in self.AUTHORIZED_SYMBOLS:
                    if len(self.active_positions) >= self.max_positions:
                        break
                    try:
                        signals = self.detect_signals(symbol)
                        for sig in signals:
                            if len(self.active_positions) < self.max_positions:
                                self.execute_trade(sig)
                                time.sleep(2)
                    except Exception as e:
                        print(f"⚠️ {symbol}: {e}"); continue

                scan_count += 1
                if scan_count % 20 == 0:
                    self.print_status()
                if scan_count % 100 == 0:
                    self._indicator_cache.clear()

                time.sleep(30)

            except Exception as e:
                print(f"⚠️ Boucle: {e}")
                time.sleep(30)

    def start_auto_trading(self):
        self.monitoring        = True
        self.monitoring_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.monitoring_thread.start()
        tp_ratios = ", ".join([f"1:{r}" for r in self.TP_RATIOS])
        print(f"\n{'⚔️ NOVA IA V7.0 ACTIVÉE':^100}")
        print(f"{'─'*100}")
        print(f"║ CASCADE H4→H1→M30→M15 | TP: {tp_ratios} | DD MAX: 5%{' '*40} ║")
        print(f"║ BOS/CHoCH | Sweep | OB Fresh | Trail OB | Corrélation | Log CSV{' '*26} ║")
        print(f"║ EQH/EQL | StopHunt | Absorption | Wick Manip | FakeMove{' '*34} ║")
        print(f"{'─'*100}\n")

    def stop_auto_trading(self):
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("⏸️ Trading arrêté")

    # ── UTILITAIRES ───────────────────────────────────────────

    def close_all_positions(self):
        tickets = list(self.active_positions.keys())
        if not tickets:
            print("ℹ️ Aucune position\n"); return
        closed = 0
        for ticket in tickets:
            pos = self.active_positions[ticket]
            close_type = mt5.ORDER_TYPE_SELL if pos.direction == "LONG" else mt5.ORDER_TYPE_BUY
            req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol,
                   "volume": pos.volume, "type": close_type, "position": ticket,
                   "deviation": 20, "magic": 999777, "comment": "CloseAll",
                   "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1; time.sleep(0.5)
        print(f"✅ {closed}/{len(tickets)} positions fermées\n")

    def list_positions(self):
        if not self.active_positions:
            print("ℹ️ Aucune position active\n"); return
        mt5_pos = mt5.positions_get(magic=999777) or []
        total   = 0
        print(f"\n{'─'*100}\n{'POSITIONS ACTIVES V7.0':^100}\n{'─'*100}")
        for ticket, pos in self.active_positions.items():
            mp = next((p for p in mt5_pos if p.ticket == ticket), None)
            if mp:
                pnl = mp.profit; total += pnl
                st  = "🟢" if pnl > 0 else "🔴"
                be  = "🔒" if pos.breakeven_set  else "⏳"
                tr  = "📈" if pos.trailing_active else "⏳"
                em  = "📐" if pos.strategy == TradingStrategy.FIBONACCI_SMC else "⚔️"
                print(f"║ {st} #{ticket} │ {pos.symbol} {pos.direction} │ {em} ${pnl:+.2f}")
                print(f"║   Lot:{pos.volume} │ BE:{be} │ Trail:{tr} │ SL:{pos.sl}")
                print(f"{'─'*100}")
        print(f"║ TOTAL P&L: ${total:+.2f}\n")

    def print_status(self):
        account = mt5.account_info()
        if account is None: return
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        dd = ((self.day_start_balance - account.equity) / self.day_start_balance * 100
              if self.day_start_balance > 0 else 0)
        print(f"\n{'─'*100}")
        print(f"📊 {datetime.now().strftime('%H:%M:%S')} | Balance:${account.balance:,.2f} | Equity:${account.equity:,.2f} | DD:{dd:.1f}%")
        print(f"   Pos:{len(self.active_positions)}/{self.max_positions} | Trades:{self.total_trades} | WR:{wr:.1f}% | PnL:${self.total_pnl:.2f}")
        print(f"{'─'*100}\n")

    def generate_report(self):
        account = mt5.account_info()
        if account is None: return
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        dd = ((self.day_start_balance - account.equity) / self.day_start_balance * 100
              if self.day_start_balance > 0 else 0)
        print(f"\n{'='*100}\n{'📊 RAPPORT NOVA IA V7.0':^100}\n{'='*100}")
        print(f"Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
        print(f"Trades: {self.total_trades} | WR: {wr:.1f}% | PnL: ${self.total_pnl:.2f}")
        print(f"Drawdown jour: {dd:.2f}% | DD MAX: {self.MAX_DAILY_DRAWDOWN*100:.0f}%")
        print(f"Cascade: H4→H1→M30→M15 | Score≥{self.MIN_SCORE} | Conf≥{self.MIN_CONFIDENCE}%")
        print(f"Log CSV: {self.log_file}")
        print(f"{'='*100}\n")

    def shutdown(self):
        print(f"\n{'🛑 ARRÊT NOVA IA V7.0':^100}")
        self.stop_auto_trading()
        self.generate_report()
        if self.active_positions:
            r = input(f"⚠️ {len(self.active_positions)} position(s). Fermer tout ? (y/n): ")
            if r.lower() == 'y':
                self.close_all_positions()
        mt5.shutdown()
        print("✅ Système arrêté proprement\n")

    def __del__(self):
        try:
            self.stop_auto_trading()
            mt5.shutdown()
        except:
            pass


# ============================================================
#  MENUS
# ============================================================

def interactive_menu():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║           NOVA IA V7.0 — H4→H1→M30→M15 — MENU                            ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║  1. Démarrer le trading automatique                                       ║
    ║  2. Arrêter le trading                                                    ║
    ║  3. Voir les positions actives                                            ║
    ║  4. Fermer toutes les positions                                           ║
    ║  5. Générer un rapport                                                    ║
    ║  6. Afficher le status                                                    ║
    ║  0. Quitter                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    print(f"\n{'='*100}")
    print(f"{'🔐 NOVA IA V7.0 — XAUUSD | DE40 | US30 | USTEC | US500':^100}")
    print(f"{'='*100}\n")

    use_cred = input("Identifiants MT5 personnalisés ? (y/n): ").strip().lower()
    mt5_login = mt5_password = mt5_server = None
    if use_cred == 'y':
        try:
            mt5_login    = int(input("Login MT5: ").strip())
            mt5_password = input("Password MT5: ").strip()
            mt5_server   = input("Server MT5: ").strip()
        except:
            print("⚠️ Login invalide, compte par défaut")
            mt5_login = None

    cap  = input("\n💰 Capital initial (défaut 10000): ").strip()
    CAPITAL = float(cap) if cap else 10000
    risk = input("📊 Risque par trade en % (défaut 2): ").strip()
    RISK = float(risk) / 100 if risk else 0.02
    mom  = input("⭐ Filtre MOMENTUM ? (y/n, défaut n): ").strip().lower()
    USE_MOM = mom == 'y'

    try:
        trader      = AutoTrader(capital=CAPITAL, risk_per_trade=RISK,
                                 mt5_login=mt5_login, mt5_password=mt5_password,
                                 mt5_server=mt5_server, use_momentum_filter=USE_MOM)
        auto_running = False
        while True:
            interactive_menu()
            choice = input("Votre choix: ").strip()
            if   choice == "1":
                if not auto_running: trader.start_auto_trading(); auto_running = True
                else: print("⚠️ Déjà actif\n")
            elif choice == "2":
                if auto_running: trader.stop_auto_trading(); auto_running = False
                else: print("⚠️ Non actif\n")
            elif choice == "3": trader.list_positions()
            elif choice == "4":
                if input("Confirmer ? (y/n): ").lower() == 'y': trader.close_all_positions()
            elif choice == "5": trader.generate_report()
            elif choice == "6": trader.print_status()
            elif choice == "0": break
            else: print("❌ Choix invalide\n")
    except KeyboardInterrupt:
        print("\n⏸️ Ctrl+C")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
    finally:
        try: trader.shutdown()
        except: pass


def auto_mode():
    print(f"\n{'='*100}\n{'🔐 NOVA IA V7.0 — MODE AUTOMATIQUE':^100}\n{'='*100}\n")
    use_cred = input("Identifiants MT5 ? (y/n): ").strip().lower()
    mt5_login = mt5_password = mt5_server = None
    if use_cred == 'y':
        try:
            mt5_login    = int(input("Login: ").strip())
            mt5_password = input("Password: ").strip()
            mt5_server   = input("Server: ").strip()
        except: mt5_login = None
    cap  = input("\n💰 Capital (défaut 10000): ").strip()
    CAPITAL = float(cap) if cap else 10000
    risk = input("📊 Risque % (défaut 2): ").strip()
    RISK = float(risk) / 100 if risk else 0.02
    mom  = input("⭐ Filtre MOMENTUM ? (y/n, défaut n): ").strip().lower()
    USE_MOM = mom == 'y'
    try:
        trader = AutoTrader(capital=CAPITAL, risk_per_trade=RISK,
                            mt5_login=mt5_login, mt5_password=mt5_password,
                            mt5_server=mt5_server, use_momentum_filter=USE_MOM)
        trader.start_auto_trading()
        print(f"\n🔄 Mode automatique lancé — Ctrl+C pour arrêter\n")
        while True:
            time.sleep(300)
            trader.generate_report()
    except KeyboardInterrupt:
        print("\n⏸️ Arrêt...")
    finally:
        try: trader.shutdown()
        except: pass


# ============================================================
#  POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║      NOVA IA V7.0 — SMC/ICT + FIBONACCI — SÉLECTION MODE                 ║
    ╠═══════════════════════════════════════════════════════════════════════════╣
    ║  1. Mode Interactif (avec menu de contrôle)                               ║
    ║  2. Mode 100% Automatique (sans interaction)                              ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    mode = input("Choisissez votre mode (1 ou 2): ").strip()
    if   mode == "1": main()
    elif mode == "2": auto_mode()
    else: print("❌ Choix invalide. Redémarrez.")
