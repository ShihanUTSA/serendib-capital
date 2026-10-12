#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statarb_ver2.py

Statistical Arbitrage Bot — LOG-RETURN variant.

Difference vs. statarb_ver1.py
------------------------------
ver1 arbitrages on RAW PRICE:
    beta   = OLS of price_A on price_B          (price-level hedge ratio)
    spread = price_A - beta * price_B - const   (price-level spread, in $)

ver2 (this file) arbitrages on LOG RETURNS:
    beta   = OLS of log-returns_A on log-returns_B   (return sensitivity)
    spread = log(A) - beta * log(B)                  (spread on LOG PRICES)

Why the spread is on log prices, not price levels:
    If  r_A = beta * r_B  (r = diff(log price)), then integrating gives
    log(A) ~= beta * log(B) + const. So a beta estimated from log returns
    is the coefficient that belongs on LOG PRICES. Pairing a return-beta
    with a price-level spread (A - beta*B) mixes units and is inconsistent.

Note on the return intercept (return_alpha):
    In the log-return regression, the intercept is a PER-PERIOD drift, not a
    level offset. It is therefore NOT subtracted as a flat constant from the
    log-price spread (that would be a unit mismatch). The rolling-mean z-score
    already absorbs slow drift. To strip drift explicitly you would remove it
    cumulatively (return_alpha * t) — left out here by design.

Caveat for interviews:
    The return-based beta is not guaranteed to equal the Engle-Granger
    cointegrating coefficient from regressing log(A) on log(B) directly.
    They are usually close for tight ETF pairs; if the ADF test fails on the
    log-price spread while returns look highly correlated, that gap is the
    first place to look.

Created on Wed Feb 11 2026
@author: shihan
"""

import asyncio
import os
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

try:
    from ib_insync import IB, Stock, util
except ImportError:  # pragma: no cover - optional dependency
    IB = None
    Stock = None
    util = None


class StatArbBot:
    """
    A Statistical Arbitrage Bot using a LOG-RETURN beta and a LOG-PRICE spread.
    Designed for live-simulation and signal generation.
    """

    def __init__(self, ticker_a, ticker_b, lookback_window=60, z_threshold=2.0,
                 data_source="auto", ib_host=None, ib_port=7496, ib_client_id=1):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.window = lookback_window
        self.z_threshold = z_threshold
        self.data_source = data_source
        self.ib_host = ib_host or os.getenv("IB_HOST", "127.0.0.1")
        self.ib_port = int(ib_port or os.getenv("IB_PORT", 7496))
        self.ib_client_id = int(ib_client_id or os.getenv("IB_CLIENT_ID", 1))
        self.data = pd.DataFrame()
        self.hedge_ratio = None      # beta on log returns
        self.return_alpha = None     # per-period drift differential (return intercept)
        self.results = None
        self.ib = None

    def _load_local_close_series(self, ticker):
        """Attempt to load a ticker's close series from local data files in the repo."""
        data_dir = Path(__file__).resolve().parents[1] / "data"
        if not data_dir.exists():
            return None

        candidates = []
        for path in data_dir.iterdir():
            if not path.is_file():
                continue
            if ticker.lower() not in path.name.lower():
                continue
            if path.suffix.lower() not in {".xlsx", ".csv"}:
                continue
            candidates.append(path)

        if not candidates:
            return None

        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:3]

        frames = []
        for path in candidates:
            try:
                if path.suffix.lower() == ".xlsx":
                    df = pd.read_excel(path, usecols=["StartTime", "ClosePrice"])
                else:
                    df = pd.read_csv(path, usecols=["StartTime", "ClosePrice"], parse_dates=["StartTime"])
            except Exception:
                continue

            if "StartTime" not in df.columns or "ClosePrice" not in df.columns:
                continue

            df = df[["StartTime", "ClosePrice"]].copy()
            df["StartTime"] = pd.to_datetime(df["StartTime"])
            df = df.set_index("StartTime").sort_index()
            frames.append(df)

        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined["ClosePrice"].resample("1D").last().dropna()

    def _connect_ib(self):
        if self.ib is not None and self.ib.isConnected():
            return self.ib
        if IB is None or Stock is None or util is None:
            raise RuntimeError("ib-insync is not installed. Install it with `pip install ib-insync`.")

        # Notebook / IPython environments often already have an event loop running.
        # Start it in a way that ib-insync expects so the connection call does not
        # trip over "This event loop is already running".
        try:
            util.startLoop()
        except Exception:
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except Exception:
                pass

        last_exc = None
        candidate_ports = [self.ib_port, 7496, 7497, 4001, 4002, 4003]
        candidate_ports = list(dict.fromkeys(candidate_ports))
        candidate_client_ids = [
            self.ib_client_id,
            self.ib_client_id + 1,
            self.ib_client_id + 2,
            self.ib_client_id + 10,
            self.ib_client_id + 100,
            self.ib_client_id + 1000,
        ]
        candidate_client_ids = list(dict.fromkeys(candidate_client_ids))

        for port in candidate_ports:
            for client_id in candidate_client_ids:
                # Explicitly clean up any previous IB connection before trying the next
                # client ID/port combination.
                if self.ib is not None:
                    try:
                        self.ib.disconnect()
                    except Exception:
                        pass
                self.ib = IB()
                try:
                    self.ib.connect(
                        self.ib_host,
                        port,
                        clientId=client_id,
                        readonly=True,
                        timeout=10,
                    )
                    return self.ib
                except Exception as exc:
                    last_exc = exc

                    # ib_insync surfaces a client-ID clash as a bare TimeoutError with an
                    # empty message, so retry on the exception type as well as the text.
                    retryable_types = (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)
                    message = str(exc).lower()
                    conflict_indicators = (
                        "already in use",
                        "client id",
                        "clientid",
                        "peer closed connection",
                        "api connection failed",
                        "timeout",
                        "connection refused",
                        "cannot connect",
                    )
                    if isinstance(exc, retryable_types) or any(
                        indicator in message for indicator in conflict_indicators
                    ):
                        continue
                    raise

        raise RuntimeError(
            f"Unable to connect to Interactive Brokers at {self.ib_host} on ports {candidate_ports} "
            f"with client IDs {candidate_client_ids}: {last_exc}"
        ) from last_exc

    def _period_to_ib_duration(self, period):
        if isinstance(period, str):
            period = period.lower()
            if period.startswith("1y") or period.startswith("365"):
                return "1 Y"
            if period.startswith("6m"):
                return "6 M"
            if period.startswith("3m"):
                return "3 M"
            if period.startswith("1m"):
                return "1 M"
            if period.startswith("2w"):
                return "2 W"
            if period.startswith("1w"):
                return "1 W"
            if period.startswith("1d"):
                return "1 D"
        return "1 Y"

    def _interval_to_ib_bar_size(self, interval):
        if isinstance(interval, str):
            interval = interval.lower()
            if interval.startswith("1m"):
                return "1 min"
            if interval.startswith("5m"):
                return "5 mins"
            if interval.startswith("15m"):
                return "15 mins"
            if interval.startswith("1h"):
                return "1 hour"
            if interval.startswith("1d"):
                return "1 day"
            if interval.startswith("1wk"):
                return "1 week"
        return "1 day"

    def _download_close_series_ib(self, ticker, period, interval):
        ib = self._connect_ib()
        contract = Stock(ticker, "SMART", "USD")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=self._period_to_ib_duration(period),
            barSizeSetting=self._interval_to_ib_bar_size(interval),
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            raise ValueError(f"Interactive Brokers returned no bars for {ticker}")

        df = util.df(bars)
        if df.empty:
            raise ValueError(f"Interactive Brokers returned an empty dataset for {ticker}")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
        else:
            raise ValueError(f"Interactive Brokers returned an unexpected bar format for {ticker}")

        if "close" in df.columns:
            return df["close"].astype(float).rename(ticker)
        if "Close" in df.columns:
            return df["Close"].astype(float).rename(ticker)

        raise ValueError(f"Interactive Brokers bars did not include close prices for {ticker}")

    def _download_close_series_yf(self, ticker, period, interval):
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )

        if isinstance(df, pd.Series):
            if df.empty:
                raise ValueError(f"No close-price data returned for {ticker}")
            return df
        if isinstance(df, pd.DataFrame):
            if df.empty:
                raise ValueError(f"No close-price data returned for {ticker}")
            if "Close" in df.columns:
                return df["Close"]
            if "Adj Close" in df.columns:
                return df["Adj Close"]

        raise ValueError(f"No close-price data returned for {ticker}")

    def _download_close_series(self, ticker, period, interval):
        if self.data_source == "ib":
            try:
                print(f"Attempting to load {ticker} from Interactive Brokers...")
                return self._download_close_series_ib(ticker, period, interval)
            except Exception as exc:
                raise RuntimeError(f"Unable to load data for {ticker} from Interactive Brokers: {exc}") from exc

        if self.data_source == "yfinance":
            try:
                return self._download_close_series_yf(ticker, period, interval)
            except Exception as exc:
                raise RuntimeError(f"Unable to load data for {ticker} from Yahoo Finance: {exc}") from exc

        # Auto mode: try Interactive Brokers first, then Yahoo Finance, then local files.
        try:
            return self._download_close_series_ib(ticker, period, interval)
        except Exception as exc:
            print(f"Interactive Brokers failed for {ticker}: {exc}")

        try:
            return self._download_close_series_yf(ticker, period, interval)
        except Exception as exc:
            print(f"Yahoo Finance fetch failed for {ticker}: {exc}")

        local_series = self._load_local_close_series(ticker)
        if local_series is not None:
            print(f"Using local data for {ticker}")
            return local_series

        raise RuntimeError(f"Unable to load price data for {ticker}")

    def fetch_data(self, period="1y", interval="1d"):
        """
        Fetches historical data for calibration.
        In a production environment, this would connect to a websocket (e.g., IBKR, Alpaca).
        """
        print(f"Fetching data for {self.ticker_a} and {self.ticker_b}...")

        df_a = self._download_close_series(self.ticker_a, period, interval)
        df_b = self._download_close_series(self.ticker_b, period, interval)

        # Merge and clean data
        self.data = pd.concat([df_a, df_b], axis=1)
        self.data.columns = [self.ticker_a, self.ticker_b]
        self.data.dropna(inplace=True)
        if self.data.empty:
            raise ValueError("No usable price data was loaded for the selected pair.")
        return self.data

    def calculate_beta_log_returns(self):
        """
        Estimate the hedge ratio (beta) by regressing the LOG RETURNS of
        ticker_a on the log returns of ticker_b:

            r_A = alpha + beta * r_B + eps,   where r = diff(log(price))

        Beta is the return sensitivity of A to B; it is applied to LOG PRICES
        when forming the spread (see _log_price_spread). This is the ver2
        replacement for ver1's price-level calculate_hedge_ratio().
        """
        if self.data.empty or self.ticker_a not in self.data.columns \
                or self.ticker_b not in self.data.columns:
            raise ValueError("Price data is not loaded yet or is missing the required columns.")

        log_prices = np.log(self.data[[self.ticker_a, self.ticker_b]])
        log_returns = log_prices.diff().dropna()

        if len(log_returns) < 2:
            raise ValueError("Not enough overlapping data to estimate beta from log returns.")

        X = sm.add_constant(log_returns[self.ticker_b])
        model = sm.OLS(log_returns[self.ticker_a], X)
        results = model.fit()

        self.hedge_ratio = results.params[self.ticker_b]   # beta on log returns
        self.return_alpha = results.params["const"]        # per-period drift differential
        self.results = results

        corr = log_returns[self.ticker_a].corr(log_returns[self.ticker_b])
        print(f"Beta (log-return hedge ratio): {self.hedge_ratio:.4f}")
        print(f"Return alpha (per-period drift): {self.return_alpha:.6f}")
        print(f"Return correlation (Pearson):  {corr:.4f}")
        return self.hedge_ratio, self.results

    def _log_price_spread(self):
        """Spread on LOG PRICES using the log-return beta: log(A) - beta * log(B)."""
        if self.hedge_ratio is None:
            raise ValueError("Beta is not calibrated yet. Call calculate_beta_log_returns() first.")
        log_a = np.log(self.data[self.ticker_a])
        log_b = np.log(self.data[self.ticker_b])
        return log_a - self.hedge_ratio * log_b

    def check_cointegration(self):
        """
        Performs the Augmented Dickey-Fuller (ADF) test on the LOG-PRICE spread
        to verify mean reversion. Correlation != Cointegration.
        """
        spread = self._log_price_spread().dropna()
        adf_result = adfuller(spread)
        p_value = adf_result[1]

        print(f"Cointegration Test P-Value: {p_value:.5f}")
        if p_value < 0.05:
            print(">> PASS: The pair is cointegrated (log-price spread is stationary).")
            return True
        else:
            print(">> FAIL: The pair is NOT cointegrated. Do not trade.")
            return False

    def generate_live_signals(self):
        """
        Generates trading signals from the rolling Z-Score of the LOG-PRICE spread.
        """
        # 1. Spread on log prices (return_alpha intentionally NOT subtracted; see header note)
        self.data['Spread'] = self._log_price_spread()

        # 2. Rolling Mean and Std (dynamic, to adapt to market regimes)
        self.data['Spread_Mean'] = self.data['Spread'].rolling(window=self.window).mean()
        self.data['Spread_Std'] = self.data['Spread'].rolling(window=self.window).std()

        # 3. Z-Score
        self.data['Z_Score'] = (self.data['Spread'] - self.data['Spread_Mean']) / self.data['Spread_Std']

        # 4. Signal
        # Z > +thr: spread too wide  -> Short A, Long B (short the spread)
        # Z < -thr: spread too narrow -> Long A, Short B (long the spread)
        current_z = self.data['Z_Score'].iloc[-1]

        print(f"\n--- Signal Report [{pd.Timestamp.now().date()}] ---")
        print(f"Current Z-Score: {current_z:.2f}")

        if current_z > self.z_threshold:
            print(f"ACTION: SELL {self.ticker_a} / BUY {self.ticker_b} (Short the Spread)")
        elif current_z < -self.z_threshold:
            print(f"ACTION: BUY {self.ticker_a} / SELL {self.ticker_b} (Long the Spread)")
        else:
            print("ACTION: HOLD / NO TRADE")

        return self.data

    def plot_performance(self):
        """Visualizes the log-price spread and Z-score for analysis."""
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Plot Spread
        ax[0].plot(self.data.index, self.data['Spread'], label='Log-Price Spread')
        ax[0].plot(self.data.index, self.data['Spread_Mean'], color='orange', label='Rolling Mean')
        ax[0].set_title(f"Log-Price Spread: log({self.ticker_a}) - {self.hedge_ratio:.2f}*log({self.ticker_b})")
        ax[0].legend()

        # Plot Z-Score
        ax[1].plot(self.data.index, self.data['Z_Score'], label='Z-Score')
        ax[1].axhline(self.z_threshold, color='red', linestyle='--')
        ax[1].axhline(-self.z_threshold, color='green', linestyle='--')
        ax[1].axhline(0, color='black', linestyle='-')
        ax[1].set_title("Z-Score Signal")
        ax[1].legend()

        plt.tight_layout()
        plt.show()


# --- Execution ---
if __name__ == "__main__":
    # ver2: arbitrage on LOG RETURNS (beta) + LOG-PRICE spread.
    # Default to resilient auto mode; set data_source="ib" if IB Gateway/TWS is running.
    # bot = StatArbBot("QQQ", "SPY", lookback_window=30, data_source="auto")
    bot = StatArbBot("IVV", "VOO", lookback_window=30, data_source="ib")
    try:
        # 1. Get Data
        bot.fetch_data()

        # 2. Calibrate Strategy (beta from log returns)
        bot.calculate_beta_log_returns()

        # 3. Check Statistical Validity on the log-price spread
        is_valid = bot.check_cointegration()

        # 4. Run Strategy
        if is_valid:
            bot.generate_live_signals()
            bot.plot_performance()
    except Exception as exc:
        print(f"Strategy stopped: {exc}")
        raise SystemExit(1)
