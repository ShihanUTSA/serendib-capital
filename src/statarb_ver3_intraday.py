#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statarb_ver3.py

Statistical Arbitrage Bot — ENGLE-GRANGER variant (industry-standard pairs trading).

Difference vs. statarb_ver2.py
------------------------------
ver2 estimated beta from LOG RETURNS (a short-horizon correlation beta) and
applied it to log prices. ver3 estimates beta directly from LOG PRICE LEVELS
via the Engle-Granger two-step method:

    Step 1: OLS on log prices:   log(A) = alpha + beta * log(B) + eps
    Step 2: test eps for stationarity (cointegration test)

    spread = log(A) - beta * log(B) - alpha   (the EG residual, mean ~0)

Why this is the right beta:
    The EG beta is the cointegrating coefficient — exactly the coefficient
    whose residual the stationarity test validates. A return-regression beta
    (ver2) is not guaranteed to equal it and can fail the ADF test even when
    a cointegrating beta exists.

Note on the intercept (level_alpha):
    Unlike ver2's per-period return drift, the EG intercept IS a level offset
    on log prices, so it is subtracted from the spread. The resulting
    residual spread is approximately mean-zero by construction.

Note on the cointegration test:
    Plain ADF on fitted OLS residuals uses the wrong null distribution and is
    biased toward finding stationarity (the fit minimizes residual variance).
    statsmodels' coint() applies the correct MacKinnon critical values.

Caveat: EG is asymmetric — regressing log(B) on log(A) gives a different
beta. For robustness, check both directions; for 3+ asset baskets use the
Johansen test instead.

Created on Wed Feb 11 2026
@author: shihan
"""

import asyncio
import contextlib
import io
import os
import random
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
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
        self.hedge_ratio = None      # cointegrating beta on log prices (Engle-Granger)
        self.level_alpha = None      # level offset from the EG log-price regression
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
        # Randomize client IDs so reruns from a live kernel don't hit error 326
        # (stale sessions from previous runs still hold the deterministic IDs).
        candidate_client_ids = random.sample(range(1000, 10000), 6)

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
            if period.startswith("1m"):
                return "1 M"
            if period.startswith("2w"):
                return "2 W"
            if period.startswith("1w"):
                return "1 W"
            if period.startswith("2d"):
                return "2 D"
            if period.startswith("1y") or period.startswith("365"):
                return "1 Y"
            if period.startswith("6m"):
                return "6 M"
            if period.startswith("3m"):
                return "3 M"
            if period.startswith("1m"):
                return "1 M"
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

    def _download_close_series_ib(self, ticker, period, interval, end_datetime=None):
        ib = self._connect_ib()
        contract = Stock(ticker, "SMART", "USD")
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=(
                pd.Timestamp(end_datetime).strftime("%Y%m%d %H:%M:%S")
                if end_datetime is not None else ""
            ),
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

    def _download_close_series_yf(self, ticker, period, interval, end_datetime=None):
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
            if end_datetime is not None:
                df = df[df.index <= pd.Timestamp(end_datetime)]
                if df.empty:
                    raise ValueError(f"No close-price data remained before {end_datetime} for {ticker}")
            return df
        if isinstance(df, pd.DataFrame):
            if df.empty:
                raise ValueError(f"No close-price data returned for {ticker}")
            if "Close" in df.columns:
                series = df["Close"]
            elif "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                series = None
            if series is not None:
                if end_datetime is not None:
                    series = series[series.index <= pd.Timestamp(end_datetime)]
                if series.empty:
                    raise ValueError(f"No close-price data remained before {end_datetime} for {ticker}")
                return series

        raise ValueError(f"No close-price data returned for {ticker}")

    def _download_close_series(self, ticker, period, interval, end_datetime=None):
        if self.data_source == "ib":
            try:
                print(f"Attempting to load {ticker} from Interactive Brokers...")
                return self._download_close_series_ib(ticker, period, interval, end_datetime)
            except Exception as exc:
                raise RuntimeError(f"Unable to load data for {ticker} from Interactive Brokers: {exc}") from exc

        if self.data_source == "yfinance":
            try:
                return self._download_close_series_yf(ticker, period, interval, end_datetime)
            except Exception as exc:
                raise RuntimeError(f"Unable to load data for {ticker} from Yahoo Finance: {exc}") from exc

        # Auto mode: try Interactive Brokers first, then Yahoo Finance, then local files.
        try:
            return self._download_close_series_ib(ticker, period, interval, end_datetime)
        except Exception as exc:
            print(f"Interactive Brokers failed for {ticker}: {exc}")

        try:
            return self._download_close_series_yf(ticker, period, interval, end_datetime)
        except Exception as exc:
            print(f"Yahoo Finance fetch failed for {ticker}: {exc}")

        local_series = self._load_local_close_series(ticker)
        if local_series is not None:
            if end_datetime is not None:
                local_series = local_series[local_series.index <= pd.Timestamp(end_datetime)]
            if local_series.empty:
                raise RuntimeError(f"No local price data remained before {end_datetime} for {ticker}")
            print(f"Using local data for {ticker}")
            return local_series

        raise RuntimeError(f"Unable to load price data for {ticker}")

    def fetch_data(self, period="1m", interval="5m", end_datetime=None):
        """
        Fetches intraday historical data for calibration.
        In a production environment, this would connect to a websocket (e.g., IBKR, Alpaca).
        """
        print(f"Fetching data for {self.ticker_a} and {self.ticker_b}...")

        df_a = self._download_close_series(self.ticker_a, period, interval, end_datetime)
        df_b = self._download_close_series(self.ticker_b, period, interval, end_datetime)

        # Merge and clean data
        self.data = pd.concat([df_a, df_b], axis=1)
        self.data.columns = [self.ticker_a, self.ticker_b]
        self.data.dropna(inplace=True)
        if self.data.empty:
            raise ValueError("No usable price data was loaded for the selected pair.")
        return self.data

    def calculate_beta_engle_granger(self):
        """
        Engle-Granger step 1: OLS on LOG PRICES (levels, not returns):

            log(A) = alpha + beta * log(B) + eps

        beta is the cointegrating coefficient; eps is the tradable spread.
        Unlike a return-regression beta (ver2), this beta is exactly the one
        the cointegration test in check_cointegration() validates.
        """
        if self.data.empty or self.ticker_a not in self.data.columns \
                or self.ticker_b not in self.data.columns:
            raise ValueError("Price data is not loaded yet or is missing the required columns.")

        if len(self.data) < 2:
            raise ValueError("Not enough overlapping data to run the Engle-Granger regression.")

        log_a = np.log(self.data[self.ticker_a])
        log_b = np.log(self.data[self.ticker_b])

        X = sm.add_constant(log_b)
        model = sm.OLS(log_a, X)
        results = model.fit()

        self.hedge_ratio = results.params[self.ticker_b]  # cointegrating beta on log prices
        self.level_alpha = results.params["const"]        # level offset (log-price intercept)
        self.results = results

        print(f"Beta (Engle-Granger, log prices): {self.hedge_ratio:.4f}")
        print(f"Alpha (log-price level offset):   {self.level_alpha:.6f}")
        print(f"R-squared (log-price fit):        {results.rsquared:.4f}")
        return self.hedge_ratio, self.results

    def _log_price_spread(self):
        """EG residual spread: log(A) - beta*log(B) - alpha (alpha is a level offset here)."""
        if self.hedge_ratio is None:
            raise ValueError("Beta is not calibrated yet. Call calculate_beta_engle_granger() first.")
        log_a = np.log(self.data[self.ticker_a])
        log_b = np.log(self.data[self.ticker_b])
        return log_a - self.hedge_ratio * log_b - self.level_alpha

    def check_cointegration(self):
        """
        Engle-Granger step 2: test the log-price residual for stationarity.
        Uses coint() (correct MacKinnon distribution for estimated residuals)
        rather than a plain ADF, which overstates significance.
        """
        log_a = np.log(self.data[self.ticker_a]).dropna()
        log_b = np.log(self.data[self.ticker_b]).dropna()
        _, p_value, _ = coint(log_a, log_b)

        print(f"Cointegration Test P-Value (Engle-Granger): {p_value:.5f}")
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
        # 1. EG residual spread (alpha already removed; mean ~0 by construction)
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


def scan_pairs(pairs, formation_period="1m", formation_interval="5m",
               trading_period="2d", trading_interval="5m", lookback_window=6,
               z_threshold=2.0, data_source="ib", ib_host=None,
               ib_port=7496, ib_client_id=1, report_path=None):
    """Fit on one month, then record every latest-day intraday signal."""
    bots = [
        StatArbBot(
            ticker_a,
            ticker_b,
            lookback_window=lookback_window,
            z_threshold=z_threshold,
            data_source=data_source,
            ib_host=ib_host,
            ib_port=ib_port,
            ib_client_id=ib_client_id,
        )
        for ticker_a, ticker_b in pairs
    ]

    shared_ib = None
    try:
        if data_source in {"ib", "auto"} and bots:
            shared_ib = bots[0]._connect_ib()
            for bot in bots[1:]:
                bot.ib = shared_ib

        results = []
        for bot in bots:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    bot.fetch_data(period=trading_period, interval=trading_interval)
                trading_data = bot.data.copy()
                latest_date = trading_data.index[-1].date()
                formation_end = pd.Timestamp(latest_date)

                with contextlib.redirect_stdout(io.StringIO()):
                    bot.fetch_data(
                        period=formation_period,
                        interval=formation_interval,
                        end_datetime=formation_end,
                    )
                    bot.calculate_beta_engle_granger()
                    is_valid = bot.check_cointegration()
                if is_valid:
                    bot.data = trading_data
                    bot.data = bot.data[bot.data.index.date == latest_date].copy()
                    with contextlib.redirect_stdout(io.StringIO()):
                        bot.generate_live_signals()

                    signal_data = bot.data.dropna(subset=["Z_Score"])
                    signal_data = signal_data[
                        (signal_data["Z_Score"] > bot.z_threshold)
                        | (signal_data["Z_Score"] < -bot.z_threshold)
                    ]
                    for signal_time, row in signal_data.iterrows():
                        current_z = row["Z_Score"]
                        signal = (
                            f"SELL {bot.ticker_a} / BUY {bot.ticker_b}"
                            if current_z > bot.z_threshold
                            else f"BUY {bot.ticker_a} / SELL {bot.ticker_b}"
                        )
                        results.append({
                            "Pair": f"{bot.ticker_a}/{bot.ticker_b}",
                            "Signal_Time": signal_time,
                            "Cointegrated": True,
                            "Z_Score": current_z,
                            "Signal": signal,
                        })

                    if signal_data.empty:
                        results.append({
                            "Pair": f"{bot.ticker_a}/{bot.ticker_b}",
                            "Signal_Time": bot.data.index[-1],
                            "Cointegrated": True,
                            "Z_Score": bot.data["Z_Score"].iloc[-1],
                            "Signal": "HOLD / NO TRADE",
                        })
                else:
                    results.append({
                        "Pair": f"{bot.ticker_a}/{bot.ticker_b}",
                        "Signal_Time": trading_data.index[-1],
                        "Cointegrated": False,
                        "Z_Score": np.nan,
                        "Signal": "SKIP / NOT COINTEGRATED",
                    })
            except Exception as exc:
                print(f"Skipping {bot.ticker_a}/{bot.ticker_b}: {exc}")
                results.append({
                    "Pair": f"{bot.ticker_a}/{bot.ticker_b}",
                    "Signal_Time": pd.NaT,
                    "Cointegrated": False,
                    "Z_Score": np.nan,
                    "Signal": f"ERROR: {exc}",
                })

        report = pd.DataFrame(results)
        if report_path:
            report.to_csv(report_path, index=False)
        return report
    finally:
        if shared_ib is not None and shared_ib.isConnected():
            shared_ib.disconnect()


# --- Execution ---
if __name__ == "__main__":
    pairs = [
        ("SPY", "IVV"), ("VOO", "IVV"), ("QQQ", "XLK"), ("IWM", "VTWO"),
        ("VTI", "ITOT"), ("EFA", "IEFA"), ("EEM", "VWO"), ("AGG", "BND"),
        ("XLE", "VDE"), ("XLF", "VFH"), ("XLK", "VGT"), ("XLI", "VIS"),
        ("XLP", "VDC"), ("XLV", "VHT"), ("XLY", "VCR"), ("XLB", "VAW"),
        ("XLC", "VOX"), ("XOM", "CVX"), ("KO", "PEP"), ("JPM", "BAC"),
    ]

    signals = scan_pairs(
        pairs,
        formation_period="1m",
        formation_interval="5m",
        trading_period="2d",
        trading_interval="5m",
        lookback_window=6,
        z_threshold=1.7,
        data_source="ib",
        report_path="intraday_signal_report.csv",
    )
    print("\n=== Intraday Pair Scan ===")
    print(signals.to_string(index=False, max_rows=None, max_cols=None))
    print(f"\nFull report saved to intraday_signal_report.csv ({len(signals)} pairs).")
