#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 06:04:31 2026

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
    A Statistical Arbitrage Bot utilizing Cointegration and Z-Score strategies.
    Designed for live-simulation and signal generation.
    """
    
    def __init__(self, ticker_a, ticker_b, lookback_window=60, z_threshold=2.0,
                 data_source="auto", ib_host=None, ib_port=7496, ib_client_id=1):
        """
        Initialize the StatArbBot.

        Functionality: store configuration and initialize internal state used
        by subsequent data-fetching, calibration and signal methods.

        Inputs:
        - `ticker_a`, `ticker_b`: symbols for the pair
        - `lookback_window`: rolling window length (integer)
        - `z_threshold`: z-score threshold for signals (float)
        - `data_source`: 'ib', 'yfinance' or 'auto'
        - `ib_host`, `ib_port`, `ib_client_id`: Interactive Brokers connection params

        Side-effects: sets `self.data`, `self.hedge_ratio`, `self.results`, and
        `self.ib` placeholders for later use.
        """

        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.window = lookback_window
        self.z_threshold = z_threshold
        self.data_source = data_source
        self.ib_host = ib_host or os.getenv("IB_HOST", "127.0.0.1")
        self.ib_port = int(ib_port or os.getenv("IB_PORT", 7496))
        self.ib_client_id = int(ib_client_id or os.getenv("IB_CLIENT_ID", 1))
        self.data = pd.DataFrame()
        self.hedge_ratio = None
        self.results = None
        self.ib = None
        
    def _load_local_close_series(self, ticker):
        """
        Load close prices from local files in the repository `data/` folder.

        Functionality: scan recent CSV/XLSX files whose filenames contain the
        `ticker`, read `StartTime` and `ClosePrice` columns, combine and
        resample to daily closes.

        Input: `ticker` (string)
        Output: pandas Series of daily close prices or `None` if no files found
        or parsing fails.
        Side-effects: none (returns a Series; does not modify `self.data`).
        """
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
        """
        Establish a connection to Interactive Brokers using `ib_insync`.

        Functionality: ensure an event loop is available, try multiple ports
        and client IDs to avoid collisions, and return an active `IB` object.

        Inputs: none (uses instance IB connection params)
        Output: an `IB` connection instance on success; raises RuntimeError if
        `ib_insync` is not installed or all attempts fail.
        Side-effects: sets `self.ib` to the connected `IB` instance.
        """

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
        """
        Map a human-friendly `period` string to an IB `durationStr`.

        Functionality: convert inputs like '1y', '3m', '1d' to IB formats
        such as '1 Y', '3 M', '1 D'. Returns a sensible default ('1 Y') for
        unknown inputs.

        Input: `period` (string)
        Output: IB duration string
        Side-effects: none
        """

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
        """
        Map a human-friendly `interval` to an IB `barSizeSetting`.

        Functionality: convert interval hints like '1m', '5m', '1d', '1wk'
        into IB bar-size strings ('1 min', '5 mins', '1 day', '1 week').

        Input: `interval` (string)
        Output: IB bar size string
        Side-effects: none
        """

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
        """
        Download historical close prices for `ticker` from Interactive Brokers.

        Functionality: request historical bars using `ib.reqHistoricalData`,
        convert the returned bars to a DataFrame, set a datetime index and
        return the close price Series (float). Raises ValueError on empty or
        unexpected responses.

        Inputs: `ticker`, `period`, `interval`
        Output: pandas Series of close prices (named by `ticker`)
        Side-effects: may call `_connect_ib()` which sets `self.ib`.
        """

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
        """
        Download historical close prices for `ticker` using `yfinance`.

        Functionality: call `yf.download` and return the `Close` (or
        `Adj Close`) column as a pandas Series. Raises ValueError if no data
        is returned.

        Inputs: `ticker`, `period`, `interval`
        Output: pandas Series of close prices
        Side-effects: none
        """

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
        """
        Orchestrate close-series download based on `self.data_source`.

        Functionality: if `data_source` is explicit it tries that source,
        otherwise in `auto` mode it attempts IB, then Yahoo Finance, then
        local files. Returns a close-price Series or raises RuntimeError if
        all attempts fail.

        Inputs: `ticker`, `period`, `interval`
        Output: pandas Series of close prices
        Side-effects: prints progress messages and may set up an IB
        connection via `_connect_ib()`.
        """

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
        Fetch historical close-price data for both tickers and prepare `self.data`.

        Functionality: download close-series for `ticker_a` and `ticker_b`,
        concatenate them into a two-column DataFrame, drop rows with NaNs,
        and store the result in `self.data` for calibration and signal
        generation.

        Inputs (optional): `period` and `interval` forwarded to downloaders.
        Output: DataFrame stored in `self.data`.
        Side-effects: updates `self.data` and may raise on no usable data.
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

    def calculate_hedge_ratio(self):
        """
        Calibrate the hedge ratio (`beta`) using Ordinary Least Squares.

        Functionality: regress `ticker_a` prices on `ticker_b` prices with a
        constant term. Stores the fitted `results` in `self.results` and the
        estimated slope in `self.hedge_ratio`.

        Inputs: uses `self.data` (must be loaded)
        Outputs: returns `(hedge_ratio, results)` and sets those as
        instance attributes.
        Side-effects: updates `self.hedge_ratio` and `self.results`.
        """
        if self.data.empty or self.ticker_a not in self.data.columns or self.ticker_b not in self.data.columns:
            raise ValueError("Price data is not loaded yet or is missing the required columns.")

        if len(self.data) < 2:
            raise ValueError("Not enough overlapping price data to calibrate the hedge ratio.")

        model = sm.OLS(self.data[self.ticker_a], sm.add_constant(self.data[self.ticker_b]))
        results = model.fit()
        self.hedge_ratio = results.params[self.ticker_b]
        self.results = results
        print(f"Hedge Ratio (Beta): {self.hedge_ratio:.4f}")
        return self.hedge_ratio , self.results

    def check_cointegration(self):
        """
        Test whether the spread is stationary (cointegration) using ADF.

        Functionality: compute the spread `A - beta * B`, run
        `statsmodels.tsa.stattools.adfuller` on the spread and interpret the
        p-value. Returns `True` if p < 0.05 (stationary) else `False`.

        Inputs: uses `self.data` and `self.hedge_ratio` (must be set)
        Output: boolean indicating whether the spread is stationary
        Side-effects: prints the p-value and PASS/FAIL messages.
        """
        spread = self.data[self.ticker_a] - self.hedge_ratio * self.data[self.ticker_b]
        adf_result = adfuller(spread)
        p_value = adf_result[1]
        
        print(f"Cointegration Test P-Value: {p_value:.5f}")
        if p_value < 0.05:
            print(">> PASS: The pair is cointegrated (spread is stationary).")
            return True
        else:
            print(">> FAIL: The pair is NOT cointegrated. Do not trade.")
            return False

    def generate_live_signals(self):
        """
        Generate trading signals from the rolling Z-score of the spread.

        Functionality: compute `Spread = A - beta*B - intercept`, compute
        rolling mean/std over `self.window`, compute Z-score and produce a
        simple action based on `self.z_threshold` (sell spread, buy spread,
        or hold). Adds `Spread`, `Spread_Mean`, `Spread_Std`, and `Z_Score`
        columns to `self.data` and returns the DataFrame.

        Inputs: requires `self.data`, `self.hedge_ratio` and `self.results`
        Output: augmented `self.data` with signal columns
        Side-effects: prints a short signal report and updates `self.data`.
        """
        # 1. Calculate the Spread
        self.data['Spread'] = self.data[self.ticker_a] - self.hedge_ratio * self.data[self.ticker_b]
        
        """ Thilina - Calculating spread using both hedge_ration and intersection (alpha) will be more meaningful and easy 
        to interpret """
        self.data['Spread'] = self.data[self.ticker_a] - self.hedge_ratio * self.data[self.ticker_b] - self.results.params['const']
        
        
        # 2. Calculate Rolling Mean and Std (Dynamic, to adapt to market regimes)
        self.data['Spread_Mean'] = self.data['Spread'].rolling(window=self.window).mean()
        self.data['Spread_Std'] = self.data['Spread'].rolling(window=self.window).std()
        
        # 3. Calculate Z-Score
        self.data['Z_Score'] = (self.data['Spread'] - self.data['Spread_Mean']) / self.data['Spread_Std']
        
        # 4. Generate Signal
        # Z > 2: Spread is too wide -> Short A, Long B (Short Spread)
        # Z < -2: Spread is too narrow -> Long A, Short B (Long Spread)
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
        """
        Plot spread and Z-score diagnostics.

        Functionality: draws two subplots — the spread with its rolling mean,
        and the Z-score with threshold lines — then shows the figure using
        matplotlib. Expects columns created by `generate_live_signals`.

        Inputs: reads `self.data` fields (`Spread`, `Spread_Mean`, `Z_Score`)
        Output: displays matplotlib figure
        Side-effects: opens a plot window (or inline plot in notebooks).
        """
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot Spread
        ax[0].plot(self.data.index, self.data['Spread'], label='Spread')
        ax[0].plot(self.data.index, self.data['Spread_Mean'], color='orange', label='Rolling Mean')
        ax[0].set_title(f"Spread: {self.ticker_a} vs {self.hedge_ratio:.2f}*{self.ticker_b}")
        ax[0].legend()
        
        # Plot Z-Score
        ax[1].plot(self.data.index, self.data['Z_Score'], label='Z-Score')
        ax[1].axhline(self.z_threshold, color='red', linestyle='--')
        ax[1].axhline(-self.z_threshold, color='green', linestyle='--')
        ax[1].axhline(0, color='black', linestyle='-')
        ax[1].set_title("Z-Score Signal")
        
        plt.tight_layout()
        plt.show()

# --- Execution ---
if __name__ == "__main__":
    # Default to the resilient auto mode so the script runs even when IB is unavailable.
    # Set data_source="ib" explicitly if your IB Gateway/Trader Workstation is running.
    #bot = StatArbBot("QQQ", "SPY", lookback_window=30, data_source="auto")
    bot = StatArbBot("XLK", "QQQ", lookback_window=30, data_source="ib")
    try:
        # 1. Get Data
        bot.fetch_data()

        # 2. Calibrate Strategy
        bot.calculate_hedge_ratio()

        # 3. Check Statistical Validity (Crucial for Quant Interviews)
        is_valid = bot.check_cointegration()

        # 4. Run Strategy
        if is_valid:
            bot.generate_live_signals()
            bot.plot_performance()
    except Exception as exc:
        print(f"Strategy stopped: {exc}")
        raise SystemExit(1)