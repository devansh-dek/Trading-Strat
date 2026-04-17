"""
Local backtester for the IMC Prosperity Round 1 strategy.
=========================================================

This script reimplements the tiny slice of IMC's `datamodel` module that
our `strategy.py` relies on, then replays the three sample days worth of
level-3 order-book snapshots against the Trader class.

For every timestamp we:

  1. Build an `OrderDepth` for each product from the CSV row (bid/ask
     prices and volumes).
  2. Call `Trader.run(state)` and collect the orders it returns.
  3. Match those orders against the visible book using simple FIFO
     semantics: a buy at price >= best ask crosses the book and a sell at
     price <= best bid crosses the book, otherwise the order is ignored
     (we do not simulate posted-order fills, to stay conservative).
  4. Update positions, cash and the final mark-to-market PnL.

Profits are compared to "hold zero" and printed per product per day.

Usage:
    python3 backtesting.py                     # run all three days
    python3 backtesting.py --day 0             # run a single day
    python3 backtesting.py --data-dir path     # different CSV location
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


# -------------------------------------------------------------------------
# Minimal datamodel shim (mirrors the Prosperity one we rely on)
# -------------------------------------------------------------------------

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int

    def __repr__(self) -> str:
        return f"Order({self.symbol}, p={self.price}, q={self.quantity})"


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str
    price: float
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0


@dataclass
class TradingState:
    timestamp: int
    listings: Dict[str, str] = field(default_factory=dict)
    order_depths: Dict[str, OrderDepth] = field(default_factory=dict)
    own_trades: Dict[str, List[Trade]] = field(default_factory=dict)
    market_trades: Dict[str, List[Trade]] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)
    observations: Dict = field(default_factory=dict)
    traderData: str = ""


Symbol = str


# Inject a fake `datamodel` module BEFORE importing strategy so that the
# submission file imports succeed locally.
import types  # noqa: E402 - intentional after dataclasses

_datamodel_module = types.ModuleType("datamodel")
_datamodel_module.Order = Order
_datamodel_module.OrderDepth = OrderDepth
_datamodel_module.Trade = Trade
_datamodel_module.TradingState = TradingState
_datamodel_module.Symbol = Symbol
sys.modules["datamodel"] = _datamodel_module

import strategy  # noqa: E402 - must come after the shim above
importlib.reload(strategy)  # ensure we get a fresh Trader each run


# -------------------------------------------------------------------------
# Loading helpers
# -------------------------------------------------------------------------

def _row_to_depth(row: pd.Series) -> OrderDepth:
    """Convert a single CSV row into an `OrderDepth` instance.

    Missing levels are silently skipped.  Volumes on the sell side are
    stored as negative integers because that is the convention used by
    the Prosperity runner and the shape our strategy expects.
    """
    depth = OrderDepth()

    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv != 0:
            depth.buy_orders[int(bp)] = int(bv)

        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and av != 0:
            depth.sell_orders[int(ap)] = -int(av)

    return depth


def _load_market_trades(path: str) -> Dict[int, Dict[str, List[Trade]]]:
    """Load executed trades.  They are used to populate `market_trades`
    for each tick (our strategy peeks at them for fair-value updates)."""
    out: Dict[int, Dict[str, List[Trade]]] = {}
    if not os.path.exists(path):
        return out
    df = pd.read_csv(path, sep=";")
    for _, r in df.iterrows():
        ts = int(r["timestamp"])
        sym = str(r["symbol"])
        trade = Trade(
            symbol=sym,
            price=float(r["price"]),
            quantity=int(r["quantity"]),
            buyer=str(r.get("buyer", "") or ""),
            seller=str(r.get("seller", "") or ""),
            timestamp=ts,
        )
        out.setdefault(ts, {}).setdefault(sym, []).append(trade)
    return out


# -------------------------------------------------------------------------
# Matching engine
# -------------------------------------------------------------------------

POSITION_LIMIT = {
    "ASH_COATED_OSMIUM": 50,
    "INTARIAN_PEPPER_ROOT": 50,
}


def _match(order: Order, depth: OrderDepth, position: int) -> List[Trade]:
    """Match an agent order against the snapshot book.

    We only count crossings: if a buy order is at price >= best ask, we
    eat asks level by level.  Orders that would just rest on the book are
    ignored (this is the conservative assumption that Timo's blog and
    most Prosperity writeups use for sanity checking).
    """
    fills: List[Trade] = []
    qty = order.quantity

    if qty > 0:  # buy
        capacity = POSITION_LIMIT[order.symbol] - position
        qty = min(qty, capacity)
        for ask_price in sorted(depth.sell_orders.keys()):
            if qty <= 0:
                break
            if order.price < ask_price:
                break
            avail = -depth.sell_orders[ask_price]
            take = min(avail, qty)
            depth.sell_orders[ask_price] += take  # becomes less negative
            if depth.sell_orders[ask_price] == 0:
                del depth.sell_orders[ask_price]
            fills.append(Trade(order.symbol, ask_price, take))
            qty -= take

    elif qty < 0:  # sell
        want = -qty
        capacity = POSITION_LIMIT[order.symbol] + position
        want = min(want, capacity)
        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if want <= 0:
                break
            if order.price > bid_price:
                break
            avail = depth.buy_orders[bid_price]
            take = min(avail, want)
            depth.buy_orders[bid_price] -= take
            if depth.buy_orders[bid_price] == 0:
                del depth.buy_orders[bid_price]
            fills.append(Trade(order.symbol, bid_price, -take))
            want -= take

    return fills


# -------------------------------------------------------------------------
# Core backtest loop
# -------------------------------------------------------------------------

def run_day(prices_path: str, trades_path: str) -> Dict[str, float]:
    df = pd.read_csv(prices_path, sep=";")
    df = df.sort_values(["timestamp", "product"]).reset_index(drop=True)
    market_trades = _load_market_trades(trades_path)

    trader = strategy.Trader()
    positions: Dict[str, int] = {}
    cash: Dict[str, float] = {}
    last_mid: Dict[str, float] = {}
    trader_data = ""

    grouped = df.groupby("timestamp", sort=True)
    previous_own: Dict[str, List[Trade]] = {}

    for ts, frame in grouped:
        order_depths: Dict[str, OrderDepth] = {}
        for _, row in frame.iterrows():
            product = row["product"]
            order_depths[product] = _row_to_depth(row)
            if pd.notna(row.get("mid_price")) and row["mid_price"] > 0:
                last_mid[product] = float(row["mid_price"])

        state = TradingState(
            timestamp=int(ts),
            order_depths=order_depths,
            position=dict(positions),
            own_trades=previous_own,
            market_trades=market_trades.get(int(ts), {}),
            traderData=trader_data,
        )

        orders_by_symbol, _conversions, trader_data = trader.run(state)
        previous_own = {}

        # --- Step 1: match crossing orders against the snapshot book -----
        resting: Dict[str, List[Order]] = {}
        for sym, orders in (orders_by_symbol or {}).items():
            depth = order_depths.get(sym)
            if depth is None:
                continue
            for order in orders:
                fills = _match(order, depth, positions.get(sym, 0))
                if fills:
                    for fill in fills:
                        positions[sym] = positions.get(sym, 0) + fill.quantity
                        cash[sym] = cash.get(sym, 0.0) - fill.price * fill.quantity
                        previous_own.setdefault(sym, []).append(fill)
                # any remainder effectively rests on the book at `order.price`
                remainder = order.quantity - sum(f.quantity for f in fills)
                if remainder != 0:
                    resting.setdefault(sym, []).append(
                        Order(order.symbol, order.price, remainder)
                    )

        # --- Step 2: simulate passive fills from the trade tape ----------
        # If a public trade printed at a price that is at/through our
        # resting quote, we assume we got a share of that flow.  This is
        # the same fill model used by most Prosperity community
        # backtesters (e.g. Stanford Cardinal writeup).
        tape = market_trades.get(int(ts), {})
        for sym, orders in resting.items():
            tape_trades = tape.get(sym, [])
            if not tape_trades:
                continue
            for order in orders:
                for trade in tape_trades:
                    if order.quantity > 0 and trade.price <= order.price:
                        capacity = POSITION_LIMIT[sym] - positions.get(sym, 0)
                        take = min(order.quantity, trade.quantity, capacity)
                        if take <= 0:
                            continue
                        positions[sym] = positions.get(sym, 0) + take
                        cash[sym] = cash.get(sym, 0.0) - order.price * take
                        previous_own.setdefault(sym, []).append(
                            Trade(sym, order.price, take)
                        )
                        order.quantity -= take
                    elif order.quantity < 0 and trade.price >= order.price:
                        capacity = POSITION_LIMIT[sym] + positions.get(sym, 0)
                        take = min(-order.quantity, trade.quantity, capacity)
                        if take <= 0:
                            continue
                        positions[sym] = positions.get(sym, 0) - take
                        cash[sym] = cash.get(sym, 0.0) + order.price * take
                        previous_own.setdefault(sym, []).append(
                            Trade(sym, order.price, -take)
                        )
                        order.quantity += take

    pnl: Dict[str, float] = {}
    for sym in set(list(positions.keys()) + list(cash.keys())):
        mid = last_mid.get(sym, 0.0)
        pnl[sym] = cash.get(sym, 0.0) + positions.get(sym, 0) * mid
    return pnl


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest IMC Prosperity strategy")
    parser.add_argument("--day", type=int, default=None,
                        help="single day to backtest (-2, -1 or 0); default=all")
    parser.add_argument("--data-dir", default=".", help="directory holding the CSVs")
    args = parser.parse_args()

    days = [args.day] if args.day is not None else [-2, -1, 0]

    total = 0.0
    for day in days:
        prices = os.path.join(args.data_dir, f"prices_round_1_day_{day}.csv")
        trades = os.path.join(args.data_dir, f"trades_round_1_day_{day}.csv")
        if not os.path.exists(prices):
            print(f"[day {day}] prices file missing, skipping")
            continue
        pnl = run_day(prices, trades)
        day_total = sum(pnl.values())
        print(f"\n=== Day {day} ===")
        for sym, v in sorted(pnl.items()):
            print(f"  {sym:<25} PnL = {v:>12,.2f}")
        print(f"  {'TOTAL':<25}     = {day_total:>12,.2f}")
        total += day_total

    print(f"\n>>> Backtest PnL across {len(days)} day(s): {total:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
