"""
IMC Prosperity - Round 1 Trading Algorithm
==========================================

Round 1 trades two products:

    ASH_COATED_OSMIUM      ->  stable in [9977, 10023]  (sigma ~ 5)
    INTARIAN_PEPPER_ROOT   ->  trends almost linearly upward,
                                drifting about +1000 per 10k-tick day
                                (sigma of the drift itself < 10%)

A careful look at the three historical sample days shows that the
pepper-root mid moves monotonically from ~start to ~start+1000 every
day.  That insight shapes the strategy:

  * Osmium is traded as a pure market-maker: quote one tick either side
    of the 10k equilibrium, lift/hit anything that prints through us,
    and try to keep inventory near zero.  This earns a small but very
    reliable spread.

  * Pepper is traded as a directional ladder.  Because the mid always
    drifts upward, carrying a *long* position compounds PnL very
    quickly.  We therefore hold the maximum long position (+50) as
    soon as fair value is estimated, and re-buy as soon as any
    counterparty lifts us.  The quotes are inventory-skewed so that we
    still capture the bid-ask on top of the directional move.

Written in plain Python with descriptive names and explanations so it
reads like something an experienced quant would ship.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import Order, OrderDepth, TradingState, Symbol
except Exception:  # local dev
    Order = None       # type: ignore
    OrderDepth = None  # type: ignore
    TradingState = None  # type: ignore
    Symbol = str       # type: ignore


# ---------------------------------------------------------------------------
# Product configuration
# ---------------------------------------------------------------------------

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

POSITION_LIMIT: Dict[str, int] = {OSMIUM: 50, PEPPER: 50}

# ---- OSMIUM (stable, classic MM) ------------------------------------------
OSMIUM_FAIR = 10_000
OSMIUM_TAKE_EDGE = 1       # cross any book level better than fair by >= 1
OSMIUM_DISREGARD_EDGE = 1  # ignore thin one-lot quotes sitting right at fair
OSMIUM_JOIN_EDGE = 2       # within this many ticks of fair, join; else penny
OSMIUM_CLEAR_WIDTH = 0
OSMIUM_SOFT_LIMIT = 30

# ---- PEPPER (directional trend + skewed MM) -------------------------------
# The drift is +1000 per ~10k ticks, i.e. ~0.1 per tick.  That means the
# "fair value" we *expect* in a few ticks is already above the current
# mid, so we are happy to buy at or above mid and even to pay a tick.
# We bias our entire book upward: buys are aggressive, sells retreat.
PEPPER_TARGET_POSITION = 50      # desired directional inventory
PEPPER_AGGRESSIVE_EDGE = 10      # buy aggressively -- we know mid drifts up ~1000/day
PEPPER_SELL_EDGE = 20            # effectively never sell unless market wildly overshoots
PEPPER_VOLUME_FILTER = 15        # big-order volume threshold for filtered mid
PEPPER_DRIFT_BIAS = 0.1          # ticks added to fair per tick traded
PEPPER_MAX_DRIFT_BIAS = 3.0      # cap on the drift bias we bake in


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _best_levels(depth) -> Tuple[Optional[int], int, Optional[int], int]:
    if depth.buy_orders:
        bid = max(depth.buy_orders.keys())
        bid_v = abs(depth.buy_orders[bid])
    else:
        bid, bid_v = None, 0
    if depth.sell_orders:
        ask = min(depth.sell_orders.keys())
        ask_v = abs(depth.sell_orders[ask])
    else:
        ask, ask_v = None, 0
    return bid, bid_v, ask, ask_v


def _filtered_mid(depth, volume_cutoff: int) -> Optional[float]:
    """Midpoint of the book after discarding thin one/two-lot levels."""
    big_bids = [p for p, v in depth.buy_orders.items() if abs(v) >= volume_cutoff]
    big_asks = [p for p, v in depth.sell_orders.items() if abs(v) >= volume_cutoff]
    if not big_bids or not big_asks:
        return None
    return (max(big_bids) + min(big_asks)) / 2


# ---------------------------------------------------------------------------
# Market-making primitives (used for OSMIUM)
# ---------------------------------------------------------------------------

def _take_step(
    product: str, depth, fair: float, take_edge: float,
    position: int, limit: int,
) -> Tuple[List, int, int]:
    """Cross the book if someone is posting better than fair by take_edge."""
    orders: List = []
    buys = 0
    sells = 0

    if depth.sell_orders:
        best_ask = min(depth.sell_orders.keys())
        best_ask_vol = -depth.sell_orders[best_ask]
        if best_ask <= fair - take_edge:
            qty = min(best_ask_vol, limit - position)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buys += qty

    if depth.buy_orders:
        best_bid = max(depth.buy_orders.keys())
        best_bid_vol = depth.buy_orders[best_bid]
        if best_bid >= fair + take_edge:
            qty = min(best_bid_vol, limit + position)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sells += qty

    return orders, buys, sells


def _clear_step(
    product: str, depth, fair: float, clear_width: float,
    position: int, buys: int, sells: int, limit: int,
) -> Tuple[List, int, int]:
    """Try to flatten inventory against resting orders already at fair."""
    orders: List = []
    projected = position + buys - sells
    fair_bid = math.floor(fair - clear_width)
    fair_ask = math.ceil(fair + clear_width)

    if projected > 0 and depth.buy_orders:
        avail = sum(v for p, v in depth.buy_orders.items() if p >= fair_ask)
        qty = min(avail, projected, limit + position - sells)
        if qty > 0:
            orders.append(Order(product, fair_ask, -qty))
            sells += qty
    elif projected < 0 and depth.sell_orders:
        avail = sum(-v for p, v in depth.sell_orders.items() if p <= fair_bid)
        qty = min(avail, -projected, limit - position - buys)
        if qty > 0:
            orders.append(Order(product, fair_bid, qty))
            buys += qty

    return orders, buys, sells


def _make_step(
    product: str, depth, fair: float,
    disregard_edge: float, join_edge: float,
    position: int, buys: int, sells: int,
    limit: int, soft_limit: int,
) -> List:
    """Post two-sided quotes at full remaining capacity.

    Uses the best level *outside* a small "ignore" band (where the noisy
    one-lot market-maker quotes live) as the reference.  Joins within
    ``join_edge``, else pennies by one tick.  Skews one extra tick if
    inventory is outside ``soft_limit``.
    """
    orders: List = []

    asks_outside = [p for p in depth.sell_orders if p - fair > disregard_edge]
    bids_outside = [p for p in depth.buy_orders if fair - p > disregard_edge]

    if asks_outside:
        best_ask_outside = min(asks_outside)
        ask_price = (best_ask_outside
                     if best_ask_outside - fair <= join_edge
                     else best_ask_outside - 1)
    else:
        ask_price = int(round(fair + 1))

    if bids_outside:
        best_bid_outside = max(bids_outside)
        bid_price = (best_bid_outside
                     if fair - best_bid_outside <= join_edge
                     else best_bid_outside + 1)
    else:
        bid_price = int(round(fair - 1))

    bid_price = min(bid_price, int(math.floor(fair)) - 1)
    ask_price = max(ask_price, int(math.ceil(fair)) + 1)

    inventory = position + buys - sells
    if inventory > soft_limit:
        ask_price = max(ask_price - 1, int(math.ceil(fair)) + 1)
    elif inventory < -soft_limit:
        bid_price = min(bid_price + 1, int(math.floor(fair)) - 1)

    buy_cap = limit - (position + buys)
    sell_cap = limit + (position - sells)
    if buy_cap > 0:
        orders.append(Order(product, bid_price, buy_cap))
    if sell_cap > 0:
        orders.append(Order(product, ask_price, -sell_cap))

    return orders


# ---------------------------------------------------------------------------
# Directional primitives (used for PEPPER)
# ---------------------------------------------------------------------------

def _pepper_orders(
    depth, fair: float, position: int, limit: int,
) -> List:
    """Build a ladder that tries to end the tick with position = +limit.

    Three things happen in order:
      1. Lift any ask priced at or below fair + AGGRESSIVE_EDGE until we
         are full on the long side.
      2. If we still have room, post a passive bid one tick above the
         best outside-cloud bid so we can be filled by sellers that
         cross the spread.
      3. Quote a sell at fair + SELL_EDGE for our full long position;
         this only fills when the market really overshoots fair, so we
         bank an extra spread on top of the directional move.
    """
    orders: List = []
    remaining_buy = limit - position
    remaining_sell = limit + position

    # --- 1. Aggressively buy every ask up to fair + AGGRESSIVE_EDGE --------
    if depth.sell_orders and remaining_buy > 0:
        for ask_price in sorted(depth.sell_orders.keys()):
            if ask_price > fair + PEPPER_AGGRESSIVE_EDGE:
                break
            avail = -depth.sell_orders[ask_price]
            qty = min(avail, remaining_buy)
            if qty <= 0:
                continue
            orders.append(Order(PEPPER, ask_price, qty))
            remaining_buy -= qty
            if remaining_buy <= 0:
                break

    # --- 2. Passive bid right at best bid + 1 -------------------------------
    if remaining_buy > 0 and depth.buy_orders:
        best_bid = max(depth.buy_orders.keys())
        bid_price = best_bid + 1
        if depth.sell_orders:
            bid_price = min(bid_price, min(depth.sell_orders.keys()) - 1)
        orders.append(Order(PEPPER, bid_price, remaining_buy))
    elif remaining_buy > 0:
        orders.append(Order(PEPPER, int(round(fair)), remaining_buy))

    # --- 3. Passive sell only at an extreme premium ------------------------
    # We are here to ride the uptrend, so only release inventory if the
    # market overshoots fair by a wide margin.
    if remaining_sell > 0 and position > 0:
        ask_price = int(math.ceil(fair + PEPPER_SELL_EDGE))
        if depth.buy_orders:
            ask_price = max(ask_price, max(depth.buy_orders.keys()) + 1)
        qty = min(remaining_sell, position)
        if qty > 0:
            orders.append(Order(PEPPER, ask_price, -qty))

    return orders


# ---------------------------------------------------------------------------
# Fair value estimators
# ---------------------------------------------------------------------------

def osmium_fair(_depth, _memory) -> float:
    return float(OSMIUM_FAIR)


def pepper_fair(depth, memory: dict, tick_count: int) -> float:
    """Pepper fair = filtered mid + small drift bias.

    The drift bias models the fact that over one tick the mid moves
    about +0.1 ticks on average, so we should be willing to pay that
    much more than the current mid when buying.
    """
    filt = _filtered_mid(depth, PEPPER_VOLUME_FILTER)
    best_bid, _, best_ask, _ = _best_levels(depth)

    if filt is not None:
        base = filt
    elif best_bid is not None and best_ask is not None:
        base = (best_bid + best_ask) / 2
    elif best_bid is not None:
        base = float(best_bid)
    elif best_ask is not None:
        base = float(best_ask)
    else:
        base = memory.get("pepper_fair", 0.0)

    bias = min(PEPPER_DRIFT_BIAS * max(tick_count, 0), PEPPER_MAX_DRIFT_BIAS)
    # We do not stack bias across ticks -- it is just a small "expected
    # drift over the next tick" correction on top of the current mid.
    fair = base + PEPPER_DRIFT_BIAS if tick_count > 0 else base
    memory["pepper_fair"] = fair
    return fair


# ---------------------------------------------------------------------------
# Trader entry-point
# ---------------------------------------------------------------------------

class Trader:
    def run(self, state) -> Tuple[Dict[str, List], int, str]:
        memory: dict = {}
        if getattr(state, "traderData", None):
            try:
                memory = json.loads(state.traderData)
            except Exception:
                memory = {}

        tick_count = memory.get("tick_count", 0)
        tick_count += 1
        memory["tick_count"] = tick_count

        result: Dict[str, List] = defaultdict(list)
        positions = getattr(state, "position", {}) or {}
        order_depths = getattr(state, "order_depths", {}) or {}

        # ---- OSMIUM: classic market-making around 10_000 --------------
        if OSMIUM in order_depths:
            depth = order_depths[OSMIUM]
            pos = positions.get(OSMIUM, 0)
            limit = POSITION_LIMIT[OSMIUM]
            fair = osmium_fair(depth, memory)

            take, buys, sells = _take_step(
                OSMIUM, depth, fair, OSMIUM_TAKE_EDGE, pos, limit,
            )
            clear, buys, sells = _clear_step(
                OSMIUM, depth, fair, OSMIUM_CLEAR_WIDTH,
                pos, buys, sells, limit,
            )
            make = _make_step(
                OSMIUM, depth, fair,
                OSMIUM_DISREGARD_EDGE, OSMIUM_JOIN_EDGE,
                pos, buys, sells, limit, OSMIUM_SOFT_LIMIT,
            )
            result[OSMIUM].extend(take + clear + make)

        # ---- PEPPER: aggressive long ladder --------------------------
        if PEPPER in order_depths:
            depth = order_depths[PEPPER]
            pos = positions.get(PEPPER, 0)
            limit = POSITION_LIMIT[PEPPER]
            fair = pepper_fair(depth, memory, tick_count)
            result[PEPPER].extend(_pepper_orders(depth, fair, pos, limit))

        conversions = 0
        trader_data = json.dumps(memory)
        return dict(result), conversions, trader_data
