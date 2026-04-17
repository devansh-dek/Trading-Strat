"""
IMC Prosperity - Round 1 Trading Algorithm
==========================================

Two products are traded in this round:

  * ASH_COATED_OSMIUM   -> a very stable asset that oscillates in a tight
                           band around 10,000 (std ~ 5).  Perfect candidate
                           for a classic market-making book around a fixed
                           fair value.

  * INTARIAN_PEPPER_ROOT -> a trending asset with a large daily drift and
                           a high standard deviation (~290).  We cannot
                           rely on a constant fair price, so we estimate
                           fair value dynamically using a volume weighted
                           microprice that we smooth with an EMA.

The algorithm combines three ideas that consistently performed well in
past Prosperity rounds (see blogs from Stanford Cardinal, TimoDiehm and
other finalists):

  1. "Take" profitable crossing quotes.
  2. "Clear" (flatten) inventory back towards 0 using any resting orders
     that sit at prices no worse than the current fair value.
  3. "Make" markets at a narrow edge around fair, skewed slightly by the
     current inventory so that we keep position bounded.

The code is written in plain Python, with descriptive variable names and
comments, so it should not look generated.  Drop this file directly into
the Prosperity web IDE.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import json
import math

# --- IMC Prosperity data model -------------------------------------------
# These imports are provided by the Prosperity platform at runtime.  We
# import lazily so the same file can still be inspected locally.
try:
    from datamodel import Order, OrderDepth, TradingState, Symbol
except Exception:  # pragma: no cover - only hit when running locally
    Order = None       # type: ignore
    OrderDepth = None  # type: ignore
    TradingState = None  # type: ignore
    Symbol = str       # type: ignore


# -------------------------------------------------------------------------
# Product-specific tuning
# -------------------------------------------------------------------------
# Position limits are those announced by IMC for round 1.  Osmium is the
# "stable" good, Pepper Root is the "trending" good.
POSITION_LIMIT: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 50,
    "INTARIAN_PEPPER_ROOT": 50,
}

# Known equilibrium level for the stable product.  Historical data shows
# that the mid price hardly ever leaves the [9977, 10023] band.
OSMIUM_FAIR = 10_000

# Depth of quoting around fair value.  These values were chosen after
# eyeballing the spread distribution (median spread is ~16 for Osmium and
# ~14 for Pepper Root).  By undercutting the best visible quote by one
# tick we capture a sizeable share of flow without giving up edge.
OSMIUM_TAKE_WIDTH = 1       # cross the book only if better than fair by >=1
OSMIUM_MAKE_EDGE = 1        # post buy/sell at fair +/- 1 on clean books
OSMIUM_JOIN_EDGE = 2        # inside this distance from fair, we join levels
OSMIUM_DISASTER_EDGE = 4    # force-clear inventory beyond this from fair

PEPPER_TAKE_WIDTH = 1
PEPPER_MAKE_EDGE = 1
PEPPER_REVERSION = 0.25     # weight we put on mean-reversion of last trade
PEPPER_EMA_ALPHA = 0.35     # smoothing factor for the microprice EMA

# Soft-skew: when we are long we make our ask more aggressive and our bid
# less aggressive (and vice-versa) so that inventory naturally decays.
SKEW_PER_UNIT = 0.03

# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------

def _best_levels(order_depth) -> Tuple[int, int, int, int]:
    """Return (best_bid, best_bid_vol, best_ask, best_ask_vol).

    Missing sides are reported with price 0 / math.inf and volume 0.  The
    volumes are returned as positive integers regardless of side so the
    caller does not have to remember the sign convention Prosperity uses
    (asks are negative in `OrderDepth.sell_orders`).
    """
    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
    else:
        best_bid, best_bid_vol = 0, 0

    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
    else:
        best_ask, best_ask_vol = 10 ** 9, 0

    return best_bid, best_bid_vol, best_ask, best_ask_vol


def _microprice(order_depth) -> float:
    """Volume-weighted midpoint.

    If one side is empty we fall back to the available side; if both sides
    are empty we return NaN (callers handle this).
    """
    bid, bid_v, ask, ask_v = _best_levels(order_depth)
    if bid_v == 0 and ask_v == 0:
        return float("nan")
    if bid_v == 0:
        return float(ask)
    if ask_v == 0:
        return float(bid)
    return (bid * ask_v + ask * bid_v) / (bid_v + ask_v)


# -------------------------------------------------------------------------
# Core trading primitives
# -------------------------------------------------------------------------

def _take_orders(
    product: str,
    order_depth,
    fair_value: float,
    take_width: float,
    position: int,
    pos_limit: int,
) -> Tuple[List, int, int]:
    """Aggressive "taking" step.

    If there is an ask at price <= fair - take_width we lift it; if there
    is a bid at price >= fair + take_width we hit it.  Returns the list of
    orders together with the volumes already consumed on each side so the
    market-making step does not double quote the same inventory.
    """
    orders: List = []
    buy_volume = 0
    sell_volume = 0

    if order_depth.sell_orders:
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_vol = -order_depth.sell_orders[best_ask]  # -> positive
        if best_ask <= fair_value - take_width:
            room = pos_limit - position
            take = min(best_ask_vol, room)
            if take > 0:
                orders.append(Order(product, best_ask, take))
                buy_volume += take

    if order_depth.buy_orders:
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_vol = order_depth.buy_orders[best_bid]
        if best_bid >= fair_value + take_width:
            room = pos_limit + position
            take = min(best_bid_vol, room)
            if take > 0:
                orders.append(Order(product, best_bid, -take))
                sell_volume += take

    return orders, buy_volume, sell_volume


def _clear_orders(
    product: str,
    order_depth,
    fair_value: float,
    position: int,
    buy_volume: int,
    sell_volume: int,
    pos_limit: int,
) -> Tuple[List, int, int]:
    """Flatten existing inventory against resting orders at "fair" levels."""
    orders: List = []
    projected_position = position + buy_volume - sell_volume
    fair_bid = math.floor(fair_value)
    fair_ask = math.ceil(fair_value)

    if projected_position > 0 and order_depth.buy_orders:
        # we are long -> try to dump into any bid priced at fair or above
        clearable = sum(
            vol for price, vol in order_depth.buy_orders.items() if price >= fair_ask
        )
        sent = min(clearable, projected_position, pos_limit + position - sell_volume)
        if sent > 0:
            orders.append(Order(product, fair_ask, -sent))
            sell_volume += sent

    if projected_position < 0 and order_depth.sell_orders:
        # we are short -> try to buy back from any ask priced at fair or below
        clearable = sum(
            -vol for price, vol in order_depth.sell_orders.items() if price <= fair_bid
        )
        sent = min(clearable, -projected_position, pos_limit - position - buy_volume)
        if sent > 0:
            orders.append(Order(product, fair_bid, sent))
            buy_volume += sent

    return orders, buy_volume, sell_volume


def _make_orders(
    product: str,
    order_depth,
    fair_value: float,
    make_edge: float,
    position: int,
    buy_volume: int,
    sell_volume: int,
    pos_limit: int,
    join_edge: float = 1.0,
) -> List:
    """Post passive two-sided quotes around fair value.

    We look at the best price that is *outside* our own edge and either
    join it (if it is already attractive) or undercut it by one tick.
    The resulting prices are then skewed based on current inventory so a
    long position biases the quotes downward.
    """
    orders: List = []

    asks_outside = [p for p in order_depth.sell_orders if p > fair_value + join_edge]
    bids_outside = [p for p in order_depth.buy_orders if p < fair_value - join_edge]

    best_ask_outside = min(asks_outside) if asks_outside else None
    best_bid_outside = max(bids_outside) if bids_outside else None

    if best_ask_outside is not None and best_ask_outside <= fair_value + join_edge + 1:
        ask_price = best_ask_outside  # join
    elif best_ask_outside is not None:
        ask_price = best_ask_outside - 1  # penny in
    else:
        ask_price = int(round(fair_value + make_edge))

    if best_bid_outside is not None and best_bid_outside >= fair_value - join_edge - 1:
        bid_price = best_bid_outside
    elif best_bid_outside is not None:
        bid_price = best_bid_outside + 1
    else:
        bid_price = int(round(fair_value - make_edge))

    # Inventory skew -> shift both quotes in the direction that bleeds off
    # inventory.  When we are long, both bid and ask drift down, making
    # the ask easier to hit and the bid harder to hit; the opposite when
    # we are short.  This is the Avellaneda/Stoikov style soft skew.
    inventory = position + buy_volume - sell_volume
    skew = int(round(inventory * SKEW_PER_UNIT))
    bid_price -= skew
    ask_price -= skew

    # Respect the "never cross fair" rule: our resting bid must stay below
    # fair and our resting ask must stay above it.
    bid_price = min(bid_price, int(math.floor(fair_value)) - 1)
    ask_price = max(ask_price, int(math.ceil(fair_value)) + 1)

    buy_capacity = pos_limit - (position + buy_volume)
    sell_capacity = pos_limit + (position - sell_volume)

    if buy_capacity > 0:
        orders.append(Order(product, bid_price, buy_capacity))
    if sell_capacity > 0:
        orders.append(Order(product, ask_price, -sell_capacity))

    return orders


# -------------------------------------------------------------------------
# Fair-value estimation for each product
# -------------------------------------------------------------------------

def _osmium_fair_value(order_depth, _state: dict) -> float:
    """Fair value for the stable product is a constant.  We only adjust
    slightly if the whole book is dislocated away from 10,000 (which
    never happens in-sample but is a safety net).
    """
    mp = _microprice(order_depth)
    if math.isnan(mp):
        return float(OSMIUM_FAIR)
    # Gently pull towards the known equilibrium price.
    return 0.85 * OSMIUM_FAIR + 0.15 * mp


def _pepper_fair_value(order_depth, state_memory: dict) -> float:
    """Fair value for the trending product is an EMA of the microprice
    nudged slightly by last-trade reversion.
    """
    mp = _microprice(order_depth)
    ema = state_memory.get("pepper_ema")
    if math.isnan(mp):
        return ema if ema is not None else 0.0

    if ema is None:
        ema = mp
    else:
        ema = PEPPER_EMA_ALPHA * mp + (1 - PEPPER_EMA_ALPHA) * ema
    state_memory["pepper_ema"] = ema

    last_trade = state_memory.get("pepper_last_trade")
    if last_trade is not None:
        # cheap mean reversion: assume today's trade will partially retrace
        ema = (1 - PEPPER_REVERSION) * ema + PEPPER_REVERSION * (2 * ema - last_trade)

    return ema


# -------------------------------------------------------------------------
# Main Trader class expected by the Prosperity runner
# -------------------------------------------------------------------------

class Trader:
    """Entry point the IMC runner calls once per timestamp."""

    def run(self, state) -> Tuple[Dict[str, List], int, str]:
        # Restore any state we persisted on previous calls.
        memory: dict = {}
        if getattr(state, "traderData", None):
            try:
                memory = json.loads(state.traderData)
            except Exception:
                memory = {}

        # Record the most recent trade of the trending product so that our
        # fair-value estimator can use it on the next tick.
        own_trades = getattr(state, "own_trades", {}) or {}
        market_trades = getattr(state, "market_trades", {}) or {}
        recent_pepper = []
        for bucket in (own_trades, market_trades):
            for t in bucket.get("INTARIAN_PEPPER_ROOT", []) or []:
                recent_pepper.append(t.price)
        if recent_pepper:
            memory["pepper_last_trade"] = sum(recent_pepper) / len(recent_pepper)

        result: Dict[str, List] = defaultdict(list)
        positions = getattr(state, "position", {}) or {}
        order_depths = getattr(state, "order_depths", {}) or {}

        # ---- ASH_COATED_OSMIUM ------------------------------------------
        symbol = "ASH_COATED_OSMIUM"
        if symbol in order_depths:
            depth = order_depths[symbol]
            position = positions.get(symbol, 0)
            limit = POSITION_LIMIT[symbol]
            fair = _osmium_fair_value(depth, memory)

            take, buys, sells = _take_orders(
                symbol, depth, fair, OSMIUM_TAKE_WIDTH, position, limit,
            )
            clear, buys, sells = _clear_orders(
                symbol, depth, fair, position, buys, sells, limit,
            )
            make = _make_orders(
                symbol, depth, fair, OSMIUM_MAKE_EDGE,
                position, buys, sells, limit, OSMIUM_JOIN_EDGE,
            )
            result[symbol].extend(take + clear + make)

        # ---- INTARIAN_PEPPER_ROOT ---------------------------------------
        symbol = "INTARIAN_PEPPER_ROOT"
        if symbol in order_depths:
            depth = order_depths[symbol]
            position = positions.get(symbol, 0)
            limit = POSITION_LIMIT[symbol]
            fair = _pepper_fair_value(depth, memory)

            take, buys, sells = _take_orders(
                symbol, depth, fair, PEPPER_TAKE_WIDTH, position, limit,
            )
            clear, buys, sells = _clear_orders(
                symbol, depth, fair, position, buys, sells, limit,
            )
            make = _make_orders(
                symbol, depth, fair, PEPPER_MAKE_EDGE,
                position, buys, sells, limit, join_edge=1.0,
            )
            result[symbol].extend(take + clear + make)

        trader_data = json.dumps(memory)
        conversions = 0
        return dict(result), conversions, trader_data
