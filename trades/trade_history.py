from collections import defaultdict
from datetime import datetime, timezone, date
from typing import Dict, Iterable, List, Set, Tuple

import tastytrade
from tastytrade.account import Account, Transaction
from tastytrade.order import OrderAction


def get_opened_trades(
    account: Account,
    session: tastytrade.session.Session,
    start_date: datetime,
    *,
    include_same_day_round_trips: bool = True,
) -> List[Transaction]:
    """Return trades that open positions after ``start_date``.

    Args:
        account: Tastytrade account to inspect for transactions.
        session: Authenticated Tastytrade session used to fetch transaction history.
        start_date: Only transactions executed strictly after this timestamp are included.
            Naive datetimes are assumed to be in UTC.
        include_same_day_round_trips: When ``False``, trades that open and close on the
            same calendar day for the same symbol are excluded.

    Returns:
        A list of transactions executed after ``start_date`` whose action opens a position.
    """
    start_date = _ensure_aware(start_date)

    trade_history = list(
        account.get_history(
            session,
            start_date=start_date.date(),
            start_at=start_date,
        )
    )

    return _filter_trades(
        trade_history,
        start_date,
        _OPENING_ACTIONS,
        include_same_day_round_trips,
    )


def get_closed_trades(
    account: Account,
    session: tastytrade.session.Session,
    start_date: datetime,
    *,
    include_same_day_round_trips: bool = True,
) -> List[Transaction]:
    """Return trades that close positions after ``start_date``.

    Args:
        account: Tastytrade account to inspect for transactions.
        session: Authenticated Tastytrade session used to fetch transaction history.
        start_date: Only transactions executed strictly after this timestamp are included.
            Naive datetimes are assumed to be in UTC.
        include_same_day_round_trips: When ``False``, trades that open and close on the
            same calendar day for the same symbol are excluded.
    """
    start_date = _ensure_aware(start_date)

    trade_history = list(
        account.get_history(
            session,
            start_date=start_date.date(),
            start_at=start_date,
        )
    )

    return _filter_trades(
        trade_history,
        start_date,
        _CLOSING_ACTIONS,
        include_same_day_round_trips,
    )


def get_same_day_round_trips(
    account: Account,
    session: tastytrade.session.Session,
    start_date: datetime,
) -> List[Tuple[Transaction, Transaction]]:
    """Return tuples of (opening trade, closing trade) executed on the same day."""
    start_date = _ensure_aware(start_date)

    trade_history = list(
        account.get_history(
            session,
            start_date=start_date.date(),
            start_at=start_date,
        )
    )

    same_day_keys = _same_day_round_trip_keys(trade_history, start_date)

    openings_by_symbol_day: Dict[Tuple[str, date], List[Transaction]] = defaultdict(list)
    closings_by_symbol_day: Dict[Tuple[str, date], List[Transaction]] = defaultdict(list)

    for trade in trade_history:
        if trade.executed_at < start_date or trade.symbol is None:
            continue
        key = (trade.symbol, trade.executed_at.date())
        if key not in same_day_keys:
            continue
        if trade.action in _OPENING_ACTIONS:
            openings_by_symbol_day[key].append(trade)
        elif trade.action in _CLOSING_ACTIONS:
            closings_by_symbol_day[key].append(trade)

    round_trips: List[Tuple[Transaction, Transaction]] = []
    for key in same_day_keys:
        openings = openings_by_symbol_day.get(key, [])
        closings = closings_by_symbol_day.get(key, [])
        count = min(len(openings), len(closings))
        for idx in range(count):
            round_trips.append((openings[idx], closings[idx]))

    return round_trips


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _filter_trades(
    trade_history: Iterable[Transaction],
    start_date: datetime,
    actions: Set[OrderAction],
    include_same_day_round_trips: bool,
) -> List[Transaction]:
    same_day_keys: Set[Tuple[str, date]] = set()
    if not include_same_day_round_trips:
        same_day_keys = _same_day_round_trip_keys(trade_history, start_date)

    filtered: List[Transaction] = []
    for trade in trade_history:
        if trade.executed_at < start_date or trade.action not in actions:
            continue
        if not include_same_day_round_trips and trade.symbol is not None:
            key = (trade.symbol, trade.executed_at.date())
            if key in same_day_keys:
                continue
        filtered.append(trade)

    return filtered


def _same_day_round_trip_keys(
    trade_history: Iterable[Transaction],
    start_date: datetime,
) -> Set[Tuple[str, date]]:
    opening_counts: Dict[Tuple[str, date], int] = defaultdict(int)
    closing_counts: Dict[Tuple[str, date], int] = defaultdict(int)

    for trade in trade_history:
        if trade.executed_at < start_date or trade.symbol is None:
            continue
        key = (trade.symbol, trade.executed_at.date())
        if trade.action in _OPENING_ACTIONS:
            opening_counts[key] += 1
        elif trade.action in _CLOSING_ACTIONS:
            closing_counts[key] += 1

    return {
        key
        for key in opening_counts.keys() & closing_counts.keys()
        if opening_counts[key] > 0 and closing_counts[key] > 0
    }


_OPENING_ACTIONS: Set[OrderAction] = {OrderAction.SELL_TO_OPEN, OrderAction.BUY_TO_OPEN}
_CLOSING_ACTIONS: Set[OrderAction] = {OrderAction.SELL_TO_CLOSE, OrderAction.BUY_TO_CLOSE}
