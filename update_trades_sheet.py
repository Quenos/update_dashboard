"""Utilities for synchronizing OPEN and CLOSED trade sheets."""
from __future__ import annotations

import bisect
import logging
import re
from collections import OrderedDict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from tastytrade.account import Transaction
from tastytrade.order import OrderAction
from trades.trade_history import get_closed_trades, get_opened_trades

from sheets import get_workbook
from strategies import (
    MatchEntry,
    evaluate_strategy_for_group,
    load_strategy_definitions,
    normalize_option_type,
)

from utils import read_timestamp, write_timestamp


logger = logging.getLogger(__name__)


LAST_RUN_PATH = "last_run_timestamp.txt"


def ensure_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
def _try_parse_futures_option_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a futures-style option symbol.

    Expected examples:
      ./CLZ5 LOZ5  251117P55  -> ticker=CL, expiration=2025-11-17, option_type=P, strike=55
      ./ESZ5 ESZ5  251117C5000 -> ticker=ES, expiration=2025-11-17, option_type=C, strike=5000

    This function is generic for any root after './', e.g. ./CL, ./ES, ./6E, etc.
    Returns None if the format does not match.
    """
    if not symbol:
        return None

    text = symbol.strip()
    if not text.startswith("./"):
        return None

    root_match = re.match(r"^\./([A-Za-z0-9]+)", text)
    if not root_match:
        return None

    ticker_root = root_match.group(1).upper()

    # Find the expiration + option type + strike trio (e.g., 251117P55)
    trio_match = re.search(r"(\d{6})([CPcp])([0-9]+(?:\.[0-9]+)?)", text)
    if not trio_match:
        logger.debug("Futures symbol detected but no date/type/strike trio found: %s", text)
        return None

    yymmdd = trio_match.group(1)
    opt_flag = trio_match.group(2).upper()
    strike_text = trio_match.group(3)

    try:
        year = 2000 + int(yymmdd[0:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        exp = date(year, month, day)
    except ValueError:
        logger.warning("Unable to parse futures option expiration from %r in %r", yymmdd, text)
        return None

    strike_value: Optional[Decimal]
    try:
        strike_value = Decimal(strike_text)
    except InvalidOperation:
        logger.warning("Unable to parse futures option strike from %r in %r", strike_text, text)
        return None

    strike_value = strike_value.quantize(Decimal("0.001"))

    return {
        "ticker": ticker_root,
        "expiration": exp,
        "option_type": opt_flag,
        "strike": strike_value,
    }

def update_closed_trades_sheet(closed_trades: List[Transaction]) -> None:
    """Move closed trades from OPEN to CLOSED while keeping sheet formulas intact."""
    if not closed_trades:
        logger.debug("No closed trades to apply to CLOSED sheet")
        return

    workbook = get_workbook()
    open_sheet = workbook.worksheet("OPEN")
    closed_sheet = workbook.worksheet("CLOSED")
    max_columns = max(open_sheet.col_count, closed_sheet.col_count)

    open_rows, _ = _load_open_sheet_rows(open_sheet)
    symbol_to_rows = _build_open_sheet_symbol_map(open_rows)
    removed_rows: List[int] = []

    for trade in closed_trades:
        symbol = trade.symbol
        if not symbol:
            logger.warning("Encountered closed trade without OCC symbol; skipping entry")
            continue

        target_quantity = _get_trade_quantity(trade)
        remaining_quantity = target_quantity if target_quantity > 0 else Decimal(1)

        rows_info = symbol_to_rows.get(symbol)
        if not rows_info:
            logger.debug(
                "Symbol %s not found in cached OPEN rows; refreshing cache", symbol
            )
            open_rows, _ = _load_open_sheet_rows(open_sheet)
            symbol_to_rows = _build_open_sheet_symbol_map(open_rows)
            removed_rows = []
            rows_info = symbol_to_rows.get(symbol)

        if not rows_info:
            logger.warning(f"Could not locate OPEN row for closed trade symbol {symbol!r}")
            continue

        moved_quantity = Decimal(0)

        while remaining_quantity > 0 and rows_info:
            row_info = rows_info.pop(0)
            original_row = row_info['row']
            row_quantity = row_info['quantity']
            if row_quantity <= 0:
                continue

            open_row = _current_row_index(original_row, removed_rows)
            try:
                _move_row_with_formulas(
                    workbook,
                    open_sheet,
                    closed_sheet,
                    open_row,
                    destination_row=2,
                    column_count=max(1, max_columns),
                )
            except Exception as exc:
                logger.error(
                    f"Failed to move row {open_row} for symbol {symbol}: {exc}",
                    exc_info=True,
                )
                rows_info.insert(0, row_info)
                break

            _apply_closed_trade_updates(closed_sheet, trade)
            moved_quantity += row_quantity
            remaining_quantity = max(Decimal(0), remaining_quantity - row_quantity)
            bisect.insort(removed_rows, original_row)

        if not rows_info:
            symbol_to_rows.pop(symbol, None)

        if remaining_quantity > 0:
            logger.warning(
                "Closed trade for %s requires quantity %s but only matched %s",
                symbol,
                target_quantity,
                moved_quantity,
            )

    _refresh_group_borders(open_sheet)

def fetch_recent_trades(account, session, *, timestamp_path: str = LAST_RUN_PATH) -> tuple[list[Transaction], list[Transaction]]:
    """Fetch opened and closed trades based on the last run timestamp."""

    last_run = read_timestamp(timestamp_path)
    if last_run is None:
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start_time = ensure_aware_datetime(last_run)

    fetch_timestamp = datetime.now(timezone.utc)

    opened_trades = get_opened_trades(
        account,
        session,
        start_time,
        include_same_day_round_trips=False,
    )
    closed_trades = get_closed_trades(
        account,
        session,
        start_time,
        include_same_day_round_trips=False,
    )

    write_timestamp(fetch_timestamp, timestamp_path)

    return opened_trades, closed_trades


def update_opened_trades_sheet(opened_trades: List[Transaction]) -> None:
    if not opened_trades:
        logger.debug("No opened trades to process for OPEN sheet")
        return

    workbook = get_workbook()
    open_sheet = workbook.worksheet("OPEN")
    _ensure_sort_manual_row(open_sheet)

    strategies = sorted(
        load_strategy_definitions(),
        key=_strategy_leg_count,
        reverse=True,
    )
    grouped_trades: "OrderedDict[Tuple[str, str], List[Dict[str, Any]]]" = OrderedDict()

    for trade in opened_trades:
        trade_info = _parse_trade_transaction(trade)
        if not trade_info:
            continue
        key = (trade_info['group_key'], trade_info['ticker'])
        grouped_trades.setdefault(key, []).append(trade_info)

    for (group_key, ticker), trades in grouped_trades.items():
        if not trades:
            continue
        try:
            _process_trade_group(open_sheet, trades, strategies)
        except Exception as exc:
            logger.error(
                "Failed to process trade group %s for %s: %s",
                group_key,
                ticker,
                exc,
                exc_info=True,
            )

    _refresh_group_borders(open_sheet)


def _build_open_sheet_symbol_map(open_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for entry in open_rows:
        symbol = entry.get('occ_symbol')
        if not symbol:
            continue
        mapping.setdefault(symbol, []).append(entry)

    for value in mapping.values():
        value.sort(key=lambda item: item['row'])
    return mapping


def _load_open_sheet_rows(
    worksheet,
) -> Tuple[List[Dict[str, Any]], int]:
    try:
        values = worksheet.get_all_values()
        formulas = worksheet.get_all_values(value_render_option='FORMULA')
    except Exception as exc:
        logger.error("Unable to read OPEN sheet values: %s", exc, exc_info=True)
        return [], 0

    if not values:
        return [], 0

    header_len = len(values[0])
    entries: List[Dict[str, Any]] = []

    for idx, raw_row in enumerate(values, start=1):
        row_values = list(raw_row)
        if len(row_values) < header_len:
            row_values.extend([''] * (header_len - len(row_values)))

        formula_row = list(formulas[idx - 1]) if idx - 1 < len(formulas) else []
        if len(formula_row) < header_len:
            formula_row.extend([''] * (header_len - len(formula_row)))

        created_raw = row_values[0].strip() if len(row_values) > 0 else ""
        number = row_values[1].strip() if len(row_values) > 1 else ""
        action = row_values[2].strip().upper() if len(row_values) > 2 else ""
        ticker_raw = row_values[3].strip() if len(row_values) > 3 else ""
        expiration_raw = row_values[4].strip() if len(row_values) > 4 else ""
        trade_type = row_values[5].strip().upper() if len(row_values) > 5 else ""
        strike_raw = row_values[6].strip() if len(row_values) > 6 else ""

        try:
            expiration = _parse_expiration(expiration_raw, created_raw)
        except ValueError:
            expiration = None

        strike_value = _parse_decimal(strike_raw) or Decimal(0)

        entry = {
            'row': idx,
            'raw': row_values,
            'occ_symbol': _build_occ_symbol_from_row(row_values),
            'number': number,
            'action': action,
            'ticker_raw': ticker_raw,
            'ticker': ticker_raw.upper(),
            'expiration': expiration,
            'expiration_raw': expiration_raw,
            'trade_type': trade_type,
            'strike': strike_value,
            'quantity': _get_row_quantity(row_values),
            'open_date': _parse_sheet_date(created_raw),
            'formulas': formula_row,
        }

        entries.append(entry)

    return entries, header_len


def _current_row_index(original_row: int, removed_rows: List[int]) -> int:
    if not removed_rows:
        return original_row
    offset = bisect.bisect_left(removed_rows, original_row)
    return original_row - offset


def _center_row_cells(sheet, row_index: int, start_column: int = 1, end_column: int = 24) -> None:
    if row_index <= 0 or start_column > end_column:
        return
    end_column = min(end_column, sheet.col_count or end_column)
    start_column = max(1, start_column)
    try:
        sheet.spreadsheet.batch_update(
            {
                "requests": [
                    {
                        "repeatCell": {
                            "range": {
                                "sheetId": sheet.id,
                                "startRowIndex": row_index - 1,
                                "endRowIndex": row_index,
                                "startColumnIndex": start_column - 1,
                                "endColumnIndex": end_column,
                            },
                            "cell": {
                                "userEnteredFormat": {
                                    "horizontalAlignment": "CENTER"
                                }
                            },
                            "fields": "userEnteredFormat.horizontalAlignment",
                        }
                    }
                ]
            }
        )
    except Exception as exc:
        logger.error(
            f"Failed to center align row {row_index} on sheet {sheet.title}: {exc}",
            exc_info=True,
        )


def _apply_six_decimal_price_format(
    sheet,
    row_index: int,
    columns: Tuple[int, ...] = (8, 9),
) -> None:
    if row_index <= 0 or not columns:
        return

    requests: List[Dict[str, Any]] = []
    for column in columns:
        if column <= 0:
            continue
        requests.append(
            {
                "repeatCell": {
                    "range": {
                        "sheetId": sheet.id,
                        "startRowIndex": row_index - 1,
                        "endRowIndex": row_index,
                        "startColumnIndex": column - 1,
                        "endColumnIndex": column,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "NUMBER",
                                "pattern": "0.000000",
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat",
                }
            }
        )

    if not requests:
        return

    try:
        sheet.spreadsheet.batch_update({"requests": requests})
    except Exception as exc:
        logger.error(
            "Failed to apply six-decimal price format on row %s for sheet %s: %s",
            row_index,
            sheet.title,
            exc,
            exc_info=True,
        )


def _build_border_request(
    sheet_id: int,
    row_index: int,
    *,
    start_column: int,
    end_column: int,
    thick: bool,
) -> Dict[str, Any]:
    border = (
        {
            "style": "SOLID_MEDIUM",
            "color": {"red": 0, "green": 0, "blue": 0},
        }
        if thick
        else {"style": "NONE"}
    )

    return {
        "updateBorders": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": row_index - 1,
                "endRowIndex": row_index,
                "startColumnIndex": start_column - 1,
                "endColumnIndex": end_column,
            },
            "bottom": border,
        }
    }


def _set_row_bottom_border(sheet, row_index: int, *, thick: bool) -> None:
    if row_index <= 0:
        return
    end_column = min(24, sheet.col_count or 24)
    request = _build_border_request(
        sheet.id,
        row_index,
        start_column=1,
        end_column=end_column,
        thick=thick,
    )
    try:
        sheet.spreadsheet.batch_update({"requests": [request]})
    except Exception as exc:
        logger.error(
            f"Failed to update border for row {row_index} on sheet {sheet.title}: {exc}",
            exc_info=True,
        )


def _copy_row_format(sheet, source_row: int, target_row: int, column_count: int) -> None:
    if source_row <= 0 or target_row <= 0 or column_count <= 0:
        return
    request = {
        "copyPaste": {
            "source": {
                "sheetId": sheet.id,
                "startRowIndex": source_row - 1,
                "endRowIndex": source_row,
                "startColumnIndex": 0,
                "endColumnIndex": column_count,
            },
            "destination": {
                "sheetId": sheet.id,
                "startRowIndex": target_row - 1,
                "endRowIndex": target_row,
                "startColumnIndex": 0,
                "endColumnIndex": column_count,
            },
            "pasteType": "PASTE_FORMAT",
            "pasteOrientation": "NORMAL",
        }
    }
    try:
        sheet.spreadsheet.batch_update({"requests": [request]})
    except Exception as exc:
        logger.error(
            f"Failed to copy format from row {source_row} to {target_row} on sheet {sheet.title}: {exc}",
            exc_info=True,
        )


def _clear_rows(sheet, start_row: int, row_count: int, column_count: int) -> None:
    if start_row <= 0 or row_count <= 0 or column_count <= 0:
        return
    blank_row = [''] * column_count
    values = [blank_row for _ in range(row_count)]
    try:
        range_label = f"A{start_row}:{_column_letter(column_count)}{start_row + row_count - 1}"
        sheet.update(values, range_label, value_input_option="USER_ENTERED")
    except Exception as exc:
        logger.error(
            f"Failed to clear rows {start_row}-{start_row + row_count - 1} on sheet {sheet.title}: {exc}",
            exc_info=True,
        )


def _build_occ_symbol_from_row(row: List[str]) -> Optional[str]:
    try:
        created_raw = (row[0] or "").strip()
        ticker_raw = (row[3] or "").strip()
        expiration_raw = (row[4] or "").strip()
        option_type_raw = (row[5] or "").strip()
        strike_raw = (row[6] or "").strip()
    except IndexError:
        return None

    if not (ticker_raw and expiration_raw and option_type_raw and strike_raw):
        return None

    try:
        expiration_date = _parse_expiration(expiration_raw, created_raw)
    except ValueError:
        return None

    option_flag = _extract_option_flag(option_type_raw)
    if option_flag is None:
        return None

    strike_value = _parse_decimal(strike_raw)
    if strike_value is None:
        return None

    strike_formatted = f"{int(round(strike_value * 1000)):08d}"
    ticker = re.sub(r"[^A-Z0-9]", "", ticker_raw.upper())
    if not ticker:
        return None
    ticker_formatted = ticker.ljust(6)[:6]
    return f"{ticker_formatted}{expiration_date.strftime('%y%m%d')}{option_flag}{strike_formatted}"


def _parse_expiration(expiration_raw: str, created_raw: Optional[str]) -> date:
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%d-%b-%Y",
        "%d-%b-%y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(expiration_raw, fmt).date()
        except ValueError:
            continue

    default_year = _extract_year(created_raw)
    if default_year is not None:
        augmented_candidates = [
            f"{expiration_raw}-{default_year}",
            f"{expiration_raw}/{default_year}",
            f"{default_year}-{expiration_raw}",
        ]
        augmented_formats = [
            "%b-%d-%Y",
            "%b/%d/%Y",
            "%d-%b-%Y",
            "%d/%b/%Y",
            "%m-%d-%Y",
            "%m/%d/%Y",
        ]
        for candidate in augmented_candidates:
            for fmt in augmented_formats:
                try:
                    return datetime.strptime(candidate, fmt).date()
                except ValueError:
                    continue
    raise ValueError(f"Could not parse expiration value: {expiration_raw}")


def _parse_decimal(value: str) -> Optional[Decimal]:
    if "/" in value:
        value = value.split("/")[0]
    cleaned = value.replace("$", "").replace(",", "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _parse_sheet_date(value: str) -> Optional[date]:
    if not value:
        return None
    formats = [
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d",
        "%m-%d-%Y",
        "%m-%d-%y",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%b-%d-%Y",
        "%b-%d-%y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _get_row_quantity(row: List[str]) -> Decimal:
    try:
        raw = row[10]
    except IndexError:
        return Decimal(0)
    quantity = _parse_decimal(raw or "")
    return quantity.copy_abs() if quantity is not None else Decimal(0)


def _get_trade_quantity(trade: Transaction) -> Decimal:
    quantity = getattr(trade, "quantity", None)
    if quantity is None:
        return Decimal(0)
    if isinstance(quantity, Decimal):
        return quantity.copy_abs()
    try:
        return Decimal(quantity).copy_abs()
    except (InvalidOperation, TypeError):
        return Decimal(0)


def _parse_trade_transaction(trade: Transaction) -> Optional[Dict[str, Any]]:
    occ_symbol = getattr(trade, "symbol", None)
    if not occ_symbol or len(occ_symbol.strip()) < 15:
        return None

    occ_symbol = occ_symbol.strip()

    # Try parsing futures-style option symbols like:
    # "./CLZ5 LOZ5  251117P55" → ticker=CL, expiration=2025-11-17, option_type=P, strike=55
    # Works generically for roots like ./CL, ./ES, ./6E, etc.
    futures_parsed: Optional[Dict[str, Any]] = _try_parse_futures_option_symbol(occ_symbol)
    if futures_parsed is not None:
        ticker = futures_parsed["ticker"]
        expiration = futures_parsed["expiration"]
        option_type = futures_parsed["option_type"]
        strike = futures_parsed["strike"]
    else:
        try:
            ticker = occ_symbol[:6].strip()
            expiration = datetime.strptime(occ_symbol[6:12], "%y%m%d").date()
            option_type = occ_symbol[12].upper()
            strike_raw = occ_symbol[13:]
            strike = (Decimal(int(strike_raw)) / Decimal(1000)).quantize(Decimal("0.001"))
        except (ValueError, InvalidOperation):
            logger.warning(f"Unable to parse OCC symbol {occ_symbol!r}")
            return None

    action = getattr(trade, "action", None)
    action_code = _action_to_sheet_code(action)
    if not action_code:
        return None

    quantity = _get_trade_quantity(trade)
    if quantity <= 0:
        quantity = Decimal(1)

    executed_at = ensure_aware_datetime(
        getattr(trade, "executed_at", datetime.now(timezone.utc))
    )
    today = datetime.now(timezone.utc).date()
    created_days_out = None
    current_days_out = None
    if expiration:
        created_days_out = (expiration - executed_at.date()).days
        current_days_out = (expiration - today).days
    underlying_symbol = getattr(trade, "underlying_symbol", None)
    display_ticker = (underlying_symbol or ticker).strip()
    order_id = getattr(trade, "order_id", None)
    transaction_id = getattr(trade, "id", None)
    group_key = str(order_id or transaction_id or occ_symbol)

    return {
        "occ_symbol": occ_symbol,
        "ticker": display_ticker.upper(),
        "display_ticker": display_ticker,
        "expiration": expiration,
        "option_type": option_type,
        "strike": strike,
        "quantity": quantity,
        "action": action,
        "action_code": action_code,
        "order_id": order_id,
        "transaction_id": transaction_id,
        "group_key": group_key,
        "executed_at": executed_at,
        "created_days_out": created_days_out,
        "current_days_out": current_days_out,
        "open_price": _format_trade_price(
            getattr(trade, "price", None),
            action,
            symbol=occ_symbol,
        ),
    }


def _action_to_sheet_code(action: Optional[OrderAction]) -> Optional[str]:
    mapping = {
        OrderAction.BUY_TO_OPEN: "BTO",
        OrderAction.SELL_TO_OPEN: "STO",
        OrderAction.BUY_TO_CLOSE: "BTC",
        OrderAction.SELL_TO_CLOSE: "STC",
    }
    if action in mapping:
        return mapping[action]
    return None


def _group_open_rows_by_ticker_and_number(
    open_rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for entry in open_rows:
        ticker = entry.get('ticker')
        number = entry.get('number')
        if not ticker or not number:
            continue
        ticker_groups = groups.setdefault(ticker, {})
        ticker_groups.setdefault(number, []).append(entry)

    for ticker_groups in groups.values():
        for rows in ticker_groups.values():
            rows.sort(key=lambda item: item['row'])
    return groups


def _insert_manual_trade(
    open_sheet,
    trade_info: Dict[str, Any],
    *,
    strategy_label: Optional[str] = None,
) -> None:
    open_rows, column_count = _ensure_sort_manual_row(open_sheet)
    sort_row, empty_row = _find_manual_section(open_rows)
    target_row = (sort_row + 1) if sort_row else len(open_rows) + 1
    template_entry = _find_previous_entry_with_formulas(open_rows, empty_row or target_row)
    template_formulas = _extract_formula_map(template_entry) if template_entry else None
    row_offset = 0
    if template_entry:
        target_for_offset = empty_row if empty_row is not None else target_row
        row_offset = target_for_offset - template_entry['row']
    target_row_index = empty_row if empty_row is not None else target_row
    row_values = _build_open_sheet_row_values(
        trade_info,
        "",
        column_count,
        strategy_label=strategy_label,
        template_formulas=template_formulas,
        row_offset=row_offset,
        target_row_index=target_row_index,
    )

    if empty_row is None:
        open_sheet.insert_row(row_values, target_row, value_input_option="USER_ENTERED")
        _center_row_cells(open_sheet, target_row)
        if _requires_six_decimal_price(trade_info.get('occ_symbol')):
            _apply_six_decimal_price_format(open_sheet, target_row)
    else:
        open_sheet.update([row_values], _row_range(empty_row, column_count), value_input_option="USER_ENTERED")
        _center_row_cells(open_sheet, empty_row)
        if _requires_six_decimal_price(trade_info.get('occ_symbol')):
            _apply_six_decimal_price_format(open_sheet, empty_row)

def _ensure_sort_manual_row(open_sheet) -> Tuple[List[Dict[str, Any]], int]:
    while True:
        open_rows, column_count = _load_open_sheet_rows(open_sheet)
        column_count = column_count or open_sheet.col_count or 21

        sort_entry = next(
            (
                entry
                for entry in open_rows
                if entry['raw'] and entry['raw'][0].strip().lower() == 'sort manually'
            ),
            None,
        )
        if sort_entry:
            return open_rows, column_count

        next_entry = next(
            (
                entry
                for entry in open_rows
                if entry['raw'] and entry['raw'][0].strip().lower() == 'next'
            ),
            None,
        )
        if not next_entry:
            return open_rows, column_count

        last_header_entry = None
        last_data_row = None

        for entry in open_rows:
            if entry['row'] >= next_entry['row']:
                break
            row_values = entry['raw'] if entry['raw'] else []
            cell_value = row_values[0].strip() if len(row_values) > 0 else ''
            number_cell = row_values[1].strip() if len(row_values) > 1 else ''
            if (
                cell_value
                and not number_cell
                and cell_value.lower() not in {'sort manually', 'next'}
            ):
                last_header_entry = entry

        for entry in reversed(open_rows):
            if entry['row'] >= next_entry['row']:
                continue
            row_values = entry['raw'] if entry['raw'] else []
            if any(cell.strip() for cell in row_values):
                last_data_row = entry['row']
                break

        header_row = last_header_entry['row'] if last_header_entry else None
        data_row = last_data_row if last_data_row else (header_row or next_entry['row'] - 1)

        if data_row:
            gap = next_entry['row'] - data_row - 1
            rows_to_insert = max(0, 3 - gap)
            if rows_to_insert > 0:
                blank_row = [''] * column_count
                for _ in range(rows_to_insert):
                    open_sheet.insert_row(blank_row, next_entry['row'], value_input_option="USER_ENTERED")
                continue
            sort_row = data_row + 4
        else:
            sort_row = max(1, next_entry['row'] - 3)

        values = [''] * column_count
        values[0] = 'Sort manually'
        open_sheet.insert_row(values, sort_row, value_input_option="USER_ENTERED")

        if header_row:
            _copy_row_format(open_sheet, header_row, sort_row, column_count)

        if data_row:
            blank_start = data_row + 1
            blank_count = max(0, sort_row - data_row - 1)
            if blank_count > 0:
                _clear_rows(open_sheet, blank_start, blank_count, column_count)

        return _load_open_sheet_rows(open_sheet)


def _refresh_group_borders(open_sheet) -> None:
    try:
        open_rows, _ = _load_open_sheet_rows(open_sheet)
    except Exception as exc:
        logger.error(f"Unable to refresh borders on OPEN sheet: {exc}", exc_info=True)
        return

    if not open_rows:
        return

    groups = _group_open_rows_by_ticker_and_number(open_rows)
    requests: List[Dict[str, Any]] = []
    end_column = min(24, open_sheet.col_count or 24)

    for ticker_groups in groups.values():
        for number, rows in ticker_groups.items():
            if not number:
                continue
            rows_sorted = sorted(rows, key=lambda item: item['row'])
            for entry in rows_sorted[:-1]:
                requests.append(
                    _build_border_request(
                        open_sheet.id,
                        entry['row'],
                        start_column=1,
                        end_column=end_column,
                        thick=False,
                    )
                )
            if rows_sorted:
                requests.append(
                    _build_border_request(
                        open_sheet.id,
                        rows_sorted[-1]['row'],
                        start_column=1,
                        end_column=end_column,
                        thick=True,
                    )
                )

    if requests:
        try:
            open_sheet.spreadsheet.batch_update({"requests": requests})
        except Exception as exc:
            logger.error(
                f"Failed to apply group border formatting on OPEN sheet: {exc}",
                exc_info=True,
            )


def _move_row_with_formulas(
    workbook,
    source_sheet,
    destination_sheet,
    source_row: int,
    *,
    destination_row: int,
    column_count: int,
) -> None:
    requests = [
        {
            "insertDimension": {
                "range": {
                    "sheetId": destination_sheet.id,
                    "dimension": "ROWS",
                    "startIndex": destination_row - 1,
                    "endIndex": destination_row,
                },
                "inheritFromBefore": False,
            }
        },
        {
            "copyPaste": {
                "source": {
                    "sheetId": source_sheet.id,
                    "startRowIndex": source_row - 1,
                    "endRowIndex": source_row,
                    "startColumnIndex": 0,
                    "endColumnIndex": column_count,
                },
                "destination": {
                    "sheetId": destination_sheet.id,
                    "startRowIndex": destination_row - 1,
                    "endRowIndex": destination_row,
                    "startColumnIndex": 0,
                    "endColumnIndex": column_count,
                },
                "pasteType": "PASTE_FORMULA",
                "pasteOrientation": "NORMAL",
            }
        },
        {
            "deleteDimension": {
                "range": {
                    "sheetId": source_sheet.id,
                    "dimension": "ROWS",
                    "startIndex": source_row - 1,
                    "endIndex": source_row,
                }
            }
        },
    ]

    workbook.batch_update({"requests": requests})


def _apply_closed_trade_updates(sheet, trade: Transaction) -> None:
    executed_at = ensure_aware_datetime(trade.executed_at)
    close_date = executed_at.astimezone(timezone.utc).strftime("%m/%d/%Y")
    closing_price = _format_trade_price(
        trade.price,
        trade.action,
        symbol=getattr(trade, "symbol", None),
    )

    updates = {
        "O2": close_date,
        "I2": closing_price,
    }

    for cell_label, value in updates.items():
        sheet.update([[value]], cell_label, value_input_option="USER_ENTERED")

    _update_result_flag(sheet, trade)
    _center_row_cells(sheet, 2)


def _update_result_flag(sheet, trade: Transaction) -> None:
    try:
        pnl_value = sheet.cell(2, 16).value  # Column P
        strategy_value = sheet.cell(2, 21).value  # Column U
    except Exception as exc:
        logger.error(f"Unable to fetch CLOSED sheet values for win/loss flag: {exc}", exc_info=True)
        return

    pnl = _parse_decimal(pnl_value or "") or Decimal(0)
    option_type = _extract_option_type(trade.symbol)

    if (
        (strategy_value or "").strip().upper() == "PMCC"
        and trade.action == OrderAction.BUY_TO_CLOSE
        and option_type == "C"
    ):
        sheet.update([["-"]], "R2", value_input_option="USER_ENTERED")
        return

    result = "W" if pnl > 0 else "L"
    sheet.update([[result]], "R2", value_input_option="USER_ENTERED")


def _extract_option_type(occ_symbol: Optional[str]) -> Optional[str]:
    if not occ_symbol or len(occ_symbol) < 13:
        return None
    return occ_symbol[12].upper()


def _extract_year(date_raw: Optional[str]) -> Optional[int]:
    if not date_raw:
        return None
    patterns = [
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d",
        "%d-%b-%Y",
        "%d-%b-%y",
        "%b-%d-%Y",
        "%b-%d-%y",
    ]
    for fmt in patterns:
        try:
            return datetime.strptime(date_raw, fmt).year
        except ValueError:
            continue
    return None


# Symbol prefixes that require special handling
_SIX_DECIMAL_PRICE_PREFIXES: Tuple[str, ...] = ("./ZB",)
_THOUSAND_MULTIPLIER_PREFIXES: Tuple[str, ...] = ("./ZB", "./CL")


def _matches_symbol_prefix(
    symbol: Optional[str],
    prefixes: Tuple[str, ...],
) -> bool:
    if symbol is None:
        return False
    normalized = symbol.strip().upper()
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _requires_six_decimal_price(symbol: Optional[str]) -> bool:
    return _matches_symbol_prefix(symbol, _SIX_DECIMAL_PRICE_PREFIXES)


def _requires_thousand_multiplier(symbol: Optional[str]) -> bool:
    return _matches_symbol_prefix(symbol, _THOUSAND_MULTIPLIER_PREFIXES)


def _format_trade_price(
    price: Optional[Any],
    action: Optional[OrderAction],
    *,
    symbol: Optional[str] = None,
) -> str:
    if price is None:
        return ""
    try:
        price_decimal = Decimal(price)
    except (InvalidOperation, TypeError):
        logger.warning(f"Unable to parse trade price value: {price}")
        return ""

    price_decimal = price_decimal.copy_abs()
    if action in {OrderAction.BUY_TO_CLOSE, OrderAction.BUY_TO_OPEN}:
        price_decimal = -price_decimal

    decimal_format = "{0:.6f}" if _requires_six_decimal_price(symbol) else "{0:.2f}"
    return decimal_format.format(price_decimal)


def _format_expiration_for_sheet(expiration: Optional[date]) -> str:
    if not expiration:
        return ""
    return expiration.strftime("%b-%d")


def _format_strike_for_sheet(strike: Decimal) -> str:
    if strike == strike.to_integral():
        return str(int(strike))
    return format(strike.normalize(), 'f').rstrip('0').rstrip('.')


def _format_quantity_for_sheet(quantity: Decimal) -> str:
    if quantity == quantity.to_integral():
        return str(int(quantity))
    return format(quantity.normalize(), 'f')


def _column_letter(column_index: int) -> str:
    column_index = max(1, column_index)
    letters = ""
    while column_index > 0:
        column_index, remainder = divmod(column_index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _row_range(row_index: int, column_count: int) -> str:
    last_column = _column_letter(column_count)
    return f"A{row_index}:{last_column}{row_index}"


_FORMULA_CELL_PATTERN = re.compile(r'(\$?[A-Z]{1,3})(\$?)(\d+)')


def _shift_formula_rows(formula: str, row_offset: int) -> str:
    if not formula or row_offset == 0:
        return formula

    def repl(match: re.Match[str]) -> str:
        column_part, row_abs_flag, row_digits = match.groups()
        if row_abs_flag == '$':
            return match.group(0)
        new_row = int(row_digits) + row_offset
        if new_row < 1:
            new_row = 1
        return f"{column_part}{row_abs_flag}{new_row}"

    return _FORMULA_CELL_PATTERN.sub(repl, formula)


def _find_manual_section(open_rows: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    sort_row = None
    empty_row = None

    for entry in open_rows:
        row_index = entry['row']
        row_values = entry['raw'] if 'raw' in entry else []
        if row_values and row_values[0].strip().lower() == 'sort manually':
            sort_row = row_index
            continue
        if sort_row and row_index > sort_row:
            if not any(cell.strip() for cell in row_values):
                empty_row = row_index
                break

    return sort_row, empty_row


def _process_trade_group(
    open_sheet,
    trade_group: List[Dict[str, Any]],
    strategies: List[Dict[str, Any]],
) -> None:
    if not trade_group:
        return

    open_rows, column_count = _load_open_sheet_rows(open_sheet)
    if column_count <= 0:
        column_count = max(open_sheet.col_count or 21, 21)

    occ_symbols_on_sheet = {
        entry.get('occ_symbol')
        for entry in open_rows
        if entry.get('occ_symbol')
    }

    if any(
        trade.get('occ_symbol') in occ_symbols_on_sheet
        for trade in trade_group
        if trade.get('occ_symbol')
    ):
        group_key = trade_group[0].get('group_key')
        logger.debug("Trade group %s already present on OPEN sheet", group_key)
        return

    tickers = {trade['ticker'] for trade in trade_group}
    if len(tickers) != 1:
        logger.warning(
            "Trade group %s contains multiple tickers %s; inserting manually",
            trade_group[0].get('group_key'),
            sorted(tickers),
        )
        for trade in trade_group:
            _insert_manual_trade(open_sheet, trade)
        return

    ticker = next(iter(tickers))
    groups = _group_open_rows_by_ticker_and_number(open_rows)
    groups_for_ticker = groups.get(ticker, {})
    today = datetime.now(timezone.utc).date()

    for strategy in strategies:
        requires_existing = strategy.get('requires_existing_group', True)
        candidate_groups = (
            _iter_candidate_groups(groups_for_ticker)
            if requires_existing
            else [(None, [])]
        )

        for number, rows in candidate_groups:
            result = evaluate_strategy_for_group(
                strategy,
                trade_group,
                rows,
                today=today,
            )
            if not result:
                continue

            ordered_trades = _ordered_new_leg_trades(strategy, result.new_assignments)
            if len(ordered_trades) != len(trade_group):
                continue

            if requires_existing:
                if number is None or not rows:
                    continue
                _insert_trades_into_existing_group(
                    open_sheet,
                    ordered_trades,
                    strategy,
                    str(number),
                    rows,
                    column_count,
                )
            else:
                _insert_trades_without_group(
                    open_sheet,
                    ordered_trades,
                    strategy,
                )
            return

    for trade in trade_group:
        _insert_manual_trade(open_sheet, trade)

def _iter_candidate_groups(
    groups_for_ticker: Optional[Dict[str, List[Dict[str, Any]]]]
) -> List[Tuple[Optional[str], List[Dict[str, Any]]]]:
    if not groups_for_ticker:
        return []

    candidates: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    for number, rows in groups_for_ticker.items():
        if not rows:
            continue
        sorted_rows = sorted(rows, key=lambda item: item['row'])
        first_row = sorted_rows[0]['row']
        candidates.append((first_row, number, sorted_rows))

    candidates.sort(key=lambda item: item[0])
    return [(str(number), rows) for _, number, rows in candidates]

def _ordered_new_leg_trades(
    strategy: Dict[str, Any],
    new_assignments: Dict[str, MatchEntry],
) -> List[Dict[str, Any]]:
    ordered: List[Dict[str, Any]] = []
    for leg in strategy.get('new_legs', []) or []:
        role = leg.get('role')
        if not role:
            continue
        entry = new_assignments.get(role)
        if entry is None:
            return []
        ordered.append(entry.data)
    return ordered

def _insert_trades_into_existing_group(
    open_sheet,
    trades: List[Dict[str, Any]],
    strategy: Dict[str, Any],
    number: str,
    group_rows: List[Dict[str, Any]],
    column_count: int,
) -> None:
    if not trades or not group_rows:
        return

    rows_sorted = sorted(group_rows, key=lambda item: item['row'])
    template_entry = rows_sorted[-1]
    template_formulas = _extract_formula_map(template_entry)
    insert_index = template_entry['row'] + 1

    _set_row_bottom_border(open_sheet, template_entry['row'], thick=False)

    for idx, trade_info in enumerate(trades):
        row_offset = insert_index - template_entry['row']
        row_values = _build_open_sheet_row_values(
            trade_info,
            number,
            column_count,
            strategy_label=strategy.get('group_label'),
            template_formulas=template_formulas,
            row_offset=row_offset,
            target_row_index=insert_index,
        )
        open_sheet.insert_row(row_values, insert_index, value_input_option="USER_ENTERED")
        _center_row_cells(open_sheet, insert_index)
        if idx < len(trades) - 1:
            _set_row_bottom_border(open_sheet, insert_index, thick=False)
        if _requires_six_decimal_price(trade_info.get('occ_symbol')):
            _apply_six_decimal_price_format(open_sheet, insert_index)
        insert_index += 1

    _set_row_bottom_border(open_sheet, insert_index - 1, thick=True)

def _insert_trades_without_group(
    open_sheet,
    trades: List[Dict[str, Any]],
    strategy: Dict[str, Any],
) -> None:
    header_text = (strategy.get('sheet_header') or '').strip()
    if header_text:
        if _insert_trades_under_header(
            open_sheet,
            trades,
            strategy,
            header_text,
        ):
            return

    for trade_info in trades:
        _insert_manual_trade(
            open_sheet,
            trade_info,
            strategy_label=strategy.get('group_label'),
        )

def _insert_trades_under_header(
    open_sheet,
    trades: List[Dict[str, Any]],
    strategy: Dict[str, Any],
    header_text: str,
) -> bool:
    if not trades:
        return True

    open_rows, column_count = _load_open_sheet_rows(open_sheet)
    column_count = column_count or open_sheet.col_count or 21

    header_entry = _find_header_entry(open_rows, header_text)
    next_identifier = _find_next_identifier(open_rows)
    template_override: Optional[Dict[str, Any]] = None
    if not header_entry:
        header_entry, open_rows, template_override = _create_header_before_sort_manual(
            open_sheet,
            header_text,
            open_rows,
            column_count,
        )
        if not header_entry:
            logger.warning(
                "Unable to locate or create header %r on OPEN sheet for strategy %s; falling back to manual insertion",
                header_text,
                strategy.get('name') or strategy.get('group_label') or 'unknown',
            )
            return False
        next_identifier = _find_next_identifier(open_rows)

    template_entry = template_override or _find_first_data_entry(open_rows)
    if template_entry is None:
        updated_rows, _ = _load_open_sheet_rows(open_sheet)
        open_rows = updated_rows
        template_entry = _find_first_data_entry(updated_rows)
        next_identifier = _find_next_identifier(open_rows)

    template_formulas = _extract_formula_map(template_entry) if template_entry else None

    insert_row = header_entry['row'] + 1

    for trade_info in trades:
        row_offset = 0
        if template_entry:
            row_offset = insert_row - template_entry['row']
        row_values = _build_open_sheet_row_values(
            trade_info,
            next_identifier,
            column_count,
            strategy_label=strategy.get('group_label'),
            template_formulas=template_formulas,
            row_offset=row_offset,
            target_row_index=insert_row,
        )
        open_sheet.insert_row(row_values, insert_row, value_input_option="USER_ENTERED")
        if template_entry:
            _copy_row_format(open_sheet, template_entry['row'], insert_row, column_count)
        _center_row_cells(open_sheet, insert_row)
        if _requires_six_decimal_price(trade_info.get('occ_symbol')):
            _apply_six_decimal_price_format(open_sheet, insert_row)
        insert_row += 1

    if trades:
        _set_row_bottom_border(open_sheet, insert_row - 1, thick=True)
        # Add a single blank row after the entire strategy group (between strategies)
        blank_row_values = [''] * column_count
        open_sheet.insert_row(blank_row_values, insert_row, value_input_option="USER_ENTERED")

    return True


def _strategy_leg_count(strategy: Dict[str, Any]) -> int:
    new_legs = strategy.get('new_legs') or []
    existing_legs = strategy.get('existing_legs') or []
    return len(new_legs) + len(existing_legs)


def _find_header_entry(
    open_rows: List[Dict[str, Any]],
    header_text: str,
) -> Optional[Dict[str, Any]]:
    header_normalized = header_text.strip().lower()
    if not header_normalized:
        return None

    for entry in open_rows:
        row_values = entry.get('raw') or []
        if not row_values:
            continue
        if (row_values[0] or '').strip().lower() == header_normalized:
            return entry
    return None


def _find_template_for_header(
    open_rows: List[Dict[str, Any]],
    header_row: int,
) -> Optional[Dict[str, Any]]:
    for entry in open_rows:
        if entry['row'] <= header_row:
            continue
        if _extract_formula_map(entry):
            return entry
    return _find_previous_entry_with_formulas(open_rows, header_row)


def _create_header_before_sort_manual(
    open_sheet,
    header_text: str,
    open_rows: List[Dict[str, Any]],
    column_count: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    sort_entry = next(
        (
            entry
            for entry in open_rows
            if entry['raw'] and entry['raw'][0].strip().lower() == 'sort manually'
        ),
        None,
    )
    if not sort_entry:
        return None, open_rows, None

    sort_row = sort_entry['row']
    header_template_entry = _find_previous_header_entry(open_rows, sort_row)
    values = [''] * column_count
    values[0] = header_text.strip()
    open_sheet.insert_row(values, sort_row, value_input_option="USER_ENTERED")

    if header_template_entry:
        _copy_row_format(open_sheet, header_template_entry['row'], sort_row, column_count)
    else:
        template_entry = _find_previous_entry_with_formulas(open_rows, sort_row)
        if template_entry:
            _copy_row_format(open_sheet, template_entry['row'], sort_row, column_count)

    updated_rows, _ = _load_open_sheet_rows(open_sheet)
    header_entry = _find_header_entry(updated_rows, header_text)
    template_entry = None
    if header_entry:
        template_entry = _find_first_data_entry(updated_rows)
    return header_entry, updated_rows, template_entry


def _find_previous_header_entry(
    open_rows: List[Dict[str, Any]],
    before_row: int,
) -> Optional[Dict[str, Any]]:
    for entry in reversed(open_rows):
        if entry['row'] >= before_row:
            continue
        row_values = entry.get('raw') or []
        if not row_values:
            continue
        label = (row_values[0] or '').strip()
        number = (row_values[1] or '').strip() if len(row_values) > 1 else ''
        if label and not number:
            return entry
    return None


def _find_first_data_entry_after(
    open_rows: List[Dict[str, Any]],
    header_row: int,
) -> Optional[Dict[str, Any]]:
    for entry in open_rows:
        if entry['row'] <= header_row:
            continue
        open_date = entry.get('open_date')
        if isinstance(open_date, date):
            return entry
    return None


def _find_first_data_entry(
    open_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for entry in open_rows:
        open_date = entry.get('open_date')
        if isinstance(open_date, date):
            return entry
    return None


def _find_next_identifier(open_rows: List[Dict[str, Any]]) -> str:
    for entry in open_rows:
        raw = entry.get('raw') or []
        if not raw:
            continue
        if raw[0].strip().lower() == 'next':
            if len(raw) > 2 and raw[2]:
                return raw[2]
            cell = entry.get('formulas') or []
            if len(cell) > 2 and cell[2] and not cell[2].startswith('='):
                return cell[2]
            return ''
    return ''


def _build_open_sheet_row_values(
    trade_info: Dict[str, Any],
    number_value: str,
    column_count: int,
    *,
    strategy_label: Optional[str] = None,
    template_formulas: Optional[Dict[int, str]] = None,
    row_offset: int = 0,
    target_row_index: Optional[int] = None,
) -> List[str]:
    values = [''] * column_count
    executed_at = trade_info['executed_at'].astimezone(timezone.utc)
    values[0] = executed_at.strftime('%m/%d/%y')
    identifier = number_value or str(trade_info.get('order_id') or trade_info.get('transaction_id') or '')
    if column_count > 1:
        values[1] = identifier
    if column_count > 2:
        values[2] = trade_info['action_code']
    if column_count > 3:
        values[3] = trade_info['display_ticker']
    if column_count > 4:
        values[4] = _format_expiration_for_sheet(trade_info.get('expiration'))
    if column_count > 5:
        opt_type = normalize_option_type(trade_info.get('option_type'))
        if opt_type == 'CALL':
            values[5] = 'CALL'
        elif opt_type == 'PUT':
            values[5] = 'PUT'
        else:
            values[5] = ''
    if column_count > 6:
        values[6] = _format_strike_for_sheet(trade_info.get('strike', Decimal(0)))
    if column_count > 7:
        values[7] = trade_info.get('open_price', '')
    if column_count > 10:
        values[10] = _format_quantity_for_sheet(trade_info.get('quantity', Decimal(1)))
    if strategy_label and column_count > 20:
        values[20] = strategy_label

    if template_formulas:
        for idx, formula in template_formulas.items():
            if 0 <= idx < column_count and not values[idx]:
                adjusted = _shift_formula_rows(formula, row_offset)
                values[idx] = adjusted

    occ_symbol = trade_info.get('occ_symbol')
    thousand_multiplier_required = _requires_thousand_multiplier(occ_symbol)

    if target_row_index is not None and thousand_multiplier_required:
        if column_count > 12:
            values[12] = f"=H{target_row_index}*K{target_row_index}*1000"
        if column_count > 15:
            values[15] = f"=J{target_row_index}*K{target_row_index}*1000"
    return values


def _extract_formula_map(entry: Dict[str, Any]) -> Dict[int, str]:
    formula_row = entry.get('formulas') or []
    mapping: Dict[int, str] = {}
    for idx, cell in enumerate(formula_row):
        if isinstance(cell, str) and cell.startswith('='):
            mapping[idx] = cell
    return mapping


def _find_previous_entry_with_formulas(
    open_rows: List[Dict[str, Any]],
    before_row: Optional[int],
) -> Optional[Dict[str, Any]]:
    threshold = before_row if before_row is not None else float('inf')
    for entry in reversed(open_rows):
        if entry['row'] >= threshold:
            continue
        if _extract_formula_map(entry):
            return entry
    return None


def _extract_option_flag(option_type_raw: str) -> Optional[str]:
    text = option_type_raw.strip().upper()
    if not text:
        return None
    if text in {"CALL", "PUT"}:
        return text[0]
    if text.endswith("C"):
        return "C"
    if text.endswith("P"):
        return "P"
    if text.startswith("C"):
        return "C"
    if text.startswith("P"):
        return "P"
    return None


__all__ = [
    "ensure_aware_datetime",
    "fetch_recent_trades",
    "update_closed_trades_sheet",
    "update_opened_trades_sheet",
]
