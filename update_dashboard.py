import bisect
import json
import logging
import logging.handlers
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pandas_market_calendars as mcal
from tastytrade.account import Transaction
from tastytrade.instruments import FutureOption, Option
from tastytrade.metrics import get_market_metrics
from tastytrade.order import InstrumentType, OrderAction
from tastytrade.session import Session

from account import Account
from config import GREEK_POLL_INTERVAL, LOG_LEVEL
from market_data import MarketData
from session import ApplicationSession
from sheets import get_row_data, get_workbook, insert_row_data
from trades.trade_history import get_closed_trades, get_opened_trades


# Configure logging
log_level = getattr(logging, LOG_LEVEL, logging.INFO)

# Configure rotating file handler (new log file each run, keep 5 backups)
file_handler = logging.handlers.RotatingFileHandler(
    'update_dashboard.log',
    maxBytes=1024*1024*10,  # 10MB max size (but we'll rotate on each run)
    backupCount=5
)

# Force rotation at startup to create a new log file for this run
file_handler.doRollover()

# Configure console handler
console_handler = logging.StreamHandler()

# Set formatter for both handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Set all packages to ERROR level to reduce noise
logging.getLogger().setLevel(logging.ERROR)

# Explicitly set common noisy packages to ERROR level
noisy_packages = [
    'tastytrade',
    'httpx', 
    'httpcore',
    'urllib3',
    'requests',
    'pandas',
    'gspread',
    'google',
    'googleapiclient',
    'oauth2client',
    'websockets',
    'asyncio'
]

for package in noisy_packages:
    logging.getLogger(package).setLevel(logging.ERROR)

# Set only __main__ (our application) to the configured log level
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# Log that rotation happened and new log file started
logger.info("="*60)
logger.info("NEW LOG SESSION STARTED - Previous logs rotated to backup files")
logger.info("="*60)

market_data: MarketData | None = None
SPY_SYMBOL = 'SPY'


def format_end_of_week_row(worksheet, row: int) -> None:
    """
    Formats the end of week row in the worksheet.

    :param worksheet: The worksheet to format.
    :param row: The row number to format.
    """
    logger.debug(f"Formatting end of week row {row}")
    light_yellow = {
        'backgroundColor': {
            'red': 1.0,
            'green': 0.95,
            'blue': 0.8
        }
    }
    logger.debug(f"Applying light yellow formatting to range A{row}:P{row}")
    worksheet.format(f'A{row}:P{row}', light_yellow)
    logger.debug("End of week row formatting completed")


def get_first_empty_row(worksheet, start_row=2) -> int:
    """
    Returns the first empty row in the given worksheet by fetching column A
    in one call and then checking in memory.
    """
    logger.debug(f"Finding first empty row starting from row {start_row}")
    # Fetch all values in the first column at once
    col_values = worksheet.col_values(1)
    logger.debug(f"Column A has {len(col_values)} values")

    # Iterate over the values in column A
    for idx, value in enumerate(col_values, start=1):
        if idx >= start_row and not value:
            logger.debug(f"Found empty cell at row {idx}")
            return idx

    # If no empty cell was found, it's the row after the last value
    empty_row = len(col_values) + 1
    logger.debug(f"No empty cells found, returning row after last value: {empty_row}")
    return empty_row


def update_dashboard() -> None:
    """
    Updates the dashboard sheet in the tracker workbook with new data.
    """
    logger.debug("Starting update_dashboard function")
    try:
        logger.debug("Creating application session")
        session = ApplicationSession().session
        logger.debug(f"Session created successfully: {type(session)}")
        
        logger.debug("Getting account information")
        account = Account(session).account
        logger.debug(f"Account retrieved: {account.account_number if hasattr(account, 'account_number') else 'Unknown'}")

        logger.debug("Fetching portfolio data")
        portfolio_data = get_portfolio_data(session, account)
        logger.debug(f"Portfolio data retrieved: {len(portfolio_data) if portfolio_data else 0} items")
        
        logger.debug("Fetching balance data")
        balance_data = get_balance_data(session, account)
        logger.debug(f"Balance data retrieved: {len(balance_data) if balance_data else 0} items")
        
        logger.debug("Fetching margin data")
        margin_data = get_margin_data(session, account)
        logger.debug(f"Margin data retrieved: {len(margin_data) if margin_data else 0} items")

        logger.debug("Updating dashboard sheet with collected data")
        update_dashboard_sheet(portfolio_data, balance_data, margin_data)
        logger.debug("update_dashboard function completed successfully")
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}", exc_info=True)
        raise  # Re-raise the exception after logging


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
                # put row info back for potential retry on refresh
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
                continue  # reload sheet state after insertion
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

    executed_at = _ensure_aware_datetime(getattr(trade, "executed_at", datetime.now(timezone.utc)))
    underlying_symbol = getattr(trade, "underlying_symbol", None)
    display_ticker = (underlying_symbol or ticker).strip()

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
        "executed_at": executed_at,
        "open_price": _format_trade_price(getattr(trade, "price", None), action),
    }


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


def _group_matches_pmcc(rows: List[Dict[str, Any]]) -> bool:
    bto_calls = [r for r in rows if r['action'] == 'BTO' and r['trade_type'] == 'CALL']
    sto_puts = [r for r in rows if r['action'] == 'STO' and r['trade_type'] == 'PUT']
    bto_puts = [r for r in rows if r['action'] == 'BTO' and r['trade_type'] == 'PUT']

    for call_entry in bto_calls:
        if not call_entry['expiration'] or call_entry['strike'] <= 0:
            continue
        for sto_put in sto_puts:
            if (
                sto_put['expiration'] == call_entry['expiration']
                and sto_put['strike'] == call_entry['strike']
            ):
                for bto_put in bto_puts:
                    if (
                        bto_put['expiration'] == call_entry['expiration']
                        and bto_put['strike'] < call_entry['strike']
                        and bto_put['strike'] > 0
                    ):
                        return True
    return False


def _find_pmcc_group(
    groups_for_ticker: Optional[Dict[str, List[Dict[str, Any]]]]
) -> Optional[Tuple[int, str, Dict[str, Any]]]:
    if not groups_for_ticker:
        return None

    for number, rows in groups_for_ticker.items():
        if _group_matches_pmcc(rows):
            insert_index = max(r['row'] for r in rows) + 1
            template_entry = max(rows, key=lambda item: item['row'])
            return insert_index, number, template_entry
    return None


def _find_leap_group(
    groups_for_ticker: Optional[Dict[str, List[Dict[str, Any]]]]
) -> Optional[Tuple[int, str, Dict[str, Any]]]:
    if not groups_for_ticker:
        return None

    for number, rows in groups_for_ticker.items():
        for entry in rows:
            if (
                entry['action'] == 'BTO'
                and entry['trade_type'] == 'CALL'
                and entry['open_date']
                and entry['expiration']
                and (entry['expiration'] - entry['open_date']).days >= 170
            ):
                insert_index = max(r['row'] for r in rows) + 1
                template_entry = max(rows, key=lambda item: item['row'])
                return insert_index, number, template_entry
    return None


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
    executed_at = _ensure_aware_datetime(trade.executed_at)
    close_date = executed_at.astimezone(timezone.utc).strftime("%m/%d/%Y")
    closing_price = _format_trade_price(trade.price, trade.action)

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


def _ensure_aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


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


def _format_trade_price(price: Optional[Any], action: Optional[OrderAction]) -> str:
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

    return f"{price_decimal:.2f}"


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


def update_opened_trades_sheet(opened_trades: List[Transaction]) -> None:
    if not opened_trades:
        logger.debug("No opened trades to process for OPEN sheet")
        return

    workbook = get_workbook()
    open_sheet = workbook.worksheet("OPEN")
    _ensure_sort_manual_row(open_sheet)

    for trade in opened_trades:
        trade_info = _parse_trade_transaction(trade)
        if not trade_info:
            continue
        try:
            _process_open_trade(open_sheet, trade_info)
        except Exception as exc:
            logger.error(
                f"Failed to process opened trade {trade.symbol}: {exc}",
                exc_info=True,
            )

    _refresh_group_borders(open_sheet)


def _process_open_trade(open_sheet, trade_info: Dict[str, Any]) -> None:
    open_rows, column_count = _load_open_sheet_rows(open_sheet)
    if column_count <= 0:
        column_count = max(open_sheet.col_count, 21)

    if trade_info['occ_symbol'] and any(
        entry.get('occ_symbol') == trade_info['occ_symbol']
        for entry in open_rows
        if entry.get('occ_symbol')
    ):
        logger.debug("Trade %s already present on OPEN sheet", trade_info['occ_symbol'])
        return

    groups = _group_open_rows_by_ticker_and_number(open_rows)
    groups_for_ticker = groups.get(trade_info['ticker'])

    today = datetime.now(timezone.utc).date()
    pmcc_applicable = _is_pmcc_candidate(trade_info, today)

    if pmcc_applicable:
        pmcc_group = _find_pmcc_group(groups_for_ticker)
        if pmcc_group:
            insert_index, number, template_entry = pmcc_group
            template_formulas = _extract_formula_map(template_entry)
            row_offset = insert_index - template_entry['row']
            row_values = _build_open_sheet_row_values(
                trade_info,
                number,
                column_count,
                True,
                template_formulas,
                row_offset=row_offset,
            )
            open_sheet.insert_row(row_values, insert_index, value_input_option="USER_ENTERED")
            _center_row_cells(open_sheet, insert_index)
            if template_entry:
                _set_row_bottom_border(open_sheet, template_entry['row'], thick=False)
            _set_row_bottom_border(open_sheet, insert_index, thick=True)
            return

        leap_group = _find_leap_group(groups_for_ticker)
        if leap_group:
            insert_index, number, template_entry = leap_group
            template_formulas = _extract_formula_map(template_entry)
            row_offset = insert_index - template_entry['row']
            row_values = _build_open_sheet_row_values(
                trade_info,
                number,
                column_count,
                True,
                template_formulas,
                row_offset=row_offset,
            )
            open_sheet.insert_row(row_values, insert_index, value_input_option="USER_ENTERED")
            _center_row_cells(open_sheet, insert_index)
            if template_entry:
                _set_row_bottom_border(open_sheet, template_entry['row'], thick=False)
            _set_row_bottom_border(open_sheet, insert_index, thick=True)
            return

    sort_row, empty_row = _find_manual_section(open_rows)
    target_row = (sort_row + 1) if sort_row else len(open_rows) + 1
    template_entry = _find_previous_entry_with_formulas(open_rows, empty_row or target_row)
    template_formulas = _extract_formula_map(template_entry) if template_entry else None
    row_offset = 0
    if template_entry:
        target_for_offset = empty_row if empty_row is not None else target_row
        row_offset = target_for_offset - template_entry['row']
    row_values = _build_open_sheet_row_values(
        trade_info,
        "",
        column_count,
        False,
        template_formulas,
        row_offset=row_offset,
    )

    if empty_row is None:
        open_sheet.insert_row(row_values, target_row, value_input_option="USER_ENTERED")
        _center_row_cells(open_sheet, target_row)
    else:
        open_sheet.update([row_values], _row_range(empty_row, column_count), value_input_option="USER_ENTERED")
        _center_row_cells(open_sheet, empty_row)


def _is_pmcc_candidate(trade_info: Dict[str, Any], today: date) -> bool:
    if trade_info.get('action') != OrderAction.SELL_TO_OPEN:
        return False
    if trade_info.get('option_type') != 'C':
        return False
    expiration = trade_info.get('expiration')
    if not expiration:
        return False
    days_out = (expiration - today).days
    return 0 <= days_out <= 50


def _build_open_sheet_row_values(
    trade_info: Dict[str, Any],
    number_value: str,
    column_count: int,
    is_pmcc: bool,
    template_formulas: Optional[Dict[int, str]] = None,
    *,
    row_offset: int = 0,
) -> List[str]:
    values = [''] * column_count
    executed_at = trade_info['executed_at'].astimezone(timezone.utc)
    values[0] = executed_at.strftime('%m/%d/%y')
    if column_count > 1:
        values[1] = number_value
    if column_count > 2:
        values[2] = trade_info['action_code']
    if column_count > 3:
        values[3] = trade_info['display_ticker']
    if column_count > 4:
        values[4] = _format_expiration_for_sheet(trade_info.get('expiration'))
    if column_count > 5:
        values[5] = 'CALL' if trade_info.get('option_type') == 'C' else 'PUT'
    if column_count > 6:
        values[6] = _format_strike_for_sheet(trade_info.get('strike', Decimal(0)))
    if column_count > 7:
        values[7] = trade_info.get('open_price', '')
    if column_count > 10:
        values[10] = _format_quantity_for_sheet(trade_info.get('quantity', Decimal(1)))
    if is_pmcc and column_count > 20:
        values[20] = 'PMCC'

    if template_formulas:
        for idx, formula in template_formulas.items():
            if 0 <= idx < column_count and not values[idx]:
                adjusted = _shift_formula_rows(formula, row_offset)
                values[idx] = adjusted
    return values


def get_portfolio_data(session: Session, account: Account) -> Dict[str, int]:
    """
    Retrieves portfolio data including beta-weighted delta and theta.
    """
    logger.debug("Starting get_portfolio_data function")
    portfolio_beta_weighted_delta, portfolio_theta, theta_vega_ratio = get_portfolio_metrics(session, account)
    logger.debug(f"Portfolio metrics calculated - BWD: {portfolio_beta_weighted_delta}, Theta: {portfolio_theta}, T/V Ratio: {theta_vega_ratio}")
    
    result = {
        'beta_weighted_delta': round(portfolio_beta_weighted_delta),
        'theta': round(portfolio_theta),
        'theta_vega_ratio': theta_vega_ratio
    }
    logger.debug(f"get_portfolio_data returning: {result}")
    return result


def get_balance_data(session: Session, account: Account) -> Dict[str, int]:
    """
    Retrieves account balance data.
    """
    logger.debug("Starting get_balance_data function")
    balance = account.get_balances(session)
    logger.debug(f"Balance object retrieved: {type(balance)}")
    logger.debug(f"Net liquidating value: {balance.net_liquidating_value}")
    
    result = {
        'net_liquidating_value': int(balance.net_liquidating_value)
    }
    logger.debug(f"get_balance_data returning: {result}")
    return result


def get_margin_data(session: Session, account: Account) -> Dict[str, Any]:
    """
    Retrieves margin requirement data.
    """
    logger.debug("Starting get_margin_data function")
    margin_requirement_report = account.get_margin_requirements(session)
    logger.debug(f"Margin requirement report retrieved with {len(margin_requirement_report.groups)} groups")
    
    margin_requirement = 0
    bil = 0
    for i, margin_report in enumerate(margin_requirement_report.groups):
        logger.debug(f"Processing margin group {i}: {margin_report.description}, Buying power: {margin_report.buying_power}")
        margin_requirement += margin_report.buying_power
        if margin_report.description == 'BIL':
            bil = margin_report.buying_power
            logger.debug(f"Found BIL group with buying power: {bil}")

    logger.debug(f"Total margin requirement: {margin_requirement}, BIL: {bil}")
    result = {
        'margin_requirement': int(margin_requirement),
        'bil': int(bil)
    }
    logger.debug(f"get_margin_data returning: {result}")
    return result


def get_portfolio_metrics(session: Session, account: Account) -> Tuple[Decimal, Decimal]:
    """
    Calculates portfolio beta-weighted delta and theta.
    """
    logger.debug("Starting get_portfolio_metrics function")
    
    logger.debug("Getting filtered positions")
    positions = get_filtered_positions(account, session)
    logger.debug(f"Retrieved {len(positions)} filtered positions")
    
    logger.debug("Getting beta data")
    betas = get_betas(session)
    logger.debug(f"Retrieved beta data for {len(betas)} symbols")
    
    logger.debug("Creating symbol map")
    symbol_map = get_symbol_map(session, positions)
    logger.debug(f"Created symbol map with {len(symbol_map)} entries")
    
    logger.debug("Adding current prices to symbol map")
    symbol_map, price_spy = add_current_prices(session, symbol_map, betas)
    logger.debug(f"Added current prices, SPY price: {price_spy}")
    
    logger.debug("Getting Greeks data")
    greeks = get_greeks(symbol_map)
    logger.debug(f"Retrieved Greeks for {len(greeks)} symbols")
    
    logger.debug("Adding ETF Greeks to symbol map")
    symbol_map = add_etf_greeks(symbol_map, greeks, betas)
    logger.debug("ETF Greeks added to symbol map")
    
    logger.debug("Adding SPY beta delta to symbol map")
    symbol_map = add_spy_beta_delta(symbol_map, greeks, betas, price_spy)
    logger.debug("SPY beta delta added to symbol map")

    logger.debug("Calculating portfolio totals")
    portfolio_beta_weighted_delta = sum(
        Decimal(x.get('beta_weighted_delta', 0)) for x in symbol_map)
    portfolio_theta = sum(Decimal(x.get('theta', 0)) for x in symbol_map)
    portfolio_vega = sum(Decimal(x.get('vega', 0)) for x in symbol_map)
    
    logger.debug(f"Portfolio BWD: {portfolio_beta_weighted_delta}, Theta: {portfolio_theta}, Vega: {portfolio_vega}")
    
    # Handle division by zero for theta/vega ratio
    if portfolio_theta != 0:
        theta_vega_ratio = f'1 : {round(portfolio_vega/portfolio_theta)}'
        logger.debug(f"Theta/Vega ratio calculated: {theta_vega_ratio}")
    else:
        theta_vega_ratio = '1 : 0'
        logger.warning("Portfolio theta is zero, setting theta/vega ratio to '1 : 0'")

    logger.debug("get_portfolio_metrics function completed")
    return portfolio_beta_weighted_delta, portfolio_theta, theta_vega_ratio


def get_filtered_positions(account: Account, session: Session) -> List[Any]:
    """
    Retrieves positions, filtering out equity positions.
    """
    positions = account.get_positions(session)
    return [x for x in positions if x.instrument_type != InstrumentType.EQUITY and x.instrument_type != InstrumentType.CRYPTOCURRENCY]


def get_betas(session: Session) -> Dict[str, Dict[str, str]]:
    """
    Get the beta data for the symbols in the beta.json file.
    """
    try:
        with open("beta.json", "r") as f:
            betas = json.load(f)

        symbols = list(set(value['ETF'] for value in betas.values()))
        metrics = get_market_metrics(session, symbols)
        for value in betas.values():
            for item in metrics:
                if value['ETF'] == item.symbol:
                    value['SPY_BETA'] = str(item.beta)
        return betas
    except FileNotFoundError:
        logger.error("beta.json file not found")
        raise
    except json.JSONDecodeError:
        logger.error("Error decoding beta.json file")
        raise
    except Exception as e:
        logger.error(f"Error in get_betas function: {str(e)}")
        raise


def get_symbol_map(session: Session, positions: List[Any]) -> List[Dict[str, Any]]:
    """
    Creates a map of symbols with their corresponding data.
    """
    symbol_map = []
    for position in positions:
        option = get_option_for_position(session, position)
        symbol_map.append({
            "symbol": position.symbol,
            "underlying_symbol": position.underlying_symbol,
            "streamer_symbol": option.streamer_symbol,
            "quantity": position.quantity,
            "direction": position.quantity_direction
        })
    return symbol_map


def get_option_for_position(session: Session, position: Any) -> Any:
    """
    Retrieves the option object for a given position.
    """
    if position.instrument_type == InstrumentType.FUTURE_OPTION:
        return FutureOption.get(session, position.symbol)
    elif position.instrument_type == InstrumentType.EQUITY_OPTION:
        return Option.get(session, position.symbol)
    else:
        raise ValueError(f"Unknown instrument type: {position.instrument_type}")


def add_current_prices(session: Session, symbol_map: List[Dict[str, Any]],
                       betas: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Decimal]:
    """
    Adds current prices to the symbol map and returns the SPY price.
    """
    global market_data

    etf_symbols = get_unique_etf_symbols(symbol_map, betas)
    logger.debug(f"Subscribing to trades for {len(etf_symbols)} symbols: {etf_symbols}")
    market_data.subscribe_trades(etf_symbols)
    
    # Wait for streamer to receive data
    logger.debug("Waiting for streamer to receive trade data...")
    time.sleep(2)  # Give streamer time to receive data
    
    current_prices = market_data.get_trades(etf_symbols)
    logger.debug(f"Retrieved {len(current_prices)} trade prices from streamer")

    for item in symbol_map:
        etf_symbol = item['etf']
        item['current_price_etf'] = float(next(
            (trade.price for trade in current_prices if
             trade.event_symbol == etf_symbol), 0))

    spy_price = Decimal(next((trade.price for trade in current_prices if
                              trade.event_symbol == SPY_SYMBOL), 0))
    
    if spy_price == 0:
        logger.warning(f"SPY price is 0! Available symbols in current_prices: {[trade.event_symbol for trade in current_prices]}")
        logger.warning("This may cause division by zero errors in beta calculations")
    else:
        logger.debug(f"SPY price retrieved successfully: {spy_price}")
    
    return symbol_map, spy_price


def get_unique_etf_symbols(symbol_map: List[Dict[str, Any]],
                           betas: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Gets a list of unique ETF symbols from the symbol map.
    """
    etf_symbols = set()
    for item in symbol_map:
        base_symbol = get_base_symbol(item['symbol'])
        item['etf'] = betas[base_symbol]['ETF']
        etf_symbols.add(item['etf'])
    if SPY_SYMBOL not in etf_symbols:
        etf_symbols.add(SPY_SYMBOL)
    return list(etf_symbols)


def get_base_symbol(symbol: str) -> str:
    """
    Extracts the base symbol from a given symbol string.
    """
    if not symbol.startswith("."):
        return symbol.split(' ')[0]
    else:
        return symbol[1:4] if symbol[2] != "M" and symbol[2:5] != 'RTY' else symbol[1:5]


def get_greeks(symbol_map: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Retrieves Greek values for the symbols in the symbol map.
    """
    global market_data
    symbols = [x['streamer_symbol'] for x in symbol_map]
    market_data.subscribe_greeks(symbols)
    greeks = []
    while symbols:
        greeks.extend(market_data.get_greeks(symbols))
        symbols = [symbol for symbol in symbols
                   if symbol not in [greek.event_symbol for greek in greeks]]
        time.sleep(GREEK_POLL_INTERVAL)
        logger.debug(f"Waiting for greeks: {symbols}")
    return {obj.event_symbol: obj for obj in greeks}


def add_etf_greeks(symbol_map: List[Dict[str, Any]],
                   greeks: Dict[str, Any],
                   betas: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds ETF-weighted delta, theta and vega to the symbol map.
    """
    for item in symbol_map:
        streamer_symbol = item['streamer_symbol']
        if streamer_symbol in greeks:
            base_symbol = get_base_symbol(item['symbol'])
            beta = Decimal(betas[base_symbol]["ETF_BETA"])
            size = Decimal(betas[base_symbol]["SIZE"])
            direction = Decimal(1 if item['direction'] == "Long" else -1)
            multiplier = Decimal(item['quantity']) * direction
            item['etf_weighted_delta'] = Decimal(
                greeks[streamer_symbol].delta) * beta * multiplier
            item['theta'] = Decimal(
                greeks[streamer_symbol].theta) * multiplier * size
            item['vega'] = Decimal(
                greeks[streamer_symbol].vega) * multiplier * size
    return symbol_map


def add_spy_beta_delta(symbol_map: List[Dict[str, Any]], greeks: Dict[str, Any],
                       betas: Dict[str, Dict[str, Any]], price_spy: Decimal) -> \
        List[Dict[str, Any]]:
    """
    Adds SPY beta-weighted delta to the symbol map.
    """
    # Check for invalid or zero price_spy to prevent division errors
    if not price_spy or price_spy == 0:
        logger.warning(f"Invalid or zero SPY price: {price_spy}. Setting all beta_weighted_delta to 0")
        for item in symbol_map:
            item['beta_weighted_delta'] = Decimal(0)
        return symbol_map
    
    for item in symbol_map:
        streamer_symbol = item['streamer_symbol']
        if streamer_symbol in greeks:
            base_symbol = get_base_symbol(item['symbol'])
            beta = betas[base_symbol]['SPY_BETA']
            if beta != 'None':
                try:
                    item['beta_weighted_delta'] = Decimal(
                        item["etf_weighted_delta"]) * Decimal(beta) * Decimal(
                        item["current_price_etf"]) / price_spy
                    logger.debug(f"Calculated beta_weighted_delta for {streamer_symbol}: {item['beta_weighted_delta']}")
                except (ZeroDivisionError, Exception) as e:
                    logger.error(f"Error calculating beta_weighted_delta for {streamer_symbol}: {e}")
                    item['beta_weighted_delta'] = Decimal(0)
            else:
                item['beta_weighted_delta'] = Decimal(0)
                logger.debug(f"No beta available for {streamer_symbol}, setting beta_weighted_delta to 0")
    return symbol_map


def update_dashboard_sheet(portfolio_data: Dict[str, Decimal],
                           balance_data: Dict[str, int],
                           margin_data: Dict[str, Any]) -> None:
    """
    Updates the dashboard sheet with the new data.
    """
    logger.debug("Starting update_dashboard_sheet function")
    logger.debug(f"Portfolio data keys: {list(portfolio_data.keys())}")
    logger.debug(f"Balance data keys: {list(balance_data.keys())}")
    logger.debug(f"Margin data keys: {list(margin_data.keys())}")
    
    logger.debug("Getting tracker workbook")
    tracker_workbook = get_workbook()
    
    logger.debug("Getting Dashboard worksheet")
    dashboard = tracker_workbook.worksheet("Dashboard")
    
    logger.debug("Finding first empty row")
    row = get_first_empty_row(dashboard, start_row=4)
    logger.debug(f"First empty row: {row}")

    # Get the date from the last non-empty row
    logger.debug("Getting last row date")
    last_row_date = dashboard.cell(row - 1, 1).value
    logger.debug(f"Last row date: {last_row_date}")

    logger.debug("Preparing new row data")
    new_row_data = prepare_new_row_data(dashboard, row,
                                        last_row_date,
                                        portfolio_data,
                                        balance_data,
                                        margin_data)
    logger.debug(f"New row data prepared with {len(new_row_data)} columns")

    is_new_row = not is_same_date(new_row_data[0], last_row_date)
    logger.debug(f"Is new row: {is_new_row} (new date: {new_row_data[0]}, last date: {last_row_date})")

    if is_new_row:
        logger.debug("Inserting new row")
        # Insert an empty row before adding new data
        dashboard.insert_row([], row)
        insert_row_data(dashboard, row, new_row_data)
        logger.debug(f"New row inserted at row {row}")
        
        if is_end_of_trading_week():
            logger.debug("End of trading week detected, formatting row")
            format_end_of_week_row(dashboard, row)
        else:
            logger.debug("Not end of trading week")
    else:
        logger.debug(f"Updating existing row {row - 1}")
        update_existing_row(dashboard, row - 1, new_row_data)
    
    logger.debug("update_dashboard_sheet function completed")


def prepare_new_row_data(dashboard, row: int,
                         last_row_date: str,
                         portfolio_data: Dict[str, Decimal],
                         balance_data: Dict[str, int],
                         margin_data: Dict[str, Any]) -> List[Any]:
    """
    Prepares the new row data for the dashboard.
    """
    new_row_data = get_row_data(dashboard, row - 1, offset=1)

    new_row_data[0] = datetime.now().strftime("%m-%d-%Y")
    new_row_data[1] = portfolio_data['beta_weighted_delta']
    new_row_data[3] = portfolio_data['theta']
    new_row_data[5] = balance_data['net_liquidating_value']
    new_row_data[6] = margin_data['margin_requirement']
    new_row_data[7] = margin_data['bil']

    if is_same_date(new_row_data[0], last_row_date):
        new_row_data[2] = adjust_formula(new_row_data[2])
        new_row_data[4] = adjust_formula(new_row_data[4])
        new_row_data[8] = adjust_formula(new_row_data[9])
        new_row_data[9] = adjust_formula(new_row_data[10])

    if len(new_row_data) > 11:
        new_row_data[10:] = [''] * (len(new_row_data) - 10)

    return new_row_data


def is_same_date(date_str1: str, date_str2: str) -> bool:
    """
    Checks if two given date strings represent the same date.
    Handles different date formats and two-digit years.
    """

    def parse_date(date_str: str) -> datetime:
        # Replace '/' with '-' for consistency
        date_str = date_str.replace('/', '-')

        # Try parsing with different formats
        formats = ["%m-%d-%Y", "%m-%d-%y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse date string: {date_str}")

    try:
        date1 = parse_date(date_str1)
        date2 = parse_date(date_str2)
        return date1.date() == date2.date()
    except ValueError as e:
        logger.error(f"Error comparing dates: {e}")
        return False


def is_end_of_trading_week() -> bool:
    """
    Checks if it's the end of the trading week.
    """
    return 4 <= datetime.now().weekday() <= 6


def update_existing_row(dashboard, row: int, new_row_data: List[Any]) -> None:
    """
    Updates an existing row in the dashboard.
    """
    start_column = 'A'
    end_column = chr(ord(start_column) + len(new_row_data) - 1)
    range_to_update = f'{start_column}{row}:{end_column}{row}'
    dashboard.update([new_row_data], range_to_update, raw=False)


def insert_new_row(dashboard, row: int, new_row_data: List[Any]) -> None:
    """
    Inserts a new row in the dashboard.
    """
    insert_row_data(dashboard, row, new_row_data)


def adjust_formula(formula: str) -> str:
    """
    Adjusts cell references in a formula to refer to the previous row.
    """

    def decrement(match):
        return match.group(1) + str(int(match.group(2)) - 1)

    return re.sub(r'([A-Z]+)(\d+)', decrement, formula)


def is_closed_today():
    logger.debug("Starting is_closed_today function")
    current_year = datetime.now().year
    today = datetime.now().date()
    logger.debug(f"Checking if exchanges are closed for today: {today} (year: {current_year})")
    
    logger.debug("Getting NYSE calendar")
    nyse = mcal.get_calendar('NYSE')
    
    logger.debug(f"Getting NYSE schedule for {current_year}")
    schedule = nyse.schedule(start_date=f"{current_year}-01-01",
                             end_date=f"{current_year}-12-31")
    logger.debug(f"NYSE schedule has {len(schedule)} trading days")
    
    logger.debug(f"Creating date range for all days in {current_year}")
    all_days = pd.date_range(start=f"{current_year}-01-01",
                             end=f"{current_year}-12-31")
    logger.debug(f"All days range has {len(all_days)} days")

    trading_days = schedule.index
    holidays = all_days.difference(trading_days)
    logger.debug(f"Found {len(holidays)} non-trading days")

    is_closed = today in holidays.date
    logger.debug(f"Is today a non-trading day? {is_closed}")
    return is_closed


def main() -> None:
    logger.info("Starting update_dashboard application")
    global market_data
    try:
        logger.info("Initializing market data connection")
        market_data = MarketData()
        
        logger.info("Starting market data streamer")
        market_data.start_streamer()
        
        logger.info("Checking if exchanges are open today")
        if not is_closed_today():
            logger.info("Exchanges are open - proceeding with dashboard update")
            update_dashboard()
            logger.info("Dashboard update completed successfully")
        else:
            logger.info('Today the exchanges are closed - skipping dashboard update')
        
        logger.info("update_dashboard application finished successfully")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)

LAST_RUN_PATH = "last_run_timestamp.txt"


def read_last_run_timestamp(path: str = LAST_RUN_PATH) -> Optional[datetime]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if not content:
            return None
        return datetime.fromisoformat(content)
    except Exception as exc:
        logger.warning(f"Unable to read last run timestamp from {path}: {exc}")
        return None


def write_last_run_timestamp(timestamp: datetime, path: str = LAST_RUN_PATH) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(timestamp.isoformat())
    except Exception as exc:
        logger.error(f"Unable to write last run timestamp to {path}: {exc}", exc_info=True)


if __name__ == "__main__":
    session = ApplicationSession().session
    account = Account(session).account
    last_run = read_last_run_timestamp()
    if last_run is None:
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start_time = _ensure_aware_datetime(last_run)

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

    write_last_run_timestamp(fetch_timestamp)

    update_closed_trades_sheet(closed_trades)
    update_opened_trades_sheet(opened_trades)
    # main()