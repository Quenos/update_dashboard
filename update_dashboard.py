import json
import logging
import re
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandas_market_calendars as mcal

from account import Account
from market_data import MarketData
from session import ApplicationSession
from sheets import get_row_data, get_workbook, insert_row_data
from config import LOG_LEVEL, GREEK_POLL_INTERVAL
from tastytrade.instruments import FutureOption, Option
from tastytrade.metrics import get_market_metrics
from tastytrade.order import InstrumentType
from tastytrade.session import Session


# Configure logging
logging.basicConfig(
    filename='update_dashboard.log',
    level=getattr(logging, LOG_LEVEL, logging.ERROR),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.getLogger().setLevel(getattr(logging, LOG_LEVEL, logging.ERROR))


logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.ERROR))

market_data: MarketData | None = None
SPY_SYMBOL = 'SPY'


def format_end_of_week_row(worksheet, row: int) -> None:
    """
    Formats the end of week row in the worksheet.

    :param worksheet: The worksheet to format.
    :param row: The row number to format.
    """
    light_yellow = {
        'backgroundColor': {
            'red': 1.0,
            'green': 0.95,
            'blue': 0.8
        }
    }
    worksheet.format(f'A{row}:P{row}', light_yellow)


def get_first_empty_row(worksheet, start_row=2) -> int:
    """
    Returns the first empty row in the given worksheet by fetching column A
    in one call and then checking in memory.
    """
    # Fetch all values in the first column at once
    col_values = worksheet.col_values(1)

    # Iterate over the values in column A
    for idx, value in enumerate(col_values, start=1):
        if idx >= start_row and not value:
            return idx

    # If no empty cell was found, it's the row after the last value
    return len(col_values) + 1


def update_dashboard() -> None:
    """
    Updates the dashboard sheet in the tracker workbook with new data.
    """
    try:
        session = ApplicationSession().session
        account = Account(session).account

        portfolio_data = get_portfolio_data(session, account)
        balance_data = get_balance_data(session, account)
        margin_data = get_margin_data(session, account)

        update_dashboard_sheet(portfolio_data, balance_data, margin_data)
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}", exc_info=True)
        raise  # Re-raise the exception after logging


def get_portfolio_data(session: Session, account: Account) -> Dict[str, int]:
    """
    Retrieves portfolio data including beta-weighted delta and theta.
    """
    portfolio_beta_weighted_delta, portfolio_theta, theta_vega_ratio = get_portfolio_metrics(session, account)
    return {
        'beta_weighted_delta': round(portfolio_beta_weighted_delta),
        'theta': round(portfolio_theta),
        'theta_vega_ratio': theta_vega_ratio
    }


def get_balance_data(session: Session, account: Account) -> Dict[str, int]:
    """
    Retrieves account balance data.
    """
    balance = account.get_balances(session)
    return {
        'net_liquidating_value': int(balance.net_liquidating_value)
    }


def get_margin_data(session: Session, account: Account) -> Dict[str, Any]:
    """
    Retrieves margin requirement data.
    """
    margin_requirement_report = account.get_margin_requirements(session)
    margin_requirement = 0
    bil = 0
    for margin_report in margin_requirement_report.groups:
        margin_requirement += margin_report.buying_power
        if margin_report.description == 'BIL':
            bil = margin_report.buying_power

    return {
        'margin_requirement': int(margin_requirement),
        'bil': int(bil)
    }


def get_portfolio_metrics(session: Session, account: Account) -> Tuple[Decimal, Decimal]:
    """
    Calculates portfolio beta-weighted delta and theta.
    """
    positions = get_filtered_positions(account, session)
    betas = get_betas(session)
    symbol_map = get_symbol_map(session, positions)
    symbol_map, price_spy = add_current_prices(session, symbol_map, betas)
    greeks = get_greeks(symbol_map)
    symbol_map = add_etf_greeks(symbol_map, greeks, betas)
    symbol_map = add_spy_beta_delta(symbol_map, greeks, betas, price_spy)

    portfolio_beta_weighted_delta = sum(
        Decimal(x.get('beta_weighted_delta', 0)) for x in symbol_map)
    portfolio_theta = sum(Decimal(x.get('theta', 0)) for x in symbol_map)
    portfolio_vega = sum(Decimal(x.get('vega', 0)) for x in symbol_map)
    theta_vega_ratio = f'1 : {round(portfolio_vega/portfolio_theta)}'

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
    market_data.subscribe_trades(etf_symbols)
    current_prices = market_data.get_trades(etf_symbols)

    for item in symbol_map:
        etf_symbol = item['etf']
        item['current_price_etf'] = float(next(
            (trade.price for trade in current_prices if
             trade.event_symbol == etf_symbol), 0))

    spy_price = Decimal(next((trade.price for trade in current_prices if
                              trade.event_symbol == SPY_SYMBOL), 0))
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
    for item in symbol_map:
        streamer_symbol = item['streamer_symbol']
        if streamer_symbol in greeks:
            base_symbol = get_base_symbol(item['symbol'])
            beta = betas[base_symbol]['SPY_BETA']
            if beta != 'None':
                item['beta_weighted_delta'] = Decimal(
                    item["etf_weighted_delta"]) * Decimal(beta) * Decimal(
                    item["current_price_etf"]) / price_spy
            else:
                item['beta_weighted_delta'] = Decimal(0)
    return symbol_map


def update_dashboard_sheet(portfolio_data: Dict[str, Decimal],
                           balance_data: Dict[str, int],
                           margin_data: Dict[str, Any]) -> None:
    """
    Updates the dashboard sheet with the new data.
    """
    tracker_workbook = get_workbook()
    dashboard = tracker_workbook.worksheet("Dashboard")
    row = get_first_empty_row(dashboard, start_row=4)

    # Get the date from the last non-empty row
    last_row_date = dashboard.cell(row - 1, 1).value

    new_row_data = prepare_new_row_data(dashboard, row,
                                        last_row_date,
                                        portfolio_data,
                                        balance_data,
                                        margin_data)

    is_new_row = not is_same_date(new_row_data[0], last_row_date)

    if is_new_row:
        # Insert an empty row before adding new data
        dashboard.insert_row([], row)
        insert_row_data(dashboard, row, new_row_data)
        
        if is_end_of_trading_week():
            format_end_of_week_row(dashboard, row)
    else:
        update_existing_row(dashboard, row - 1, new_row_data)


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
    current_year = datetime.now().year
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=f"{current_year}-01-01",
                             end_date=f"{current_year}-12-31")
    all_days = pd.date_range(start=f"{current_year}-01-01",
                             end=f"{current_year}-12-31")

    trading_days = schedule.index
    holidays = all_days.difference(trading_days)

    return datetime.now().date() in holidays.date


def main() -> None:
    global market_data
    try:
        market_data = MarketData()
        market_data.start_streamer()
        if not is_closed_today():
            update_dashboard()
        else:
            logging.info('Today the exchanges are closed')
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}", exc_info=True)


if __name__ == "__main__":
    main()
