import json
import logging
import re
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandas_market_calendars as mcal
from tastytrade.instruments import FutureOption, Option
from tastytrade.metrics import get_market_metrics
from tastytrade.order import InstrumentType
from tastytrade.session import Session

from account import Account
from config import GREEK_POLL_INTERVAL
from market_data import MarketData
from session import ApplicationSession
from sheets import get_row_data, get_workbook, insert_row_data


logger = logging.getLogger(__name__)

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
        logger.info("Initializing market data connection")
        market_data = MarketData()
        
        logger.info("Starting market data streamer")
        market_data.start_streamer()
        
        logger.info("Checking if exchanges are open today")
        if is_closed_today():
            logger.info("Exchanges are closed - exiting")
            return

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