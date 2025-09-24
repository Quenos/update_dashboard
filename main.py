"""Command-line entry point for update_dashboard workflows."""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler

from config import LOG_LEVEL
from account import Account
from session import ApplicationSession
from update_dashboard import update_dashboard
from update_trades_sheet import (
    fetch_recent_trades,
    update_closed_trades_sheet,
    update_opened_trades_sheet,
)


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL, logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = RotatingFileHandler(
        'update_dashboard.log', maxBytes=1024 * 1024 * 10, backupCount=5
    )
    file_handler.doRollover()
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

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
        'asyncio',
    ]

    for package in noisy_packages:
        logging.getLogger(package).setLevel(logging.ERROR)

    for name in ('main', 'update_dashboard', 'update_trades_sheet'):
        logging.getLogger(name).setLevel(level)


def run_dashboard() -> None:
    update_dashboard()


def run_trades() -> None:
    session = ApplicationSession().session
    account = Account(session).account
    opened_trades, closed_trades = fetch_recent_trades(account, session)
    update_closed_trades_sheet(closed_trades)
    update_opened_trades_sheet(opened_trades)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update dashboard and trade sheets")
    parser.add_argument(
        "command",
        choices={"dashboard", "trades"},
        help="Task to run",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    LOGGER.info('=' * 60)
    LOGGER.info('NEW LOG SESSION STARTED - Previous logs rotated to backup files')
    LOGGER.info('=' * 60)
    args = parse_args()
    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "trades":
        run_trades()
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
