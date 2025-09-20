"""Helper utilities for common Google Sheets operations.

The module provides convenience helpers for working with tabs (worksheets),
rows, and cells without repeating boilerplate gspread logic.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

import gspread
from google.oauth2.service_account import Credentials
from gspread.cell import Cell
from gspread.exceptions import WorksheetNotFound
from gspread.utils import rowcol_to_a1
from gspread.worksheet import Worksheet

from config import SPREADSHEET_URL


def get_workbook() -> gspread.spreadsheet.Spreadsheet:
    """Return the main Google Sheets workbook for the dashboard."""
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
    client = gspread.authorize(creds)
    return client.open_by_url(SPREADSHEET_URL)


def create_worksheet(
    spreadsheet: gspread.spreadsheet.Spreadsheet,
    title: str,
    rows: int = 100,
    cols: int = 20,
) -> Worksheet:
    """Return a worksheet by title, creating it if it does not exist."""
    try:
        return spreadsheet.worksheet(title)
    except WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)


# Cache worksheets so repeated lookups avoid extra API calls.
_WORKSHEET_CACHE: dict[str, Worksheet] = {}


def _normalize_title(tab_title: str) -> str:
    """Return a normalized worksheet title for consistent cache lookups."""
    return tab_title.strip()


def get_worksheet(
    tab_title: str,
    *,
    create_if_missing: bool = False,
    rows: int = 100,
    cols: int = 20,
) -> Worksheet:
    """Return a worksheet (tab) by title, optionally creating it when missing."""
    normalized_title = _normalize_title(tab_title)
    if not normalized_title:
        raise ValueError("Worksheet title must not be empty.")

    worksheet = _WORKSHEET_CACHE.get(normalized_title)
    if worksheet is not None:
        return worksheet

    workbook = get_workbook()
    try:
        worksheet = workbook.worksheet(normalized_title)
    except WorksheetNotFound:
        if not create_if_missing:
            raise
        worksheet = create_worksheet(workbook, normalized_title, rows=rows, cols=cols)

    _WORKSHEET_CACHE[normalized_title] = worksheet
    return worksheet


def clear_cached_worksheet(tab_title: Optional[str] = None) -> None:
    """Remove cached worksheets.

    Args:
        tab_title: When provided, only that worksheet cache entry is removed.
            Otherwise, the entire worksheet cache is cleared.
    """
    if tab_title is None:
        _WORKSHEET_CACHE.clear()
        return

    _WORKSHEET_CACHE.pop(_normalize_title(tab_title), None)


def get_row_values(
    tab_title: str,
    row_number: int,
    *,
    value_render_option: str = "FORMATTED_VALUE",
) -> List[Any]:
    """Return the values for a single row."""
    if row_number < 1:
        raise ValueError("Row numbers must start at 1.")

    worksheet = get_worksheet(tab_title)
    return worksheet.row_values(row_number, value_render_option=value_render_option)


def get_rows(
    tab_title: str,
    start_row: int,
    end_row: Optional[int] = None,
    *,
    value_render_option: str = "FORMATTED_VALUE",
) -> List[List[Any]]:
    """Return values for a row range. Includes the end row when provided."""
    if start_row < 1:
        raise ValueError("Row numbers must start at 1.")
    if end_row is not None and end_row < start_row:
        raise ValueError("end_row must not be less than start_row.")

    worksheet = get_worksheet(tab_title)
    row_range = f"{start_row}:{end_row}" if end_row is not None else f"{start_row}:{start_row}"
    return worksheet.get(row_range, value_render_option=value_render_option)


def update_row_values(
    tab_title: str,
    row_number: int,
    values: Sequence[Any],
    *,
    value_input_option: str = "USER_ENTERED",
) -> None:
    """Replace the contents of a row with the provided values."""
    if row_number < 1:
        raise ValueError("Row numbers must start at 1.")

    worksheet = get_worksheet(tab_title)
    # gspread expects a 2D list for an update that targets a row range.
    worksheet.update(
        f"{row_number}:{row_number}",
        [list(values)],
        value_input_option=value_input_option,
    )


def append_row(
    tab_title: str,
    values: Sequence[Any],
    *,
    value_input_option: str = "USER_ENTERED",
    table_range: Optional[str] = None,
) -> None:
    """Append a row of values to the worksheet."""
    worksheet = get_worksheet(tab_title)
    worksheet.append_row(
        list(values),
        value_input_option=value_input_option,
        table_range=table_range,
    )


def get_cell_value(
    tab_title: str,
    row: int,
    column: int,
    *,
    value_render_option: str = "FORMATTED_VALUE",
) -> Any:
    """Return a single cell value identified by row and column."""
    if row < 1 or column < 1:
        raise ValueError("Row and column numbers must start at 1.")

    worksheet = get_worksheet(tab_title)
    cell_label = rowcol_to_a1(row, column)
    return worksheet.acell(cell_label, value_render_option=value_render_option).value


def set_cell_value(
    tab_title: str,
    row: int,
    column: int,
    value: Any,
    *,
    value_input_option: str = "USER_ENTERED",
) -> None:
    """Update a single cell value identified by row and column."""
    if row < 1 or column < 1:
        raise ValueError("Row and column numbers must start at 1.")

    worksheet = get_worksheet(tab_title)
    cell_label = rowcol_to_a1(row, column)
    worksheet.update(cell_label, value, value_input_option=value_input_option)


def get_range_values(
    tab_title: str,
    a1_range: str,
    *,
    value_render_option: str = "FORMATTED_VALUE",
) -> List[List[Any]]:
    """Return values for an arbitrary A1 range."""
    worksheet = get_worksheet(tab_title)
    return worksheet.get(a1_range, value_render_option=value_render_option)


def update_range_values(
    tab_title: str,
    a1_range: str,
    values: Sequence[Sequence[Any]],
    *,
    value_input_option: str = "USER_ENTERED",
) -> None:
    """Update an arbitrary A1 range with the provided values."""
    worksheet = get_worksheet(tab_title)
    worksheet.update(a1_range, [list(row) for row in values], value_input_option=value_input_option)


def batch_update_cells(
    tab_title: str,
    cell_updates: Iterable[tuple[int, int, Any]],
    *,
    value_input_option: str = "USER_ENTERED",
) -> None:
    """Apply several cell updates in a single request."""
    updates = list(cell_updates)
    if not updates:
        return

    worksheet = get_worksheet(tab_title)
    cells = []
    for row, column, value in updates:
        if row < 1 or column < 1:
            raise ValueError("Row and column numbers must start at 1.")
        cells.append(Cell(row=row, col=column, value=value))

    worksheet.update_cells(cells, value_input_option=value_input_option)
