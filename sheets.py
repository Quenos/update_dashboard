import re

import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from gspread.exceptions import WorksheetNotFound


def get_workbook():
    """
    Retrieves and returns a Google Sheets workbook.

    Returns:
        gspread.models.Spreadsheet: The Google Sheets workbook.
    """
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
    client = gspread.authorize(creds)
    from config import SPREADSHEET_URL
    return client.open_by_url(SPREADSHEET_URL)


def get_row_data(worksheet, row: int, offset: int) -> list[str]:
    row_data = worksheet.row_values(row, value_render_option='FORMULA')
    new_row_data = []
    for cell in row_data:
        if isinstance(cell, str) and cell.startswith('='):  # Check if the cell contains a formula
            # Increment row numbers in the formula
            updated_formula = re.sub(r'([A-Z]+)(\d+)', lambda m: f"{m.group(1)}{int(m.group(2)) + offset}", cell)
            new_row_data.append(updated_formula)
        else:
            new_row_data.append(cell)

    return new_row_data


def insert_row_data(worksheet, row: int, values: list[str]):
    """
    Inserts row data into the specified row of the worksheet.
    """
    for col in range(0, len(values)):
        worksheet.update_cell(row, col + 1, values[col])

def create_worksheet(spreadsheet, title, rows=100, cols=20):
    try:
        worksheet = spreadsheet.worksheet(title)
    except WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
    return worksheet


def copy_format(source_sheet: str, destination_sheet: str):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file("credentials.json", scopes=scopes)
    service = build('sheets', 'v4', credentials=creds)
    from config import SPREADSHEET_ID
    spreadsheet_id = SPREADSHEET_ID

    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheets = spreadsheet.get('sheets', '')

    destination_sheet_id = [sheet['properties']['sheetId'] for sheet in sheets if sheet['properties']['title'] == destination_sheet][0]
    
    request = {
        "requests": [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": destination_sheet_id,
                        "startRowIndex": 4,
                        "endRowIndex": 5
                    },
                    "cell": {"userEnteredFormat": {}},
                    "fields": "userEnteredFormat"
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": destination_sheet_id,
                        "startRowIndex": 4,
                        "endRowIndex": 5,
                        "startColumnIndex": 0,
                        "endColumnIndex": 1
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "DATE",
                                "pattern": "m/d/y"
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat"
                }
            },
            {
                "repeatCell": {
                    "range": {
                        "sheetId": destination_sheet_id,
                        "startRowIndex": 4,
                        "endRowIndex": 5,
                        "startColumnIndex": 7,
                        "endColumnIndex": 8
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "CURRENCY",
                                "pattern": "\"$\"#,##0.00"
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat"
                }
            }
        ]
    }

    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body=request
    ).execute()


if __name__ == '__main__':
    copy_format('OPEN', 'Draft Entries')
