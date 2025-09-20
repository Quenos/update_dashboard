import os

from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID = os.getenv('TT_ACCOUNT_ID')
TT_API_CLIENT_SECRET = os.getenv('TT_API_CLIENT_SECRET')
TT_REFRESH_TOKEN = os.getenv('TT_REFRESH_TOKEN')
SPREADSHEET_URL = os.getenv('SPREADSHEET_URL', 'https://docs.google.com/spreadsheets/d/1A528mC08exsLG9tXSoVhF-vZsuRGwspOxRpEhPNwbpE/edit?usp=sharing')
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '1A528mC08exsLG9tXSoVhF-vZsuRGwspOxRpEhPNwbpE')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG').upper()
GREEK_POLL_INTERVAL = int(os.getenv('GREEK_POLL_INTERVAL', '5'))
