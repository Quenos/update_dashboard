import logging

from tastytrade import Account as ac

from config import ACCOUNT_ID

logger = logging.getLogger(__name__)

class Account:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Account, cls).__new__(cls)
        return cls._instance

    def __init__(self, session):
        if not hasattr(self, 'initialized'):
            if not ACCOUNT_ID:
                raise ValueError("TT_ACCOUNT_ID environment variable not set")
            try:
                self._account = ac.get(session, ACCOUNT_ID)
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Account: {str(e)}")
                raise

    @property
    def account(self):
        if not hasattr(self, '_account'):
            raise AttributeError("Account not initialized")
        return self._account
