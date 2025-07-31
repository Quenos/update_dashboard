from tastytrade import Account as ac
import logging

logger = logging.getLogger(__name__)

class Account:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Account, cls).__new__(cls)
        return cls._instance

    def __init__(self, session):
        if not hasattr(self, 'initialized'):  # Check if it's the first time __init__ is called
            try:
                self._account = ac.get(session, '5WV52737')
                self.initialized = True  # Mark as initialized
            except Exception as e:
                logger.error(f"Failed to initialize Account: {str(e)}")
                raise

    @property
    def account(self):
        if not hasattr(self, '_account'):
            raise AttributeError("Account not initialized")
        return self._account
