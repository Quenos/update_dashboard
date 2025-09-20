import logging
from datetime import datetime, timedelta

from tastytrade import OAuthSession

from config import TT_API_CLIENT_SECRET, TT_REFRESH_TOKEN

logger = logging.getLogger(__name__)


class ApplicationSession:
    """
    It provides access to the  session object and reads configuration from a config file.
    """

    def __init__(self) -> None:
        try:
            if not TT_API_CLIENT_SECRET or not TT_REFRESH_TOKEN:
                raise ValueError("API credentials not configured")
            self._session = OAuthSession(TT_API_CLIENT_SECRET, TT_REFRESH_TOKEN)
            self._last_refresh = datetime.utcnow()
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing ApplicationSession: {str(e)}")
            raise

    @property
    def session(self) -> OAuthSession:
        """
        Get the  session object.

        Returns:
            Session: The  session object.
        """
        if not hasattr(self, '_session'):
            raise AttributeError("Session not initialized")
        if datetime.utcnow() - self._last_refresh > timedelta(minutes=30):
            self._session.refresh()
            self._last_refresh = datetime.utcnow()
        return self._session
