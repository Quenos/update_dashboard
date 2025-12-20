import asyncio
from threading import Event, Lock, Thread
from typing import Dict, Iterable, Optional, Any
from session import ApplicationSession

from tastytrade import DXLinkStreamer
from tastytrade.dxfeed import Trade, Quote, Greeks


class MarketDataStreamer:
    def __init__(self, session: "ApplicationSession"):
        self.session = session
        self.streamer: Optional[DXLinkStreamer] = None

        # symbols (deduped)
        self._trade_symbols: set[str] = set()
        self._quote_symbols: set[str] = set()
        self._greek_symbols: set[str] = set()

        # latest events (shared across threads)
        self._trades: Dict[str, Trade] = {}
        self._quotes: Dict[str, Quote] = {}
        self._greeks: Dict[str, Greeks] = {}

        # threading + loop
        self._tasks: list[asyncio.Task[Any]] = []
        self._thread: Optional[Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = Event()

        # locks
        self._data_lock = Lock()
        self._symbol_lock = Lock()

        # startup error capture
        self._startup_exc: Optional[BaseException] = None

    # ----------- thread-safe snapshots -----------

    @property
    def trades(self) -> Dict[str, Trade]:
        with self._data_lock:
            return dict(self._trades)

    @property
    def quotes(self) -> Dict[str, Quote]:
        with self._data_lock:
            return dict(self._quotes)

    @property
    def greeks(self) -> Dict[str, Greeks]:
        with self._data_lock:
            return dict(self._greeks)

    # ----------- non-async subscription API (add-only) -----------

    def subscribe_to_trades(self, symbols: Iterable[str]) -> None:
        self._subscribe_non_async(Trade, symbols, which="trade")

    def subscribe_to_quotes(self, symbols: Iterable[str]) -> None:
        self._subscribe_non_async(Quote, symbols, which="quote")

    def subscribe_to_greeks(self, symbols: Iterable[str]) -> None:
        self._subscribe_non_async(Greeks, symbols, which="greek")

    def _subscribe_non_async(self, event_cls: type, symbols: Iterable[str], which: str) -> None:
        symbols_set = set(symbols)
        if not symbols_set:
            return

        with self._symbol_lock:
            if which == "trade":
                new = symbols_set - self._trade_symbols
                if not new:
                    return
                self._trade_symbols |= new
            elif which == "quote":
                new = symbols_set - self._quote_symbols
                if not new:
                    return
                self._quote_symbols |= new
            elif which == "greek":
                new = symbols_set - self._greek_symbols
                if not new:
                    return
                self._greek_symbols |= new
            else:
                raise ValueError("unknown subscription type")

        # Not running yet: startup will subscribe from the sets
        if not self._ready.is_set() or self._loop is None or self.streamer is None:
            return

        # Running: schedule subscribe onto the streamer's loop (non-blocking)
        asyncio.run_coroutine_threadsafe(self._subscribe_more(event_cls, list(new)), self._loop)

    # ----------- async internals -----------

    async def _connect_and_subscribe(self) -> None:
        self.streamer = await DXLinkStreamer(self.session)

        with self._symbol_lock:
            trade_syms = list(self._trade_symbols)
            quote_syms = list(self._quote_symbols)
            greek_syms = list(self._greek_symbols)

        if trade_syms:
            await self.streamer.subscribe(Trade, trade_syms)
        if quote_syms:
            await self.streamer.subscribe(Quote, quote_syms)
        if greek_syms:
            await self.streamer.subscribe(Greeks, greek_syms)

    async def handle_trades(self) -> None:
        assert self.streamer is not None
        async for trade in self.streamer.listen(Trade):
            with self._data_lock:
                self._trades[trade.event_symbol] = trade

    async def handle_quotes(self) -> None:
        assert self.streamer is not None
        async for quote in self.streamer.listen(Quote):
            with self._data_lock:
                self._quotes[quote.event_symbol] = quote

    async def handle_greeks(self) -> None:
        assert self.streamer is not None
        async for greeks in self.streamer.listen(Greeks):
            with self._data_lock:
                self._greeks[greeks.event_symbol] = greeks

    async def _run(self) -> None:
        try:
            await self._connect_and_subscribe()

            # IMPORTANT: start consumers even if no symbols yet
            self._tasks = [
                asyncio.create_task(self.handle_trades()),
                asyncio.create_task(self.handle_quotes()),
                asyncio.create_task(self.handle_greeks()),
            ]

        except BaseException as e:
            self._startup_exc = e
            raise
        finally:
            self._ready.set()

        # cancellation-safe shutdown
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _subscribe_more(self, event_cls: type, symbols: list[str]) -> None:
        if self.streamer is None or not symbols:
            return
        await self.streamer.subscribe(event_cls, symbols)

    async def _stop_async(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self.streamer is not None:
            await self.streamer.close()
            self.streamer = None

    # ----------- non-async lifecycle -----------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._startup_exc = None
        self._ready.clear()

        def runner() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run())
            finally:
                self._loop.close()

        self._thread = Thread(target=runner, daemon=True)
        self._thread.start()

        self._ready.wait()
        if self._startup_exc is not None:
            raise RuntimeError("MarketDataStreamer failed to start") from self._startup_exc

    def stop(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        fut = asyncio.run_coroutine_threadsafe(self._stop_async(), loop)
        fut.result()

        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass

        if self._thread:
            self._thread.join(timeout=10)

        self._loop = None
        self._thread = None