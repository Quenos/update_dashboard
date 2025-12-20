import asyncio
from dataclasses import dataclass, field
from enum import Enum
from threading import Thread, Event, Lock, current_thread

from typing import Any, ClassVar, Dict, List, Optional, Iterable

from tastytrade.streamer import DXLinkStreamer, Greeks, Quote, Trade

from session import ApplicationSession


class EventType(str, Enum):
    GREEKS = 'Greeks'
    TRADE = 'Trade'
    QUOTE = 'Quote'

@dataclass
class MarketData():
    database: Any = field(init=True, default=None)

    # singleton class
    _instance: ClassVar['MarketData'] = None

    _subscribed_symbols: Dict[str, List[str]] = field(init=False, default_factory=dict) 
    _new_symbols: Dict[str, List[str]] = field(init=False, default_factory=dict) 
    _cached_events: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict) 
    _stop_streaming: bool = field(init=False, default=False)
    _thread_runs: bool = field(init=False, default=False)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MarketData, cls).__new__(cls)
        return cls._instance

    def __post_init__(self):
        if not hasattr(self, '_initialized'):
            self._new_symbols[EventType.GREEKS] = []
            self._new_symbols[EventType.TRADE] = []
            self._new_symbols[EventType.QUOTE] = []
            self._subscribed_symbols[EventType.GREEKS] = []
            self._subscribed_symbols[EventType.TRADE] = []
            self._subscribed_symbols[EventType.QUOTE] = []
            self._cached_events[EventType.GREEKS] = {}
            self._cached_events[EventType.TRADE] = {}
            self._cached_events[EventType.QUOTE] = {}

    def subscribe_greeks(self, symbols: list[str]) -> List[Greeks] | None:
        return self._subscribe_symbol(EventType.GREEKS, symbols)

    def subscribe_trades(self, symbols: list[str]) -> List[Trade] | None:
        return self._subscribe_symbol(EventType.TRADE, symbols)
    
    def subscribe_quotes(self, symbols: list[str]) -> Quote:
        return self._subscribe_symbol(EventType.QUOTE, symbols)
    
    def get_greeks(self, symbols: list[str]) -> List[Greeks] | None:
        return self._get_events(EventType.GREEKS, symbols)

    def get_trades(self, symbols: list[str]) -> List[Trade] | None:
        return self._get_events(EventType.TRADE, symbols)
    
    def get_quotes(self, symbols: list[str]) -> Quote:
        return self._get_events(EventType.QUOTE, symbols)

    def start_streamer(self) -> None:
        if self._thread_runs:
            return
        self._thread_runs = True
        thread = Thread(target=self._streamer_thread, daemon=True)
        thread.start()

    def stop_streamer(self) -> None:
        self._stop_streaming = True
        self._thread_runs = False

    def _subscribe_symbol(self, event_type: EventType, symbols: List[str]) -> None:
        new_symbols = list(set(symbols) - set(self._subscribed_symbols[event_type]))
        if new_symbols:
            self._new_symbols[event_type] = new_symbols
            self._subscribed_symbols[event_type] = list(set(self._subscribed_symbols[event_type]) | set(symbols))

    def _get_events(self, event_type: EventType, symbols: List[str]) -> List[Greeks] | List[Trade] | List[Quote] | None:
        market_data = list(self._cached_events[event_type].values())
        market_data = [data for data in market_data if data.event_symbol in symbols]
        return market_data

    async def _fetch_data(self, event_type: EventType) -> None:
        event_type_map = {
            EventType.GREEKS: Greeks,
            EventType.TRADE: Trade,
            EventType.QUOTE: Quote
        }
        session = ApplicationSession().session
        async with DXLinkStreamer(session) as streamer:
            streamer.fill_event_time = True
            if self.database and not streamer.database:
                streamer.database = self.database
            while not self._stop_streaming:
                if self._new_symbols[event_type]:
                    await streamer.subscribe(event_type_map[event_type], symbols=self._new_symbols[event_type])
                    self._new_symbols[event_type] = []
                if self.database:
                    await asyncio.sleep(10)
                    continue
                event = streamer.get_event_nowait(event_type_map[event_type])
                if event:
                    self._cached_events[event_type][event.event_symbol] = event
                await asyncio.sleep(0.1)

    async def _start_streamers(self) -> None:
        await asyncio.gather(
            self._fetch_data(EventType.QUOTE),
            self._fetch_data(EventType.TRADE),
            self._fetch_data(EventType.GREEKS)
        )

    def _streamer_thread(self) -> None:
        asyncio.run(self._start_streamers())


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

        # optional: capture startup exception so start() can raise it
        self._startup_exc: Optional[BaseException] = None

    # ----------- thread-safe snapshots (read side) -----------

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

    # ----------- non-async subscription API -----------

    def subscribe_to_trades(self, symbols: Iterable[str]) -> None:
        self._subscribe_non_async(Trade, symbols, which="trade")

    def subscribe_to_quotes(self, symbols: Iterable[str]) -> None:
        self._subscribe_non_async(Quote, symbols, which="quote")

    def subscribe_to_greeks(self, symbols: Iterable[str]) -> None:
        self._subscribe_non_async(Greeks, symbols, which="greek")

    def _subscribe_non_async(self, event_cls: type, symbols: Iterable[str], which: str) -> None:
        # if not running yet, initial connect will pick them up
        is_stopped = False
        if self.is_running():
            self.stop()
            is_stopped = True

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

        if is_stopped:
            self.start()

    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and self._loop is not None
        )
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
        agen = self.streamer.listen(Trade)
        try:
            async for trade in agen:
                with self._data_lock:
                    self._trades[trade.event_symbol] = trade
        except asyncio.CancelledError:
            # let us cleanly close the async generator
            raise
        finally:
            await agen.aclose()

    async def handle_quotes(self) -> None:
        assert self.streamer is not None
        agen = self.streamer.listen(Quote)
        try:
            async for quote in agen:
                with self._data_lock:
                    self._quotes[quote.event_symbol] = quote
        except asyncio.CancelledError:
            # let us cleanly close the async generator
            raise
        finally:
            await agen.aclose()

    async def handle_greeks(self) -> None:
        assert self.streamer is not None
        agen = self.streamer.listen(Greeks)
        try:
            async for greeks in agen:
                with self._data_lock:
                    self._greeks[greeks.event_symbol] = greeks
        except asyncio.CancelledError:
            # let us cleanly close the async generator
            raise
        finally:
            await agen.aclose()
            with self._data_lock:
                self._greeks[greeks.event_symbol] = greeks

    async def _run(self) -> None:
        try:
            await self._connect_and_subscribe()

            self._tasks = []
            if self._trade_symbols:
                self._tasks.append(asyncio.create_task(self.handle_trades()))
            if self._quote_symbols:
                self._tasks.append(asyncio.create_task(self.handle_quotes()))
            if self._greek_symbols:
                self._tasks.append(asyncio.create_task(self.handle_greeks()))

        except BaseException as e:
            self._startup_exc = e
            raise
        finally:
            # Always release start() even if startup fails
            self._ready.set()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _stop_async(self) -> None:
        # cancel consumers
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # close streamer
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

            main_task = self._loop.create_task(self._run())
            self._main_task = main_task

            try:
                self._loop.run_until_complete(main_task)
            except asyncio.CancelledError:
                pass
            finally:
                loop = self._loop
                if loop is None or loop.is_closed():
                    return

                # 1) Cancel anything still pending
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                # 2) IMPORTANT: finish async-generator finalizers (fixes async_generator_athrow)
                loop.run_until_complete(loop.shutdown_asyncgens())

                # 3) (Optional but good) shutdown executor threads cleanly
                loop.run_until_complete(loop.shutdown_default_executor())

                loop.close()

        loop = self._loop
        if loop is not None and not loop.is_closed():
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
        self._thread = Thread(target=runner, daemon=True)
        self._thread.start()

        self._ready.wait()
        if self._startup_exc is not None:
            raise RuntimeError("MarketDataStreamer failed to start") from self._startup_exc

    def stop(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None:
            return

        # schedule async cleanup (optional but good)
        try:
            asyncio.run_coroutine_threadsafe(self._stop_async(), loop)
        except RuntimeError:
            pass

        # cancel the main task so run_until_complete can finish cleanly
        main_task = getattr(self, "_main_task", None)
        if main_task is not None and not main_task.done():
            loop.call_soon_threadsafe(main_task.cancel)

        # join from non-loop thread
        if thread and thread.is_alive() and current_thread() is not thread:
            thread.join(timeout=10)

        self._loop = None
        self._thread = None