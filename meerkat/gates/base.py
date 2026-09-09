from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from typing import Iterable, Optional

from meerkat.events import EventType, GateFire, StreamEvent
from meerkat.runtime.logging import RuntimeLogger


class Gate(ABC):
    def __init__(self, gate_id: str, input_types: Iterable[EventType]) -> None:
        self.gate_id = gate_id
        self.input_types = tuple(input_types)
        self.logger: Optional[RuntimeLogger] = None
        self.queue_size = 256
        self.worker_count = 1
        self.runtime_queue: Optional[asyncio.Queue[StreamEvent]] = None
        self.terminal_response = False

    def set_logger(self, logger: RuntimeLogger) -> None:
        self.logger = logger

    def set_runtime_queue(self, queue: asyncio.Queue[StreamEvent]) -> None:
        self.runtime_queue = queue

    async def warmup(self) -> None:
        return None

    @abstractmethod
    async def process(self, event: StreamEvent) -> Optional[GateFire]:
        raise NotImplementedError
