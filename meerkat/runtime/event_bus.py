from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional

from meerkat.events import EventType, StreamEvent


class EventBus:
    """Async pub/sub bus with per-subscriber queues.

    Publishing fans out each event to every subscriber interested in that event
    type. Subscribers consume independently, which lets cheap gates run in
    parallel without blocking ingest.
    """

    def __init__(self, queue_size: int = 256) -> None:
        self._queue_size = queue_size
        self._subscribers: DefaultDict[EventType, List[asyncio.Queue[StreamEvent]]] = defaultdict(list)
        self._all_subscribers: List[asyncio.Queue[StreamEvent]] = []

    def subscribe(
        self,
        event_types: Optional[Iterable[EventType]] = None,
        queue_size: Optional[int] = None,
    ) -> asyncio.Queue[StreamEvent]:
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=queue_size or self._queue_size)
        if event_types is None:
            self._all_subscribers.append(queue)
            return queue
        for event_type in event_types:
            self._subscribers[event_type].append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[StreamEvent]) -> None:
        self._all_subscribers = [item for item in self._all_subscribers if item is not queue]
        for event_type, queues in list(self._subscribers.items()):
            self._subscribers[event_type] = [item for item in queues if item is not queue]

    async def publish(self, event: StreamEvent) -> None:
        queues = list(self._all_subscribers)
        queues.extend(self._subscribers.get(event.type, []))
        for queue in queues:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    async def publish_many(self, events: Iterable[StreamEvent]) -> None:
        for event in events:
            await self.publish(event)

    def subscriber_counts(self) -> Dict[str, int]:
        counts = {event_type.value: len(queues) for event_type, queues in self._subscribers.items()}
        counts["*"] = len(self._all_subscribers)
        return counts
