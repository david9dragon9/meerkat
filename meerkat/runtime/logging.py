from __future__ import annotations

from time import perf_counter
from typing import Optional


class RuntimeLogger:
    """Timestamped run log, on two clocks.

    *Wall* is time spent actually processing the stream, and *stream* is the
    position in the media the line is about. They are different numbers: a
    model request that starts at stream 4.5s may only answer at wall 5.4s.

    The wall clock starts when the first media event arrives and stops while
    ingest is paused, so it always reads as time the stream was being worked
    on. Without that, pausing playback for a minute would silently add a
    minute to every subsequent line.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._start = perf_counter()
        self._paused_at: Optional[float] = None
        self._paused_seconds = 0.0
        self._stream_started = False

    def reset_wall_clock(self) -> None:
        """Start the wall clock, discarding anything before the first event."""
        self._start = perf_counter()
        self._paused_at = None
        self._paused_seconds = 0.0
        self._stream_started = True

    def pause(self) -> None:
        """Stop counting wall time; ingest is not running."""
        if self._paused_at is None:
            self._paused_at = perf_counter()

    def resume(self) -> None:
        if self._paused_at is not None:
            self._paused_seconds += perf_counter() - self._paused_at
            self._paused_at = None

    @property
    def paused(self) -> bool:
        return self._paused_at is not None

    def elapsed_seconds(self) -> float:
        now = self._paused_at if self._paused_at is not None else perf_counter()
        return max(0.0, now - self._start - self._paused_seconds)

    def elapsed_ms(self) -> int:
        return int(self.elapsed_seconds() * 1000)

    def log(self, stream_time_ms: Optional[int], message: str, bold: bool = False) -> None:
        if not self.enabled:
            return
        rendered_message = f"\033[1m{message}\033[0m" if bold else message
        if not self._stream_started:
            # Nothing has been ingested yet, so there is no wall clock to read
            # against; mark these as pre-stream rather than implying otherwise.
            print(f"PRE: {self.elapsed_seconds():7.2f}: {rendered_message}", flush=True)
            return
        stream_suffix = "" if stream_time_ms is None else f" stream={stream_time_ms / 1000:.2f}s"
        print(f"{self.elapsed_seconds():7.2f}: {rendered_message}{stream_suffix}", flush=True)
