from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter
from typing import DefaultDict, Dict, List


@dataclass
class Counter:
    count: int = 0

    def inc(self, value: int = 1) -> None:
        self.count += value


@dataclass
class TimerSeries:
    samples_ms: List[float] = field(default_factory=list)

    def add(self, elapsed_ms: float) -> None:
        self.samples_ms.append(elapsed_ms)

    def summary(self) -> Dict[str, float]:
        if not self.samples_ms:
            return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        ordered = sorted(self.samples_ms)
        p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return {
            "count": float(len(ordered)),
            "avg_ms": mean(ordered),
            "p95_ms": ordered[p95_index],
            "max_ms": ordered[-1],
        }


class Metrics:
    def __init__(self) -> None:
        self.counters: DefaultDict[str, Counter] = defaultdict(Counter)
        self.timers: DefaultDict[str, TimerSeries] = defaultdict(TimerSeries)

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name].inc(value)

    def start_timer(self) -> float:
        return perf_counter()

    def stop_timer(self, name: str, start: float) -> None:
        self.timers[name].add((perf_counter() - start) * 1000)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "counters": {name: float(counter.count) for name, counter in self.counters.items()},
            "timers": {name: timer.summary() for name, timer in self.timers.items()},
        }
