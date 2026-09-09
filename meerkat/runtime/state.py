from __future__ import annotations

import asyncio
from copy import deepcopy
import re
from typing import Any, Iterable

from meerkat.funnel.spec import StateUpdateSpec


class RuntimeState:
    def __init__(self, initial_values: dict[str, Any] | None = None) -> None:
        self._values: dict[str, Any] = deepcopy(initial_values or {})
        self._lock = asyncio.Lock()

    async def apply(
        self,
        updates: Iterable[StateUpdateSpec],
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._lock:
            applied: dict[str, Any] = {}
            for update in updates:
                changed, value = self._apply_update(update, evidence or {})
                if changed:
                    applied[update.key] = deepcopy(value)
            return applied

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return deepcopy(self._values)

    def _apply_update(self, update: StateUpdateSpec, evidence: dict[str, Any]) -> tuple[bool, Any]:
        operation = update.operation.lower()
        value = self._render_value(update.value, evidence)
        if operation == "increment":
            current = self._values.get(update.key, 0)
            self._values[update.key] = current + value
        elif operation == "increment_unique":
            values = self._coerce_unique_values(value)
            seen_key = f"__seen_{update.key}"
            seen = self._values.setdefault(seen_key, set())
            if not isinstance(seen, set):
                seen = set()
                self._values[seen_key] = seen
            new_values = [item for item in values if item not in seen]
            if not new_values:
                return False, self._values.get(update.key, 0)
            seen.update(new_values)
            current = self._values.get(update.key, 0)
            self._values[update.key] = current + len(new_values)
        elif operation == "set":
            self._values[update.key] = value
        elif operation == "append":
            current = self._values.setdefault(update.key, [])
            if not isinstance(current, list):
                current = []
                self._values[update.key] = current
            current.append(value)
        elif operation in {"set_text", "write_text"}:
            self._values[update.key] = "" if value is None else str(value)
        elif operation in {"append_text", "note"}:
            current_text = str(self._values.get(update.key, "") or "")
            next_text = "" if value is None else str(value)
            self._values[update.key] = next_text if not current_text else f"{current_text}\n{next_text}"
        else:
            raise ValueError(f"Unsupported state update operation: {update.operation}")
        return True, self._values[update.key]

    def _render_value(self, value: Any, evidence: dict[str, Any]) -> Any:
        if not isinstance(value, str):
            return value
        state_snapshot = self._values

        exact = re.fullmatch(r"\{(evidence|state)\.([A-Za-z_][A-Za-z0-9_]*)\}", value)
        if exact:
            namespace = exact.group(1)
            key = exact.group(2)
            source = evidence if namespace == "evidence" else state_snapshot
            return source.get(key, value)

        def replace(match: re.Match[str]) -> str:
            namespace = match.group(1)
            key = match.group(2)
            source = evidence if namespace == "evidence" else state_snapshot
            if key not in source:
                return match.group(0)
            return str(source[key])

        return re.sub(r"\{(evidence|state)\.([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)

    def _coerce_unique_values(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        elif isinstance(value, tuple) or isinstance(value, set):
            values = list(value)
        else:
            values = [value]
        return sorted({str(item).strip().lower() for item in values if str(item).strip()})
