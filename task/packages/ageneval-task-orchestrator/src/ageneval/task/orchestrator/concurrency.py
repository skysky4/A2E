"""Controller-owned concurrency pools and crash-safe permit accounting."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PermitPool:
    """An observable asyncio semaphore used only in the Controller process."""

    limit: int
    _semaphore: asyncio.Semaphore = field(init=False)
    active: int = 0
    high_water: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("permit limit must be positive")
        self._semaphore = asyncio.Semaphore(self.limit)

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        # There is deliberately no await between the semaphore acquisition and
        # accounting. Cancellation therefore cannot create an unrecorded lease.
        self.active += 1
        self.high_water = max(self.high_water, self.active)

    def release(self) -> None:
        if self.active <= 0:
            raise RuntimeError("permit pool released without an active permit")
        self.active -= 1
        self._semaphore.release()

    @asynccontextmanager
    async def permit(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()


class PermitLedger:
    """Small local helper retained for callers that need scoped acquisitions."""

    def __init__(self) -> None:
        self._pools: list[PermitPool] = []

    async def acquire(self, pool: PermitPool) -> None:
        await pool.acquire()
        self._pools.append(pool)

    def release(self, pool: PermitPool) -> None:
        for index in range(len(self._pools) - 1, -1, -1):
            if self._pools[index] is pool:
                self._pools.pop(index)
                pool.release()
                return
        raise RuntimeError("permit was not held by this ledger")

    def release_all(self) -> None:
        while self._pools:
            self._pools.pop().release()


class PermitBroker:
    """The sole owner of Campaign permits.

    Workers request lifecycle transitions over IPC. The Controller performs
    both the semaphore operation and ledger update in one event loop, removing
    the acquire/write and delete/release races of a multiprocessing ledger.
    """

    def __init__(self, limits: Mapping[str, int]) -> None:
        self._pools = {name: PermitPool(limit) for name, limit in limits.items()}
        self._held: dict[str, list[str]] = defaultdict(list)

    async def acquire(self, owner: str, resource: str) -> None:
        if resource in self._held.get(owner, []):
            raise RuntimeError(f"{owner} already holds {resource}")
        try:
            pool = self._pools[resource]
        except KeyError as exc:
            raise KeyError(f"unknown permit resource: {resource}") from exc
        await pool.acquire()
        self._held[owner].append(resource)

    async def acquire_many(self, owner: str, resources: tuple[str, ...]) -> None:
        """Acquire resources in order and roll back this batch on failure."""
        acquired: list[str] = []
        try:
            for resource in resources:
                await self.acquire(owner, resource)
                acquired.append(resource)
        except BaseException:
            for resource in reversed(acquired):
                self.release(owner, resource)
            raise

    def release(self, owner: str, resource: str) -> None:
        held = self._held.get(owner)
        if not held or resource not in held:
            raise RuntimeError(f"{owner} does not hold {resource}")
        # Remove the ledger entry and release synchronously in one event-loop
        # turn; no worker can die between these two Controller-owned actions.
        reverse_index = held[::-1].index(resource)
        held.pop(len(held) - reverse_index - 1)
        self._pools[resource].release()
        if not held:
            self._held.pop(owner, None)

    def release_many(self, owner: str, resources: tuple[str, ...]) -> None:
        """Release a lifecycle batch after validating every lease exists."""
        held = self._held.get(owner, [])
        missing = [resource for resource in resources if resource not in held]
        if missing:
            raise RuntimeError(f"{owner} does not hold resources: {missing}")
        for resource in resources:
            self.release(owner, resource)

    def release_all(self, owner: str) -> None:
        held = self._held.pop(owner, [])
        for resource in reversed(held):
            self._pools[resource].release()

    def release_prefix(self, prefix: str) -> None:
        for owner in [item for item in self._held if item.startswith(prefix)]:
            self.release_all(owner)

    def held(self, owner: str) -> tuple[str, ...]:
        return tuple(self._held.get(owner, ()))

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "limit": pool.limit,
                "active": pool.active,
                "high_water": pool.high_water,
            }
            for name, pool in self._pools.items()
        }


class RuntimeMetrics:
    """Controller-owned counters for real processes and blocking activity."""

    def __init__(self) -> None:
        self.processes = {
            "active": 0,
            "high_water": 0,
            "started": 0,
            "completed": 0,
            "crashed": 0,
            "force_killed": 0,
        }
        self.activities: dict[str, dict[str, int]] = defaultdict(
            lambda: {"active": 0, "high_water": 0, "started": 0}
        )
        self._activity_owners: dict[str, list[str]] = defaultdict(list)

    def process_event(self, state: str, _pid: int) -> None:
        if state == "start":
            self.processes["active"] += 1
            self.processes["started"] += 1
            self.processes["high_water"] = max(
                self.processes["high_water"], self.processes["active"]
            )
            return
        if self.processes["active"] <= 0:
            raise RuntimeError("Trial process accounting underflow")
        self.processes["active"] -= 1
        self.processes["completed"] += 1
        if state == "killed":
            self.processes["force_killed"] += 1

    def process_crashed(self) -> None:
        self.processes["crashed"] += 1

    def activity(self, owner: str, name: str, state: str) -> None:
        counter = self.activities[name]
        if state == "start":
            counter["active"] += 1
            counter["started"] += 1
            counter["high_water"] = max(counter["high_water"], counter["active"])
            self._activity_owners[owner].append(name)
        elif state == "end":
            if counter["active"] <= 0:
                raise RuntimeError(f"activity accounting underflow: {name}")
            counter["active"] -= 1
            held = self._activity_owners.get(owner, [])
            if name in held:
                held.remove(name)
            if not held:
                self._activity_owners.pop(owner, None)
        else:
            raise ValueError(f"unknown activity state: {state}")

    def release_activities(self, owner: str) -> None:
        for name in self._activity_owners.pop(owner, []):
            counter = self.activities[name]
            if counter["active"] > 0:
                counter["active"] -= 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "trial_processes": dict(self.processes),
            "activities": {
                name: dict(counter) for name, counter in sorted(self.activities.items())
            },
        }
