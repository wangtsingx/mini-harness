"""Run supervision: hard wall-clock deadline, cancellation and backpressure around an event stream.

The loop runs in its own Task and pushes events through a bounded queue:
  * deadline hit  -> the Task is cancelled (in-flight model call / tools are torn down; the loop's
                     `finally` repairs history), queued events are flushed, then `on_timeout()` is emitted;
  * consumer stops (break / aclose / cancelled) -> the Task is cancelled and awaited, nothing leaks;
  * slow consumer -> bounded queue applies backpressure to the producer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from mini_harness.core.events import Event

_END = object()


class _Failure:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


async def _stop(task: asyncio.Task[Any]) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def supervise(
    events: AsyncGenerator[Event, None],
    *,
    timeout_s: float,
    on_timeout: Callable[[], Event],
    queue_size: int = 64,
) -> AsyncIterator[Event]:
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_size)

    async def pump() -> None:
        try:
            async for ev in events:
                await queue.put(ev)
            await queue.put(_END)
        except Exception as e:  # noqa: BLE001 - forwarded to the consumer
            await queue.put(_Failure(e))
        finally:
            await events.aclose()  # runs the loop's cleanup even if cancelled while suspended at a yield

    task = asyncio.create_task(pump(), name="harness-run")
    deadline = asyncio.get_running_loop().time() + timeout_s
    try:
        while True:
            try:
                async with asyncio.timeout_at(deadline):
                    item = await queue.get()
            except TimeoutError:
                await _stop(task)
                while not queue.empty():  # flush what was produced before the deadline
                    leftover = queue.get_nowait()
                    if leftover is not _END and not isinstance(leftover, _Failure):
                        yield leftover
                yield on_timeout()
                return
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.exc
            yield item
    finally:
        await _stop(task)
