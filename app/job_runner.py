"""Background runner for confirmed upload jobs (one container, one pool).

The confirm route hands the pipeline run to this pool and redirects at
once, so a minutes-long clip never rides a web request. At most
``max_workers`` jobs run together; later confirmations wait in the
pool's queue. Nothing here is durable: the job row is the truth, and a
container stop leaves its running jobs for the hosted boot to fail
honestly (:func:`app.service.fail_interrupted_jobs`). Cancel and expiry
keep working through their own durable paths while a job runs.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("partfidelity.jobs")


class JobRunner:
    """Run one callable per job id off the request thread."""

    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pf-job"
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[Any]] = {}
        self._closed = False

    def submit(self, job_id: str, run: Callable[[], Any]) -> bool:
        """Queue ``run`` for ``job_id``; False when that job is already queued."""
        with self._lock:
            if self._closed:
                raise RuntimeError("job runner is closed")
            current = self._futures.get(job_id)
            if current is not None and not current.done():
                return False
            future = self._pool.submit(self._run, job_id, run)
            self._futures[job_id] = future
        future.add_done_callback(lambda done: self._forget(job_id, done))
        return True

    def _run(self, job_id: str, run: Callable[[], Any]) -> Any:
        try:
            return run()
        except Exception as error:
            # The job's own failure paths settle the row; this only keeps
            # an unexpected error from vanishing with the thread.
            logger.error("background job=%s raised: %s", job_id, type(error).__name__)
            raise

    def _forget(self, job_id: str, done: Future[Any]) -> None:
        with self._lock:
            if self._futures.get(job_id) is done:
                del self._futures[job_id]

    def wait(self, job_id: str, timeout: float | None = None) -> None:
        """Block until ``job_id``'s run ends (tests, shutdown); errors swallowed."""
        with self._lock:
            future = self._futures.get(job_id)
        if future is None:
            return
        try:
            future.result(timeout=timeout)
        except Exception:
            pass

    def shutdown(self) -> None:
        """Stop taking work; queued runs that have not started are dropped."""
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)


__all__ = ("JobRunner",)
