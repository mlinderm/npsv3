"""`FileStoreCachePool` with two fixes for megatron-energon 7.4.x (both worth reporting upstream).

1. `_get_data` cancels a prefetch that hasn't started and reads the data directly *while holding the pool's lock*.
   Every prefetch thread needs that lock after its read, and every `get_lazy` before it, so under slow I/O the
   pool serializes behind one direct read at a time. Here the direct read happens without the lock.
2. `_cache_out_task` reserves cache space and then, if every reference was dropped in the meantime (e.g., a cooked
   sample that was later rejected or filtered), returns without releasing the reservation. The leaked size and count
   accumulate until prefetches wait for space indefinitely. Here the reservation is returned.

On other energon versions the stock pool behavior is used (with a warning), since these overrides copy its internals.
"""

from __future__ import annotations

import logging
import pickle
from typing import TYPE_CHECKING

from megatron.energon import FileStoreCachePool

if TYPE_CHECKING:
    from megatron.energon.cache.base import FileStore
    from megatron.energon.cache.file_cache_pool import _PendingTask
    from megatron.energon.source_info import SourceInfo

logger = logging.getLogger(__name__)


def _energon_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("megatron-energon")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


SUPPORTED_ENERGON = ("7.4.",)
_OVERRIDES_ENABLED = _energon_version().startswith(SUPPORTED_ENERGON)
if not _OVERRIDES_ENABLED:  # pragma: no cover
    logger.warning(
        "megatron-energon %s is not one of %s: using the stock FileStoreCachePool (lock and reservation fixes off)",
        _energon_version(), SUPPORTED_ENERGON,
    )


class CachePool(FileStoreCachePool):
    """Drop-in `FileStoreCachePool` (see the module docstring)"""

    def _get_data(self, ds: FileStore, fname: str, entry: _PendingTask) -> tuple[object, SourceInfo]:
        if not _OVERRIDES_ENABLED:  # pragma: no cover
            return super()._get_data(ds, fname, entry)

        with self._lock:
            cancelled = entry.send_to_cache_future.cancel()
        if not cancelled:
            # The prefetch is running or done (a running future can't become pending again, so the stock method's own
            # `cancel()` fails too). That branch already waits without holding the lock.
            return super()._get_data(ds, fname, entry)

        # The prefetch never ran: read directly, without holding the lock
        try:
            result = ds[fname]
        finally:
            with self._lock:
                self._decrement_refcount_and_cleanup(key=(ds.get_path(), fname))
                # Requeue for the remaining references, unless another reader that also found the prefetch
                # cancelled already requeued it (two cache-out tasks for one entry would collide on its cache_path)
                if entry.refcount > 0 and not self._shutting_down and entry.send_to_cache_future.cancelled():
                    entry.send_to_cache_future = self._worker_pool.submit(self._cache_out_task, ds, fname, entry)
        return result

    def _cache_out_task(self, ds: FileStore, fname: str, entry: _PendingTask) -> bool:
        if not _OVERRIDES_ENABLED:  # pragma: no cover
            return super()._cache_out_task(ds, fname, entry)

        # Copied from energon 7.4.1, except as noted
        with self._lock:
            if self._shutting_down:
                return False

        if self.method == "raw":
            data, entry.source_info = ds._get_raw(fname)
        elif self.method == "pickle":
            data, entry.source_info = ds[fname]
            data = pickle.dumps(data)
        else:
            msg = f"Invalid method: {self.method}"
            raise ValueError(msg)

        with self._lock:
            entry.data_size = file_size = len(data)
            while (
                self.current_cache_count + 1 > self.max_cache_count
                or self.current_cache_size + entry.data_size > self.max_cache_size
            ):
                self._cache_space_available.wait()
                if entry.require_data_now or self._shutting_down:
                    return False

            if self._shutting_down or entry.refcount <= 0:
                # Changed: check *before* reserving, so an abandoned entry doesn't leak its reservation
                return False
            self.current_cache_size += file_size
            self.current_cache_count += 1

        try:
            assert entry.cache_path is None, f"cache_path should be None, but is {entry.cache_path!r}"
            cache_path = self._make_cache_path(ds, fname)
            self._write_to_cache(cache_path, data)
        except:
            with self._lock:
                self.current_cache_size -= file_size
                self.current_cache_count -= 1
                self._cache_space_available.notify_all()
            raise
        else:
            with self._lock:
                entry.cache_path = cache_path
                if entry.refcount <= 0:
                    # Changed: every reference was dropped while writing, after which nothing would return its
                    # reservation, or remove the file. The cache path depends only on (ds, fname), so leave the file
                    # to a newer entry for the same key, if there is one.
                    if self._pending_tasks.get((ds.get_path(), fname)) is None:
                        self._remove_cached_file(entry)
                    else:
                        entry.cache_path = None
                        self.current_cache_size -= file_size
                        self.current_cache_count -= 1
                        self._cache_space_available.notify_all()
                    return False
        return True
