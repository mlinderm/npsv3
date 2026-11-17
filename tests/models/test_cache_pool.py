import gc
import threading
import time

from megatron.energon.cache.base import FileStore
from megatron.energon.source_info import SourceInfo

from npsv3.models.cache_pool import CachePool


class _Store(FileStore):
    def __init__(self, delay=0.0, on_read=None):
        self.delay = delay
        self.on_read = on_read
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        if self.on_read is not None:
            self.on_read(key)
        time.sleep(self.delay)
        return key.encode() * 100, SourceInfo(dataset_path="d", index=key, shard_name="s", file_names=(key,))

    def get_path(self):
        return "store"


def test_abandoned_prefetches_release_their_reservation(tmp_path):
    pool = CachePool(parent_cache_dir=tmp_path, num_workers=4)
    ds = _Store(delay=0.05)
    for i in range(20):
        lazy = pool.get_lazy(ds, f"k{i}")
        time.sleep(0.01)  # The prefetch is running
        del lazy
        gc.collect()
    time.sleep(0.5)
    pool.close()
    assert not pool._pending_tasks
    assert pool.current_cache_size == 0
    assert pool.current_cache_count == 0
    assert not any(pool.cache_dir.iterdir())


def test_direct_read_does_not_hold_the_lock(tmp_path):
    pool = CachePool(parent_cache_dir=tmp_path, num_workers=1)
    release = threading.Event()
    held = []

    def on_read(key):
        if key == "blocker":
            release.wait(5)
        else:
            held.append(pool._lock._is_owned())

    ds = _Store(on_read=on_read)
    blocker = pool.get_lazy(ds, "blocker")  # Occupies the only worker
    time.sleep(0.05)
    lazies = [pool.get_lazy(ds, "a"), pool.get_lazy(ds, "a"), pool.get_lazy(ds, "b")]
    assert lazies[0].get() == b"a" * 100  # Cancelled prefetch -> direct read
    release.set()
    assert blocker.get() == b"blocker" * 100
    assert lazies[1].get() == b"a" * 100  # Requeued for the second reference
    assert lazies[2].get() == b"b" * 100
    assert held
    assert not any(held)
    del blocker, lazies
    gc.collect()
    pool.close()
    assert pool.current_cache_size == 0
    assert not pool._pending_tasks


def test_matches_stock_results(tmp_path):
    pool = CachePool(parent_cache_dir=tmp_path, num_workers=3)
    ds = _Store(delay=0.002)
    lazies = [pool.get_lazy(ds, f"k{i % 25}") for i in range(200)]
    sample = {"__sources__": ()}
    for i, lazy in enumerate(lazies):
        assert lazy.get(sample) == f"k{i % 25}".encode() * 100
    assert len(sample["__sources__"]) == 200
    del lazies
    gc.collect()
    pool.close()
    assert pool.current_cache_size == 0
    assert pool.current_cache_count == 0
