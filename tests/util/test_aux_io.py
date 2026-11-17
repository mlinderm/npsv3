import dataclasses
import io
import os
import tarfile
import threading
import time

import pytest
from megatron.energon.cache.file_store import WebdatasetFileStore
from megatron.energon.epathlib import EPath

import npsv3.models.energon  # noqa: F401 -- installs the patches, as in production
from npsv3.models.energon import prepare_aux_tree
from npsv3.util import aux_io


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """`configure` changes process-wide settings and the environment; restore both (and drop the cache) afterwards"""
    for env in (
        aux_io.ENV_READ_BUFFER_BYTES, aux_io.ENV_AUX_BLOCK_BYTES, aux_io.ENV_AUX_BLOCK_CACHE_MBYTES,
        aux_io.ENV_AUX_MAX_OPEN_FILES,
    ):
        monkeypatch.setenv(env, os.environ.get(env, ""))
    saved = dataclasses.replace(aux_io.settings())
    yield
    for field in dataclasses.fields(saved):
        setattr(aux_io._settings, field.name, getattr(saved, field.name))
    if aux_io._cache is not None:
        aux_io._cache.close()
    aux_io._cache = None


def _random_file(path, size, seed=0):
    data = bytes((i * 131 + seed * 7 + (i >> 9)) & 0xFF for i in range(size))
    path.write_bytes(data)
    return data


class TestReadBuffer:
    def test_install_is_idempotent(self):
        import multistorageclient.file as msc_file

        patched = msc_file.PosixFile.__init__
        aux_io.install()
        aux_io._install_read_buffer_patch()
        assert msc_file.PosixFile.__init__ is patched
        assert patched._npsv3_read_buffer_patch
        assert not hasattr(patched._npsv3_original, "_npsv3_read_buffer_patch")  # Not wrapped twice

    def test_binary_reads_are_capped_and_correct(self, tmp_path, monkeypatch):
        import multistorageclient.file as msc_file

        aux_io.configure(read_buffer_bytes=64 * 1024)
        path = tmp_path / "data.bin"
        data = _random_file(path, 1 << 20)

        calls = []
        real_open = open

        def spy(*args, **kwargs):
            calls.append(kwargs.get("buffering"))
            return real_open(*args, **kwargs)

        monkeypatch.setattr(msc_file, "open", spy, raising=False)
        with EPath(path).open("rb") as f:
            f.seek(300_000)
            assert f.read(26 * 1024) == data[300_000 : 300_000 + 26 * 1024]
            # The buffer holds at most the cap (it would hold the rest of a 4 MiB st_blksize otherwise)
            assert 0 < len(f._file.peek()) <= 64 * 1024
            f.seek(10)
            assert f.read(5) == data[10:15]
        assert calls == [64 * 1024]

    def test_text_writes_and_explicit_buffering_unchanged(self, tmp_path, monkeypatch):
        import multistorageclient.file as msc_file

        aux_io.configure(read_buffer_bytes=64 * 1024)
        calls = []
        real_open = open

        def spy(*args, **kwargs):
            calls.append((kwargs.get("mode"), kwargs.get("buffering")))
            return real_open(*args, **kwargs)

        monkeypatch.setattr(msc_file, "open", spy, raising=False)
        path = EPath(tmp_path / "text.txt")
        path.write_text("hello\n")  # write mode, unchanged
        assert path.read_text() == "hello\n"  # text mode, unchanged
        storage_client = path.fs
        with storage_client.open(path._internal_str_path, "rb", buffering=4096) as f:  # explicit, unchanged
            assert f.read() == b"hello\n"
        assert [buffering for _mode, buffering in calls] == [-1, -1, 4096]

    def test_zero_disables(self, tmp_path):
        aux_io.configure(read_buffer_bytes=0)
        path = tmp_path / "data.bin"
        data = _random_file(path, 1 << 16)
        with EPath(path).open("rb") as f:
            assert f.read() == data

    def test_configure_exports_environment(self):
        aux_io.configure(read_buffer_bytes=12345, aux_block_cache_mbytes=7.5)
        assert os.environ[aux_io.ENV_READ_BUFFER_BYTES] == "12345"
        assert os.environ[aux_io.ENV_AUX_BLOCK_CACHE_MBYTES] == "7.5"
        assert aux_io.AuxIOSettings().read_buffer_bytes == 12345  # what a new process would read
        with pytest.raises(ValueError, match="must be >= 0"):
            aux_io.configure(read_buffer_bytes=-1)


class TestAuxBlockCache:
    def test_reads_match_file(self, tmp_path):
        path = tmp_path / "shard.tar"
        data = _random_file(path, 10_000 + 3)
        cache = aux_io.AuxBlockCache(block_bytes=1024, capacity_bytes=1 << 20)
        cases = [(0, 10), (1000, 100), (1020, 10), (1023, 1024), (9_990, 13), (9_990, 100), (0, 0), (500, 5000), (7, 9996)]
        for offset, size in cases:
            assert cache.read(str(path), offset, size) == data[offset : offset + size], (offset, size)
        assert cache.stats.direct_reads == 0
        cache.close()

    def test_direct_threshold(self, tmp_path):
        path = tmp_path / "shard.tar"
        data = _random_file(path, 10_000)
        cache = aux_io.AuxBlockCache(block_bytes=1024, capacity_bytes=1 << 20, direct_threshold=4096)
        assert cache.read(str(path), 500, 5000) == data[500:5500]
        assert cache.stats.direct_reads == 1
        assert cache.cached_bytes == 0
        cache.close()

    def test_span_is_one_read_and_caches_its_edges(self, tmp_path):
        path = tmp_path / "shard.tar"
        data = _random_file(path, 16 * 1024)
        cache = aux_io.AuxBlockCache(block_bytes=1024, capacity_bytes=1 << 20)
        # A large part covering blocks 2-7, then its neighbors
        assert cache.read(str(path), 2 * 1024 + 100, 5 * 1024 + 50) == data[2 * 1024 + 100 : 7 * 1024 + 150]
        assert cache.stats.preads == 1
        assert cache.stats.misses == 6
        assert cache.cached_bytes == 2 * 1024  # blocks 2 and 7 only
        assert cache.read(str(path), 7 * 1024 + 150, 200) == data[7 * 1024 + 150 : 7 * 1024 + 350]
        assert cache.read(str(path), 2 * 1024, 100) == data[2 * 1024 : 2 * 1024 + 100]
        assert cache.stats.preads == 1
        # Partly cached: only the missing blocks are read, again with one pread
        assert cache.read(str(path), 6 * 1024, 4 * 1024) == data[6 * 1024 : 10 * 1024]
        assert cache.stats.preads == 2
        cache.close()

    def test_consecutive_parts_share_reads(self, tmp_path):
        path = tmp_path / "shard.tar"
        data = _random_file(path, 64 * 1024)
        cache = aux_io.AuxBlockCache(block_bytes=16 * 1024, capacity_bytes=1 << 20)
        offset = 0
        while offset < len(data):
            size = min(700, len(data) - offset)
            assert cache.read(str(path), offset, size) == data[offset : offset + size]
            offset += size + 512  # a tar header between parts
        assert cache.stats.misses == 4
        assert cache.stats.preads == 4
        assert cache.stats.bytes_read == len(data)
        assert cache.stats.opens == 1
        cache.close()

    def test_lru_capacity(self, tmp_path):
        path = tmp_path / "shard.tar"
        _random_file(path, 8 * 1024)
        cache = aux_io.AuxBlockCache(block_bytes=1024, capacity_bytes=3 * 1024)
        for index in range(8):
            cache.read(str(path), index * 1024, 10)
        assert cache.cached_bytes == 3 * 1024
        assert cache.stats.evicted_blocks == 5
        cache.read(str(path), 7 * 1024, 10)  # Most recent: still cached
        assert cache.stats.hits == 1
        cache.read(str(path), 0, 10)  # Evicted
        assert cache.stats.misses == 9
        cache.close()

    def test_open_file_limit(self, tmp_path):
        paths = [tmp_path / f"{i}.tar" for i in range(5)]
        contents = [_random_file(path, 2048, seed=i) for i, path in enumerate(paths)]
        cache = aux_io.AuxBlockCache(block_bytes=1024, capacity_bytes=0, max_open_files=2)  # nothing cached
        for _ in range(2):
            for path, data in zip(paths, contents, strict=True):
                assert cache.read(str(path), 100, 10) == data[100:110]
        assert len(cache._handles) == 2
        assert cache.stats.opens == 10
        cache.close()

    def test_concurrent_misses_are_coalesced(self, tmp_path, monkeypatch):
        path = tmp_path / "shard.tar"
        data = _random_file(path, 4096)
        cache = aux_io.AuxBlockCache(block_bytes=4096, capacity_bytes=1 << 20)
        real_pread = os.pread

        def slow_pread(*args):
            time.sleep(0.2)
            return real_pread(*args)

        monkeypatch.setattr(os, "pread", slow_pread)
        results = [None] * 16
        barrier = threading.Barrier(len(results))

        def read(i):
            barrier.wait()
            results[i] = cache.read(str(path), i * 100, 50)

        threads = [threading.Thread(target=read, args=(i,)) for i in range(len(results))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == [data[i * 100 : i * 100 + 50] for i in range(len(results))]
        assert cache.stats.misses == 1
        assert cache.stats.preads == 1
        assert cache.stats.coalesced + cache.stats.hits == len(results) - 1
        cache.close()

    @pytest.mark.parametrize("capacity", [0, 3 * 1024, 1 << 20])
    def test_concurrent_overlapping_spans(self, tmp_path, capacity):
        paths = [tmp_path / f"{i}.tar" for i in range(3)]
        contents = {str(path): _random_file(path, 50_000 + i, seed=i) for i, path in enumerate(paths)}
        cache = aux_io.AuxBlockCache(block_bytes=1024, capacity_bytes=capacity, max_open_files=2)
        errors = []

        def worker(seed):
            import random

            rng = random.Random(seed)
            try:
                for _ in range(400):
                    path = rng.choice(list(contents))
                    offset = rng.randrange(0, 50_000)
                    size = rng.choice([1, 100, 1000, 1024, 3000, 9000])
                    assert cache.read(path, offset, size) == contents[path][offset : offset + size]
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert not any(thread.is_alive() for thread in threads), "deadlock"
        assert not errors, errors[0]
        assert not cache._inflight
        assert cache.cached_bytes <= capacity
        cache.close()

    def test_errors_reach_every_waiter(self, tmp_path):
        cache = aux_io.AuxBlockCache(block_bytes=1024, capacity_bytes=1 << 20)
        with pytest.raises(FileNotFoundError):
            cache.read(str(tmp_path / "missing.tar"), 0, 10)
        assert not cache._inflight

    @pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
    def test_fork_gets_a_fresh_cache(self, tmp_path):
        aux_io.configure(aux_block_cache_mbytes=16)
        path = tmp_path / "shard.tar"
        data = _random_file(path, 4096)
        parent = aux_io.get_aux_block_cache()
        assert parent.read(str(path), 10, 10) == data[10:20]
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child
            try:
                child = aux_io.get_aux_block_cache()
                ok = child is not parent and child.pid == os.getpid() and child.stats.opens == 0
                ok = ok and child.read(str(path), 10, 10) == data[10:20]
                os.write(write_fd, b"1" if ok else b"0")
            finally:
                os._exit(0)
        os.close(write_fd)
        result = os.read(read_fd, 1)
        os.waitpid(pid, 0)
        os.close(read_fd)
        assert result == b"1"
        assert aux_io.get_aux_block_cache() is parent


def _write_aux_tree(directory, num_samples=50, num_shards=2):
    directory.mkdir()
    contents = {}
    per_shard = num_samples // num_shards
    for shard in range(num_shards):
        with tarfile.open(directory / f"shard_{shard}.tar", "w") as tar:
            for i in range(shard * per_shard, (shard + 1) * per_shard):
                for ext, size in (("real.bin", 300 + 37 * i), ("sims.bin", 1500 + 101 * i)):
                    name = f"key{i:04d}.{ext}"
                    payload = os.urandom(size)
                    info = tarfile.TarInfo(name)
                    info.size = size
                    tar.addfile(info, io.BytesIO(payload))
                    contents[name] = payload
    prepare_aux_tree(directory, num_workers=1)
    return contents


class TestWebdatasetFileStore:
    def test_part_reads_use_block_cache(self, tmp_path):
        contents = _write_aux_tree(tmp_path / "imgs_ds")
        aux_io.configure(aux_block_bytes=16 * 1024, aux_block_cache_mbytes=4)

        store = WebdatasetFileStore(EPath(tmp_path / "imgs_ds"))
        cached = {name: store[name] for name in contents}
        cache = aux_io.get_aux_block_cache()
        assert cache.stats.hits > 0
        # Far fewer reads than parts: consecutive parts are adjacent
        assert cache.stats.preads < len(contents) / 4

        aux_io.configure(aux_block_cache_mbytes=0)
        assert aux_io.get_aux_block_cache() is None
        stock = {name: store[name] for name in contents}  # the stock energon path
        for name, payload in contents.items():
            assert cached[name][0] == payload == stock[name][0]
            assert cached[name][1] == stock[name][1]  # SourceInfo
        store.close()
