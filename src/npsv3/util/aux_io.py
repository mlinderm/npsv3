"""I/O tuning for random-access Energon aux-store reads on parallel filesystems (GPFS, Lustre).

npsv3 reads each sample's pixels by key from Energon *aux* trees (`WebdatasetFileStore`, see
`npsv3.models.energon.cook_group`). With megatron-energon 7.4.1 and multistorageclient 1.1.0 every such read

1. opens the shard afresh, since `WebdatasetFileStore` passes `disable_cache=True` (needed for thread safety
   under `FileStoreCachePool`'s prefetch threads, NVIDIA/Megatron-Energon#202);
2. through multistorageclient's `PosixFile`, which opens with `buffering=-1`, so Python sizes its
   `BufferedReader` from `st_blksize` (e.g., 4 MiB on some GPFS setups; `EPath.open` ignores its `block_size` argument);
3. and then seeks and reads one ~70 KiB part, i.e. one full 4 MiB filesystem read per part, discarded at close.

Under load that read amplification (and the serialization it causes in the cache pool) stalls training. This module
provides two mitigations, installed by `install()` (called when `npsv3.models.energon` is imported, so forked
DataLoader workers inherit it):

- **Capped read buffer** (`read_buffer_bytes`, default 64 KiB): binary-mode, read-only `PosixFile`s opened with the
  default `buffering=-1` get a buffer of this size instead. Text mode, writes, and explicit `buffering` are unchanged.
- **Aux block cache** (`aux_block_bytes`, default 256 KiB; `aux_block_cache_mbytes`, default 512 MiB per process):
  `WebdatasetFileStore` part reads from local (POSIX) shards are served from a process-wide LRU of aligned
  blocks read with `os.pread` from per-process, LRU-bounded, unbuffered file descriptors. Consecutive samples of a
  dataset are adjacent in its pixel shards, and Energon cooks (and so prefetches) them in on-disk order within
  each sequence run, so one block read serves many parts. Concurrent misses for the same block are coalesced, and a
  part spanning several missing blocks reads them with one `pread` (caching only its edge blocks, which its neighbors
  share).

Each setting can be given by environment variable (`NPSV3_READ_BUFFER_BYTES`, `NPSV3_AUX_BLOCK_BYTES`,
`NPSV3_AUX_BLOCK_CACHE_MBYTES`, `NPSV3_AUX_MAX_OPEN_FILES`) or by `configure()`, which also exports them to the
environment so child processes agree. A value of 0 disables the corresponding mitigation.

The block cache's memory is *in addition to* the (on-disk) `FileStoreCachePool` cap, `max_cache_size_gbytes`: it is
held in RAM by every process that reads aux data, i.e. every DataLoader worker (train and val) of every rank.

Sizing: a worker reads from (blended datasets) x `parallel_shard_iters` (16 by default for training) sequences at once,
and a sequence's current block survives until its next read only if the cache holds about one block per sequence. Below
that the cache reads whole blocks that are evicted before they are used. Simulated with the real part layout and 102
datasets (`scripts/aux-io-benchmark/simulate_interleaved.py`), relative to the capped buffer alone (~1.45 reads and
1.19x the part bytes per part):

    parallel_shard_iters  block   cache    reads/part  bytes read / part bytes
    16 (1632 sequences)   256 KiB 512 MiB  0.53        1.23   (the defaults)
    16                    1 MiB   512 MiB  0.53        2.92   (thrashing: don't)
    16                    1 MiB   2 GiB    0.25        1.46
    4  (408 sequences)    256 KiB 512 MiB  0.37        1.01
    4                     1 MiB   1 GiB    0.17        1.06
    1  (102 sequences)    1 MiB   512 MiB  0.17        1.04
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ENV_READ_BUFFER_BYTES = "NPSV3_READ_BUFFER_BYTES"
ENV_AUX_BLOCK_BYTES = "NPSV3_AUX_BLOCK_BYTES"
ENV_AUX_BLOCK_CACHE_MBYTES = "NPSV3_AUX_BLOCK_CACHE_MBYTES"
ENV_AUX_MAX_OPEN_FILES = "NPSV3_AUX_MAX_OPEN_FILES"

DEFAULT_READ_BUFFER_BYTES = 64 * 1024
DEFAULT_AUX_BLOCK_BYTES = 256 * 1024
DEFAULT_AUX_BLOCK_CACHE_MBYTES = 512.0
DEFAULT_AUX_MAX_OPEN_FILES = 256


def _env_number(name: str, default, kind=int):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return kind(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r, using %r", name, value, default)
        return default


@dataclass
class AuxIOSettings:
    read_buffer_bytes: int = field(
        default_factory=lambda: _env_number(ENV_READ_BUFFER_BYTES, DEFAULT_READ_BUFFER_BYTES)
    )
    aux_block_bytes: int = field(default_factory=lambda: _env_number(ENV_AUX_BLOCK_BYTES, DEFAULT_AUX_BLOCK_BYTES))
    aux_block_cache_mbytes: float = field(
        default_factory=lambda: _env_number(ENV_AUX_BLOCK_CACHE_MBYTES, DEFAULT_AUX_BLOCK_CACHE_MBYTES, float)
    )
    aux_max_open_files: int = field(
        default_factory=lambda: _env_number(ENV_AUX_MAX_OPEN_FILES, DEFAULT_AUX_MAX_OPEN_FILES)
    )


_settings = AuxIOSettings()


def settings() -> AuxIOSettings:
    """The current settings (read from the environment at import, updated by `configure`)"""
    return _settings


def configure(
    *,
    read_buffer_bytes: int | None = None,
    aux_block_bytes: int | None = None,
    aux_block_cache_mbytes: float | None = None,
    aux_max_open_files: int | None = None,
) -> AuxIOSettings:
    """Update the settings (None leaves a setting unchanged) and export them to the environment, so processes started
    later (forked or spawned) use the same values. Changing the block cache settings discards this process's cache."""
    global _cache
    updates = {
        "read_buffer_bytes": (read_buffer_bytes, ENV_READ_BUFFER_BYTES, int),
        "aux_block_bytes": (aux_block_bytes, ENV_AUX_BLOCK_BYTES, int),
        "aux_block_cache_mbytes": (aux_block_cache_mbytes, ENV_AUX_BLOCK_CACHE_MBYTES, float),
        "aux_max_open_files": (aux_max_open_files, ENV_AUX_MAX_OPEN_FILES, int),
    }
    changed_cache = False
    for name, (given, env, kind) in updates.items():
        if given is None:
            continue
        value = kind(given)
        if value < 0:
            msg = f"{name} must be >= 0, got {value}"
            raise ValueError(msg)
        if name == "aux_block_bytes" and value == 0:
            msg = "aux_block_bytes must be > 0 (disable the block cache with aux_block_cache_mbytes=0)"
            raise ValueError(msg)
        if getattr(_settings, name) != value:
            changed_cache |= name != "read_buffer_bytes"
            setattr(_settings, name, value)
        os.environ[env] = str(value)
    if changed_cache:
        with _cache_lock:
            old, _cache = _cache, None
        if old is not None:
            old.close()
    return _settings


# --------------------------------------------------------------------------------------------------------------------
# Step 1: cap the read buffer of multistorageclient's PosixFile
# --------------------------------------------------------------------------------------------------------------------


def _is_binary_read_only(mode: str) -> bool:
    return "b" in mode and "r" in mode and not any(c in mode for c in "wax+")


def _install_read_buffer_patch() -> bool:
    try:
        import multistorageclient.file as msc_file
    except ImportError:  # pragma: no cover - multistorageclient is a dependency of energon
        return False

    cls = msc_file.PosixFile
    if getattr(cls.__init__, "_npsv3_read_buffer_patch", False):
        return True  # idempotent

    original = cls.__init__
    signature = inspect.signature(original)
    if "buffering" not in signature.parameters or "mode" not in signature.parameters:
        logger.warning("multistorageclient PosixFile.__init__%s has no mode/buffering; read buffer not capped", signature)
        return False

    def __init__(self, *args, **kwargs):  # noqa: N807
        cap = _settings.read_buffer_bytes
        if cap > 0:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = bound.arguments
            if arguments["buffering"] == -1 and _is_binary_read_only(arguments["mode"]):
                arguments["buffering"] = cap
                return original(*bound.args, **bound.kwargs)
        return original(self, *args, **kwargs)

    __init__._npsv3_read_buffer_patch = True
    __init__._npsv3_original = original
    __init__.__doc__ = original.__doc__
    cls.__init__ = __init__
    return True


# --------------------------------------------------------------------------------------------------------------------
# Step 2: a process-wide block cache for aux tar part reads
# --------------------------------------------------------------------------------------------------------------------


class _Handle:
    __slots__ = ("fd", "refs", "evicted")

    def __init__(self, fd: int):
        self.fd = fd
        self.refs = 0
        self.evicted = False


class _Pending:
    __slots__ = ("event", "data", "error")

    def __init__(self):
        self.event = threading.Event()
        self.data: bytes | None = None
        self.error: BaseException | None = None


@dataclass
class BlockCacheStats:
    hits: int = 0
    misses: int = 0
    coalesced: int = 0
    direct_reads: int = 0
    preads: int = 0
    bytes_read: int = 0
    opens: int = 0
    evicted_blocks: int = 0


class AuxBlockCache:
    """Thread-safe LRU of aligned file blocks, read with `os.pread`.

    Blocks are keyed by `(path, offset // block_bytes)`. A read spanning blocks joins them; a read larger than
    `direct_threshold` bytes (default 64 blocks) bypasses the cache. File descriptors are opened lazily, kept in an LRU of at most
    `max_open_files` (an evicted descriptor still in use is closed by its last user), and must not be shared
    across `fork` (see `get_aux_block_cache`).
    """

    def __init__(
        self,
        *,
        block_bytes: int = DEFAULT_AUX_BLOCK_BYTES,
        capacity_bytes: int = int(DEFAULT_AUX_BLOCK_CACHE_MBYTES * 1024 * 1024),
        max_open_files: int = DEFAULT_AUX_MAX_OPEN_FILES,
        direct_threshold: int | None = None,
    ):
        if block_bytes <= 0:
            msg = "block_bytes must be > 0"
            raise ValueError(msg)
        self.block_bytes = block_bytes
        self.capacity_bytes = capacity_bytes
        self.max_open_files = max(1, max_open_files)
        self.direct_threshold = 64 * block_bytes if direct_threshold is None else direct_threshold
        self.stats = BlockCacheStats()
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._blocks: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._size = 0
        self._inflight: dict[tuple[str, int], _Pending] = {}
        self._handles: OrderedDict[str, _Handle] = OrderedDict()

    @property
    def cached_bytes(self) -> int:
        return self._size

    # -- file descriptors ------------------------------------------------------------------------------------------

    def _acquire(self, path: str) -> _Handle:
        with self._lock:
            handle = self._handles.get(path)
            if handle is not None:
                handle.refs += 1
                self._handles.move_to_end(path)
                return handle
        # Open outside the lock: opens on a parallel filesystem can be slow
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        with self._lock:
            self.stats.opens += 1
            handle = self._handles.get(path)
            if handle is not None:
                os.close(fd)  # Another thread got there first
            else:
                handle = self._handles[path] = _Handle(fd)
            handle.refs += 1
            self._handles.move_to_end(path)
            while len(self._handles) > self.max_open_files:
                _, old = self._handles.popitem(last=False)
                old.evicted = True
                if old.refs == 0:
                    os.close(old.fd)
            return handle

    def _release(self, handle: _Handle) -> None:
        with self._lock:
            handle.refs -= 1
            if handle.evicted and handle.refs == 0:
                os.close(handle.fd)

    def _pread(self, path: str, offset: int, size: int) -> bytes:
        """Read up to `size` bytes at `offset` (fewer only at EOF)"""
        handle = self._acquire(path)
        try:
            chunks = []
            remaining = size
            preads = 0
            while remaining > 0:
                chunk = os.pread(handle.fd, remaining, offset)
                preads += 1
                if not chunk:
                    break
                chunks.append(chunk)
                offset += len(chunk)
                remaining -= len(chunk)
        finally:
            self._release(handle)
        data = chunks[0] if len(chunks) == 1 else b"".join(chunks)
        with self._lock:
            self.stats.preads += preads
            self.stats.bytes_read += len(data)
        return data

    # -- blocks ----------------------------------------------------------------------------------------------------

    def read(self, path: str, offset: int, size: int) -> bytes:
        """Return `size` bytes of `path` at `offset` (fewer only at EOF, like `file.read`).

        The blocks the range covers that are neither cached nor being read by another thread are read with one
        `pread`. Of those, only the range's first and last blocks are cached: they may hold the neighboring parts,
        while a large part's interior blocks hold nothing else.
        """
        if size <= 0:
            return b""
        block_bytes = self.block_bytes
        first = offset // block_bytes
        last = (offset + size - 1) // block_bytes
        if self.capacity_bytes <= 0 or size > self.direct_threshold:
            with self._lock:
                self.stats.direct_reads += 1
            return self._pread(path, offset, size)

        blocks: dict[int, bytes | memoryview] = {}
        waits: list[tuple[int, _Pending]] = []
        owned: list[tuple[int, _Pending]] = []
        with self._lock:
            for index in range(first, last + 1):
                key = (path, index)
                data = self._blocks.get(key)
                if data is not None:
                    self._blocks.move_to_end(key)
                    self.stats.hits += 1
                    blocks[index] = data
                elif (pending := self._inflight.get(key)) is not None:
                    self.stats.coalesced += 1
                    waits.append((index, pending))
                else:
                    pending = self._inflight[key] = _Pending()
                    self.stats.misses += 1
                    owned.append((index, pending))

        if owned:
            low, high = owned[0][0], owned[-1][0]
            try:
                raw = self._pread(path, low * block_bytes, (high - low + 1) * block_bytes)
            except BaseException as error:
                with self._lock:
                    for index, pending in owned:
                        pending.error = error
                        del self._inflight[(path, index)]
                for _, pending in owned:
                    pending.event.set()
                raise
            view = memoryview(raw)
            with self._lock:
                for index, pending in owned:
                    chunk = view[(index - low) * block_bytes : (index - low + 1) * block_bytes]
                    del self._inflight[(path, index)]
                    if index in (first, last) and len(chunk) <= self.capacity_bytes:
                        chunk = pending.data = bytes(chunk) if len(owned) > 1 else raw
                        self._blocks[(path, index)] = chunk
                        self._size += len(chunk)
                    else:
                        pending.data = chunk
                    blocks[index] = chunk
                while self._size > self.capacity_bytes:
                    _, old = self._blocks.popitem(last=False)
                    self._size -= len(old)
                    self.stats.evicted_blocks += 1
            # Before waiting on others, so that two readers of overlapping ranges can't wait on each other
            for _, pending in owned:
                pending.event.set()

        for index, pending in waits:
            pending.event.wait()
            if pending.error is not None:
                raise pending.error
            blocks[index] = pending.data

        start = offset - first * block_bytes
        if first == last:
            return bytes(blocks[first][start : start + size])
        return b"".join(blocks[index] for index in range(first, last + 1))[start : start + size]

    def close(self) -> None:
        """Drop the cached blocks and close every descriptor not currently in use (the rest close on release)"""
        with self._lock:
            self._blocks.clear()
            self._size = 0
            handles, self._handles = list(self._handles.values()), OrderedDict()
            for handle in handles:
                handle.evicted = True
                if handle.refs == 0:
                    os.close(handle.fd)


_cache: AuxBlockCache | None = None
_cache_lock = threading.Lock()


def get_aux_block_cache() -> AuxBlockCache | None:
    """This process's block cache, created on first use, or None if disabled (`aux_block_cache_mbytes=0`)"""
    global _cache
    cache = _cache
    if cache is not None and cache.pid == os.getpid():
        return cache
    if _settings.aux_block_cache_mbytes <= 0:
        return None
    with _cache_lock:
        if _cache is None or _cache.pid != os.getpid():
            _cache = AuxBlockCache(
                block_bytes=_settings.aux_block_bytes,
                capacity_bytes=int(_settings.aux_block_cache_mbytes * 1024 * 1024),
                max_open_files=_settings.aux_max_open_files,
            )
        return _cache


def _after_fork_in_child() -> None:
    """Forget the parent's cache: its locks may be held and its in-flight reads will never complete in the child.
    Close the inherited descriptors so they don't count against the child's limit."""
    global _cache, _cache_lock
    parent, _cache = _cache, None
    _cache_lock = threading.Lock()
    if parent is not None:
        for handle in list(parent._handles.values()):
            with contextlib.suppress(OSError):
                os.close(handle.fd)


def _install_block_cache_patch() -> bool:
    from megatron.energon.cache.file_store import WebdatasetFileStore
    from megatron.energon.flavors.webdataset.itar_reader import ITarReader
    from megatron.energon.source_info import SourceInfo

    current = WebdatasetFileStore.__dict__.get("_get_part_by_raw_sample_pointer")
    if getattr(current, "_npsv3_block_cache_patch", False):
        return True  # idempotent

    original = ITarReader._get_part_by_raw_sample_pointer
    if list(inspect.signature(original).parameters) != ["self", "raw_sample_pointer", "entry_name"]:
        logger.warning("Unexpected ITarReader._get_part_by_raw_sample_pointer signature; aux block cache disabled")
        return False

    def _posix_path(store, tar_file_id: int) -> str | None:
        paths = store.__dict__.get("_npsv3_posix_paths")
        if paths is None:
            paths = store.__dict__.setdefault("_npsv3_posix_paths", {})
        try:
            return paths[tar_file_id]
        except KeyError:
            epath = store.tar_filepaths[tar_file_id]
            path = epath.fs.get_posix_path(epath._internal_str_path)
            paths[tar_file_id] = path
            return path

    def _get_part_by_raw_sample_pointer(self, raw_sample_pointer, entry_name):
        cache = get_aux_block_cache() if self.disable_cache else None
        path = _posix_path(self, raw_sample_pointer.tar_file_id) if cache is not None else None
        if path is None:
            return original(self, raw_sample_pointer, entry_name)
        raw_data = cache.read(path, raw_sample_pointer.raw_byte_offset, raw_sample_pointer.raw_byte_size)
        return raw_data, SourceInfo(
            dataset_path=self.base_path,
            index=entry_name,
            shard_name=self.tar_filenames[raw_sample_pointer.tar_file_id],
            file_names=(entry_name,),
        )

    _get_part_by_raw_sample_pointer._npsv3_block_cache_patch = True
    _get_part_by_raw_sample_pointer.__doc__ = original.__doc__
    WebdatasetFileStore._get_part_by_raw_sample_pointer = _get_part_by_raw_sample_pointer
    return True


_installed = False


def install() -> None:
    """Install both patches (idempotent). The settings are read at each call, so `configure` can change them later."""
    global _installed
    if _installed:
        return
    _install_read_buffer_patch()
    _install_block_cache_patch()
    os.register_at_fork(after_in_child=_after_fork_in_child)
    _installed = True
