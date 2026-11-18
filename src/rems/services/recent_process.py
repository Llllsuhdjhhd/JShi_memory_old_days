"""Process boundary only: all recent memory operations run in native Mem0."""
from __future__ import annotations

import multiprocessing
import threading

from .recent_memory import Mem0RecentMemory


class RecentWorkerError(RuntimeError):
    pass


def _worker(connection, configuration):
    import signal
    # The owning JShi process drains work and sends close. Console Ctrl+C must
    # not kill the native store before that drain has finished.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from ..config import REMSConfig
    recent = Mem0RecentMemory(REMSConfig(**configuration, _env_file=None))
    try:
        while True:
            try:
                request_id, method, args, kwargs = connection.recv()
            except EOFError:
                break
            try:
                if method == "close":
                    recent.close()
                    connection.send((request_id, True, None))
                    break
                if method not in {"prepare", "recall", "ingest_batch", "maintain", "disk_bytes"}:
                    raise ValueError("Unsupported recent-memory operation")
                result = getattr(recent, method)(*args, **kwargs)
                connection.send((request_id, True, result))
            except Exception as exc:
                connection.send((request_id, False, f"{type(exc).__name__}: {exc}"))
    finally:
        recent.close()
        connection.close()


class Mem0RecentProcess:
    """One owner of the recent DB, with serialized IPC and acknowledged writes.

    Long-term work runs concurrently in the parent. A failed recent request is
    an exception, never an acknowledgement; the existing dual-store retry
    contract remains in charge of the JShi cursor.
    """

    def __init__(self, config):
        self.config = config
        settings = Mem0RecentMemory(config, background=False)
        self.root, self.policy = settings.root, settings.policy
        self._process = self._connection = None
        self._lock = threading.RLock()
        self._closed = False
        self._sequence = 0
        self.runtime_info = None

    def _request(self, method, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RecentWorkerError("Recent memory is closed")
            if self._process is None:
                context = multiprocessing.get_context("spawn")
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(child, self.config.model_dump(mode="python")),
                    name="rems-native-mem0", daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._process, self._connection = process, parent
            if not self._process.is_alive():
                raise RecentWorkerError("Native Mem0 worker exited; restart the memory backend")
            try:
                self._sequence += 1
                request_id = self._sequence
                self._connection.send((request_id, method, args, kwargs))
                while True:
                    while not self._connection.poll(0.2):
                        if not self._process.is_alive():
                            raise RecentWorkerError("Native Mem0 worker exited before acknowledgement")
                    response_id, success, result = self._connection.recv()
                    # A previously interrupted caller may have left an old ack.
                    if response_id == request_id:
                        break
            except (EOFError, BrokenPipeError, OSError) as exc:
                raise RecentWorkerError("Native Mem0 worker connection closed before acknowledgement") from exc
            if not success:
                raise RecentWorkerError(result)
            return result

    def prepare(self):
        self.runtime_info = self._request("prepare")
        return self.runtime_info

    def recall(self, subject_id, query, **kwargs):
        return self._request("recall", subject_id, query, **kwargs)

    def ingest_batch(self, batch):
        return self._request("ingest_batch", batch)

    def maintain(self, **kwargs):
        return self._request("maintain", **kwargs)

    def disk_bytes(self):
        return self._request("disk_bytes")

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self._process is not None and self._process.is_alive():
                    self._request("close")
            finally:
                self._closed = True
                if self._connection is not None:
                    self._connection.close()
                if self._process is not None:
                    self._process.join(timeout=10)
                    if self._process.is_alive():
                        # Only this object's child, after the normal close request.
                        self._process.terminate()
                        self._process.join(timeout=10)
                    if self._process.is_alive():
                        raise RecentWorkerError("Native Mem0 worker did not stop")
                    self._process.close()
