import pytest

from rems.services.recent_process import Mem0RecentProcess, RecentWorkerError


def test_native_worker_rejects_unknown_operation_and_closes(config, tmp_path):
    config.recent_memory.path = str(tmp_path/'recent-child')
    proxy=Mem0RecentProcess(config)
    try:
        with pytest.raises(RecentWorkerError, match="Unsupported"):
            proxy._request("not-an-api")
        assert proxy._process.is_alive()
        with pytest.raises(RecentWorkerError, match="Unsupported"):
            proxy._request("still-not-an-api")
    finally:
        proxy.close()
    proxy.close()
    with pytest.raises(RecentWorkerError, match="closed"):
        proxy.recall("subject","query")


def test_dead_worker_is_not_a_successful_write(config, tmp_path):
    config.recent_memory.path = str(tmp_path/'recent-dead')
    proxy=Mem0RecentProcess(config)
    try:
        with pytest.raises(RecentWorkerError):
            proxy._request("not-an-api")
        # Kill only the test-owned idle child to exercise the failure contract.
        proxy._process.terminate()
        proxy._process.join(timeout=10)
        with pytest.raises(RecentWorkerError, match="exited"):
            proxy.ingest_batch(None)
    finally:
        proxy.close()
