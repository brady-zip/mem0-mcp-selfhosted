"""Concurrency tests for call_with_graph() threading lock.

Verifies that the threading.Lock in helpers.py prevents state corruption
when concurrent calls use different enable_graph values.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock

import pytest

from mem0_mcp_selfhosted.helpers import call_with_graph


class _CapturingMemory:
    """Mock Memory that captures enable_graph value at execution time."""

    def __init__(self):
        self.graph = MagicMock()  # graph is not None
        self.enable_graph = False
        self.captured_values: list[bool] = []
        self._lock = threading.Lock()

    def capture_state(self) -> bool:
        """Capture the current enable_graph value (called inside locked region)."""
        val = self.enable_graph
        with self._lock:
            self.captured_values.append(val)
        return val


class TestConcurrentAlternatingEnableGraph:
    def test_each_thread_observes_correct_value(self):
        """Fire 10 threads with alternating enable_graph, verify each sees correct value."""
        mem = _CapturingMemory()
        results: dict[int, bool] = {}

        def worker(idx: int) -> tuple[int, bool]:
            enable = idx % 2 == 0  # True for even, False for odd
            val = call_with_graph(mem, enable, False, mem.capture_state)
            return idx, val

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(worker, i) for i in range(10)]
            for f in as_completed(futures):
                idx, val = f.result()
                results[idx] = val

        # Each thread should have seen its own requested value
        for idx, observed in results.items():
            expected = idx % 2 == 0
            assert observed == expected, (
                f"Thread {idx}: expected enable_graph={expected}, got {observed}"
            )


class TestLockSerializesExecution:
    def test_no_concurrent_execution(self):
        """Verify the lock prevents overlapping execution of the inner function."""
        concurrent_count = 0
        max_concurrent = 0
        lock = threading.Lock()

        class SlowMemory:
            def __init__(self):
                self.graph = MagicMock()
                self.enable_graph = False

        mem = SlowMemory()

        def slow_fn():
            nonlocal concurrent_count, max_concurrent
            with lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent:
                    max_concurrent = concurrent_count
            # Simulate work inside the locked region
            import time
            time.sleep(0.01)
            with lock:
                concurrent_count -= 1
            return True

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(call_with_graph, mem, True, False, slow_fn)
                for _ in range(5)
            ]
            for f in as_completed(futures):
                f.result()

        assert max_concurrent == 1, (
            f"Lock failed: max concurrent executions was {max_concurrent}, expected 1"
        )


class TestEnableGraphWithNoGraphStore:
    def test_enable_graph_true_but_graph_is_none(self):
        """enable_graph=True should result in False when memory.graph is None."""
        mem = MagicMock()
        mem.graph = None
        mem.enable_graph = False

        captured = []

        def capture():
            captured.append(mem.enable_graph)
            return True

        call_with_graph(mem, True, False, capture)
        assert captured[0] is False, (
            "enable_graph should be False when memory.graph is None"
        )


class TestNoGraphFastPath:
    """Verify the lock-free fast path when memory.graph is None."""

    def test_no_graph_skips_lock_allows_concurrent_execution(self):
        """With no graph store, concurrent calls run in parallel (not serialized)."""
        import time

        class NoGraphMemory:
            def __init__(self):
                self.graph = None
                self.enable_graph = False

        mem = NoGraphMemory()
        results = []

        def slow_fn():
            time.sleep(0.05)
            return True

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(call_with_graph, mem, None, False, slow_fn) for _ in range(5)]
            for f in as_completed(futures):
                results.append(f.result())
        elapsed = time.monotonic() - start

        assert all(results)
        # Concurrent: should finish well under 5 * 0.05 = 0.25s
        assert elapsed < 0.15, (
            f"Fast path appears to be serialized: took {elapsed:.3f}s, expected < 0.15s"
        )

    def test_no_graph_concurrent_all_observe_false(self):
        """All threads observe enable_graph=False in the fast path."""

        class NoGraphMemory:
            def __init__(self):
                self.graph = None
                self.enable_graph = False

        mem = NoGraphMemory()
        observed = []
        lock = threading.Lock()

        def capture():
            with lock:
                observed.append(mem.enable_graph)
            return True

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(call_with_graph, mem, None, False, capture) for _ in range(20)]
            for f in as_completed(futures):
                f.result()

        assert len(observed) == 20
        assert all(v is False for v in observed), (
            f"Expected all False, got: {observed}"
        )

    def test_graph_lock_timeout_raises(self, monkeypatch):
        """RuntimeError raised when graph lock cannot be acquired within timeout."""
        import mem0_mcp_selfhosted.helpers as helpers_mod

        monkeypatch.setenv("MEM0_LOCK_TIMEOUT_SECS", "0.05")
        # Reset cached timeout so env var is re-read
        helpers_mod._GRAPH_LOCK_TIMEOUT_SECS = None

        class GraphMemory:
            def __init__(self):
                self.graph = MagicMock()  # non-None → slow path
                self.enable_graph = False

        mem = GraphMemory()
        barrier = threading.Event()
        release_event = threading.Event()

        def hold_lock():
            """Acquire the lock and hold it until signalled."""
            helpers_mod._graph_lock.acquire()
            barrier.set()
            release_event.wait(timeout=2.0)
            helpers_mod._graph_lock.release()

        holder = threading.Thread(target=hold_lock, daemon=True)
        holder.start()
        barrier.wait()  # ensure lock is held before we proceed

        try:
            with pytest.raises(RuntimeError, match="Graph lock timeout"):
                call_with_graph(mem, True, True, lambda: True)
        finally:
            release_event.set()  # signal holder to release the lock
            holder.join(timeout=1.0)
            helpers_mod._GRAPH_LOCK_TIMEOUT_SECS = None
