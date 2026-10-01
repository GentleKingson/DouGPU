"""Experimental bounded ready-first scheduling; no additional accelerator owner."""
from collections import deque
from multiprocessing.connection import wait
import time
from .actors import ActorPool
from .encoding import unpack_requests
from .replay import PackedSamples


class ReadyActorPool(ActorPool):
    def __init__(self, cfg, seeds):
        super().__init__(cfg, seeds)
        self.ready = deque(range(len(self.conns)))
        self.deadlines = {}
        self.steps = [0] * len(self.conns)
        self.counts = {'submitted': 0, 'received': 0, 'peak_pending': 0, 'max_step_lead': 0}

    def collect(self, block=False):
        samples, wins = [], []
        if not self._pending:
            return PackedSamples({}), wins
        by_conn = {self.conns[i]: i for i in self._pending}
        timeout = max(0., min(self.deadlines.values())-time.monotonic()) if block else 0.
        readable = wait(by_conn, timeout=timeout)
        if not readable and (block or min(self.deadlines.values()) <= time.monotonic()):
            raise TimeoutError('Ready-first actor timed out')
        for conn in readable:
            i = by_conn[conn]
            status, value = conn.recv()
            if status != 'ok':
                raise RuntimeError(value)
            packed, batch, winners = value
            n = self.cfg.envs_per_worker
            self.requests[i*n:(i+1)*n] = unpack_requests(packed)
            self._pending.remove(i)
            del self.deadlines[i]
            self.ready.append(i)
            samples.append(batch)
            wins.extend(winners)
            self.counts['received'] += 1
        return PackedSamples.merge(samples), wins

    def select(self):
        # FIFO among ready workers, and at most two steps ahead of the slowest.
        floor = min(self.steps)
        limit = max(1, self.cfg.infer_batch//self.cfg.envs_per_worker)
        selected = []
        for _ in range(len(self.ready)):
            i = self.ready.popleft()
            if len(selected) < limit and self.steps[i] < floor+2:
                selected.append(i)
            else:
                self.ready.append(i)
        return selected

    def submit_selected(self, indices, choices, version):
        n = self.cfg.envs_per_worker
        if len(choices) != len(indices)*n or len(set(indices)) != len(indices):
            raise ValueError('Invalid ready-first submission')
        for off, i in enumerate(indices):
            self.submit(choices[off*n:(off+1)*n], version, i, i+1)
            self.deadlines[i] = time.monotonic()+self.cfg.worker_timeout
            self.steps[i] += 1
            self.counts['submitted'] += 1
        self.counts['peak_pending'] = max(self.counts['peak_pending'], len(self._pending))
        self.counts['max_step_lead'] = max(self.counts['max_step_lead'], max(self.steps)-min(self.steps))

    def drain(self):
        samples, wins = [], []
        while self._pending:
            batch, winners = self.collect(block=True)
            samples.append(batch)
            wins.extend(winners)
        return PackedSamples.merge(samples), wins

    def close(self):
        if self._pending:
            # Failed runs restore learner state, not actor games. A worker might
            # be blocked sending a large episode; do not block sending 'close'.
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            for process in self.processes:
                process.join(timeout=2)
            for conn in self.conns:
                conn.close()
        else:
            super().close()
