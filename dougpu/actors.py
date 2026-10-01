"""Bounded CPU actor rounds. Only the parent process touches JAX/TPU."""
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback
import numpy as np
from .environment import make_game
from .encoding import encode, oracle_label, terminal_target, pack_requests, unpack_requests, PackedRequests
from .replay import PackedSamples

class Ring:
    def __init__(self, engine, n, seed, opponents=0, historical_fraction=.5):
        self.engine, self.rng = engine, np.random.default_rng(seed)
        self.opponents, self.historical_fraction = opponents, historical_fraction
        self.match_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.games = [make_game(engine, self.rng) for _ in range(n)]
        self.matches = [self._match() for _ in range(n)]
        self.trajectories = [[] for _ in range(n)]
        self.pending = None

    def _match(self):
        if not self.opponents or self.match_rng.random() >= self.historical_fraction:
            return 0, True
        return int(self.match_rng.integers(1, self.opponents+1)), bool(self.match_rng.integers(2))

    def requests(self):
        # Encode once; labels are not transmitted as policy inputs.
        self.pending = [encode(g.public()) for g in self.games]
        if self.opponents:
            for r, (opponent, own_landlord) in zip(self.pending, self.matches):
                r['policy_id'] = 0 if (r['role'] == 0) == own_landlord else opponent
        return self.pending

    def advance(self, choices, version):
        if self.pending is None or len(choices) != len(self.games):
            raise RuntimeError("Actor request/response protocol violation")
        samples, wins = [], []
        for i, (g, idx) in enumerate(zip(self.games, choices)):
            r = self.pending[i]
            idx = int(idx)
            if not r.get('policy_id', 0):
                self.trajectories[i].append((r["tokens"], r["state"], r["actions"][idx], r["role"],
                                            oracle_label(g.oracle(), r["role"]), version))
            g.step(idx)
            if g.done:
                wins.append(g.winner)
                for t, s, a, role, bel, ver in self.trajectories[i]:
                    samples.append((t, s, a, role, bel, terminal_target(g.winner, role), ver))
                self.games[i] = make_game(self.engine, self.rng)
                self.matches[i] = self._match()
                self.trajectories[i] = []
        self.pending = None
        return samples, wins


def _worker(conn, engine, n, seed, opponents=0, historical_fraction=.5):
    os.environ["JAX_PLATFORMS"] = "cpu"  # defensive; this module never imports JAX
    try:
        ring = Ring(engine, n, seed, opponents, historical_fraction)
        conn.send(("ok", pack_requests(ring.requests())))
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                return
            if cmd == 'eval_start':
                from .evaluation import EvaluationGames
                jobs, use_rule = payload
                eval_games = EvaluationGames(engine, jobs, use_rule)
                conn.send(('ok', (eval_games.observe(), [])))
                continue
            if cmd == 'eval_step':
                conn.send(('ok', eval_games.advance(payload)))
                continue
            if cmd == 'eval_close':
                eval_games = None
                continue
            if cmd != 'step':
                raise ValueError('Unknown actor command: '+cmd)
            choices, version = payload
            samples, wins = ring.advance(choices, version)
            conn.send(("ok", (pack_requests(ring.requests()), PackedSamples.pack(samples), wins)))
    except EOFError:
        pass
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()


class ActorPool:
    def __init__(self, cfg, seeds, *, packed=False):
        self.cfg, self.processes, self.conns, self.requests = cfg, [], [], []
        self.packed = packed
        self._pending = set()
        self._deadline = None
        ctx = mp.get_context("spawn")
        try:
            for seed in seeds:
                parent, child = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(child, cfg.engine, cfg.envs_per_worker, int(seed),
                                                     len(cfg.historical_opponents), cfg.historical_fraction),
                                daemon=True)
                p.start()
                child.close()
                self.processes.append(p)
                self.conns.append(parent)
            batches = [self._receive(conn) for conn in self.conns]
            self.requests = (PackedRequests.merge(batches) if self.packed else
                             [r for batch in batches for r in unpack_requests(batch)])
        except BaseException:
            self.close()
            raise

    def _receive(self, conn):
        if not conn.poll(self.cfg.worker_timeout):
            raise TimeoutError("CPU actor timed out; current learner state will be checkpointed")
        status, value = conn.recv()
        if status != "ok":
            raise RuntimeError(value)
        return value

    def submit(self, choices, version, first=0, last=None):
        last = len(self.conns) if last is None else last
        n = self.cfg.envs_per_worker
        indices = range(first, last)
        if not 0 <= first < last <= len(self.conns) or len(choices) != (last-first)*n:
            raise ValueError('Actor submission size or worker range mismatch')
        if self._pending.intersection(indices):
            raise RuntimeError('A worker may advance only once per actor round')
        if not self._pending:
            self._deadline = time.monotonic() + self.cfg.worker_timeout
        for i in indices:
            self.conns[i].send(("step", (choices[(i-first)*n:(i-first+1)*n], version)))
            self._pending.add(i)

    def receive(self):
        if len(self._pending) != len(self.conns):
            raise RuntimeError('Submit every worker before draining an actor round')
        outstanding = {self.conns[i]: i for i in self._pending}
        ordered = [None] * len(self.conns)
        while outstanding:
            ready = wait(outstanding, timeout=max(0., self._deadline-time.monotonic()))
            if not ready:
                raise TimeoutError('CPU actor round timed out; learner state will be checkpointed')
            for conn in ready:
                status, value = conn.recv()
                if status != 'ok':
                    raise RuntimeError(value)
                ordered[outstanding.pop(conn)] = value
        requests, samples, wins = [], [], []
        # Completion order never changes replay insertion or policy RNG order.
        for req, sam, win in ordered:
            if self.packed:
                requests.append(req)
            else:
                requests.extend(unpack_requests(req))
            samples.append(sam)
            wins.extend(win)
        self.requests = PackedRequests.merge(requests) if self.packed else requests
        self._pending.clear()
        self._deadline = None
        return PackedSamples.merge(samples), wins

    def step(self, choices, version):
        self.submit(choices, version)
        return self.receive()

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (OSError, EOFError):
                pass
        for p in self.processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        for conn in self.conns:
            conn.close()


class EvaluationPool:
    """Borrow idle training workers without modifying their in-flight training games."""
    def __init__(self, pool, jobs, workers, use_rule):
        self.pool = pool
        self.use_rule = use_rule
        if pool._pending:
            raise RuntimeError('Evaluation requires a fully drained training round')
        self.conns = pool.conns[:min(workers, len(jobs), len(pool.conns))]
        if not self.conns:
            raise ValueError('Parallel evaluation needs at least one worker')
        for i, conn in enumerate(self.conns):
            chunk = jobs[len(jobs)*i//len(self.conns):len(jobs)*(i+1)//len(self.conns)]
            conn.send(('eval_start', (chunk, use_rule)))
        self._collect()

    def _collect(self):
        self.ids, self.current, self.requests, self.rules, self.counts = [], [], [], [], []
        outcomes = []
        for conn in self.conns:
            (ids, current, packed, rules), wins = self.pool._receive(conn)
            self.ids.extend(ids)
            self.current.extend(current)
            requests = unpack_requests(packed)
            expected = sum(current) if self.use_rule else len(ids)
            if len(requests) != expected:
                raise RuntimeError('Evaluation encoded request count mismatch')
            # Rule turns retain their game positions but carry no policy input.
            encoded = iter(requests)
            self.requests.extend(next(encoded) if not self.use_rule or own else None for own in current)
            self.rules.extend(rules)
            self.counts.append(len(ids))
            outcomes.extend(wins)
        return outcomes

    def step(self, choices):
        if len(choices) != len(self.ids):
            raise ValueError('Evaluation request/response size mismatch')
        offset = 0
        for conn, count in zip(self.conns, self.counts):
            conn.send(('eval_step', choices[offset:offset+count]))
            offset += count
        return self._collect()

    def close(self):
        for conn in self.conns:
            try:
                conn.send(('eval_close', None))
            except (OSError, EOFError):
                pass
