"""
Loop-closure back-end: detect revisited places, add loop constraints, optimize
the pose graph and rebuild the map. Runs in a background thread so the
front-end keeps processing scans in real time.

For every few new keyframes k:

 1. CANDIDATES ("have I been here before?")
    Old keyframes j whose estimated (x, y) is within `search_radius` of
    keyframe k, AND that are at least `min_path_m` back along the travelled
    path. The path condition ignores the keyframes the robot has just passed,
    which are always nearby but are not a loop.

 2. VERIFICATION (scan matching)
    Proximity alone is not proof: the estimate has drifted, and identical
    rack aisles look alike. Keyframe k's scan is matched against a submap
    built from keyframe j and its path neighbours, searching a window of
    +/- `window` metres and +/- `window_angle` around the current estimate
    (correlative branch-and-bound matcher). The match is accepted only with a
    high score and a high inlier ratio. It returns the measured relative pose
    z_jk and its covariance, which become a "loop" edge in the pose graph.

 3. OPTIMIZATION
    If the new edge disagrees with the current estimate (the map is really
    off), the whole graph is optimized: the accumulated error is spread over
    every keyframe of the loop, weighted by how certain each edge is.
    Loop edges have a robust kernel, and impossible ones are rejected.

 4. MAP REBUILD
    A brand-new occupancy grid is ray-traced from every keyframe at its
    optimized pose. The result is handed to the front-end, which swaps it in
    and carries along the keyframes it added while the back-end was working.
"""

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
import traceback

import numpy as np

from .keyframes import insert_keyframe
from .occupancy_grid import OccupancyGrid
from .pose_graph import information_from_covariance, make_edge, optimize, OptimizationStats
from .scan_matcher import CorrelativeScanMatcher
from .se2 import relative, relative_many, transform_points


@dataclass
class LoopClosureConfig:
    enabled: bool = True
    threaded: bool = True
    min_path_m: float = 10.0          # j must be this far back along the path
    search_radius: float = 3.0        # (x, y) distance for candidates [m]
    max_candidates: int = 2           # matched per check (from different passes)
    window: float = 1.5               # translation search half-width [m]
    window_angle: float = math.radians(15.0)
    resolution: float = 0.05
    sigma: float = 0.05
    submap_path_m: float = 2.5        # submap = keyframes within +/- this path
    submap_max_keyframes: int = 25
    min_score: float = 0.50
    min_inliers: float = 0.60
    check_every: int = 2              # check every N keyframes
    max_backlog: int = 40             # unchecked keyframes kept while busy
    sigma_xy_min: float = 0.03        # floor on loop-edge uncertainty
    sigma_theta_min: float = math.radians(0.5)
    huber_delta: float = 2.0
    outlier_chi2: float = 16.27       # 99.9% chi-square, 3 DOF
    trigger_xy: float = 0.05          # optimize if a closure disagrees by more
    trigger_theta: float = math.radians(1.0)
    max_pending_closures: int = 10    # or after this many small closures
    rebuild_min_correction: float = 0.02   # skip the map rebuild below this [m]


@dataclass
class LoopClosure:
    old_id: int
    new_id: int
    score: float
    inliers: float
    innovation_xy: float
    innovation_theta: float


@dataclass
class BackendResult:
    generation: int
    last_id: int
    poses: np.ndarray                 # optimized poses of keyframes 0..last_id
    grid: OccupancyGrid               # map rebuilt from those poses
    local_map: object                 # front-end local map rebuilt from those poses
    stats: OptimizationStats
    closures: list
    rebuild_time_s: float


class LoopClosureBackend:

    def __init__(self, config, keyframes, graph, graph_lock, grid_kwargs,
                 local_map_factory=None):
        self.cfg = config
        self._kfs = keyframes            # shared list, append-only (front-end)
        self._graph = graph              # shared PoseGraph
        self._lock = graph_lock          # protects graph and keyframe list
        self._grid_kwargs = dict(grid_kwargs)
        self._grid_kwargs["clamp"] = True
        self._local_map_factory = local_map_factory
        self.matcher = CorrelativeScanMatcher(config.resolution, config.sigma)

        self._generation = 0
        self._latest_kf = -1
        self._next_check = 0
        self._force = False
        self._pending = 0
        self._result = None
        self._result_lock = threading.Lock()
        self.events = deque(maxlen=200)  # human-readable log lines for the node
        self.busy = False

        self._stop = False
        self._wake = threading.Event()
        self._thread = None
        if config.threaded:
            self._thread = threading.Thread(target=self._run, name="slam-backend",
                                            daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------
    # Front-end API
    # ------------------------------------------------------------------
    def notify_keyframe(self, kf_id):
        self._latest_kf = kf_id
        self._kick()

    def request_optimization(self):
        self._force = True
        self._kick()

    def poll(self):
        """Return a finished result for the front-end to apply (or None)."""
        with self._result_lock:
            res, self._result = self._result, None
        if res is not None:
            self._wake.set()             # the back-end may continue
            if res.generation != self._generation:
                return None
        return res

    def reset(self):
        with self._result_lock:
            self._generation += 1
            self._result = None
        self._latest_kf = -1
        self._next_check = 0
        self._pending = 0
        self._force = False

    def shutdown(self):
        self._stop = True
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def drain_events(self):
        out = []
        while self.events:
            out.append(self.events.popleft())
        return out

    # ------------------------------------------------------------------
    def _kick(self):
        if self._thread is None:
            self._cycle()                # synchronous mode (tests)
        else:
            self._wake.set()

    def _run(self):
        while not self._stop:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            if self._stop:
                break
            try:
                self._cycle()
            except Exception:            # never let the thread die silently
                self.events.append("back-end error:\n" + traceback.format_exc())
                self.busy = False

    # ------------------------------------------------------------------
    def _cycle(self):
        cfg = self.cfg
        with self._result_lock:
            if self._result is not None:
                return                   # wait until the front-end applied it
        k_latest = self._latest_kf
        # Every check_every-th keyframe not checked yet. While the thread was
        # busy (e.g. rebuilding the map) keyframes pile up; they are checked
        # now, so a loop is never missed because the back-end was busy.
        todo = list(range(self._next_check, k_latest + 1, cfg.check_every)) \
            if cfg.enabled and k_latest >= 0 else []
        if len(todo) > cfg.max_backlog:
            todo = todo[-cfg.max_backlog:]
        if not todo and not self._force:
            return

        self.busy = True
        gen = self._generation
        with self._lock:
            n = self._graph.n
            poses = self._graph.poses.copy()
            edges = list(self._graph.edges)
            kfs = self._kfs[:n]
        if n < 2:
            self.busy = False
            return

        run_opt = self._force
        self._force = False
        closures = []
        todo = [k for k in todo if k < n]
        if todo:
            self._next_check = todo[-1] + cfg.check_every
        for k in todo:
            for j, z, info, m, innov in self._detect(k, poses, kfs):
                edge = make_edge(j, k, z, info, "loop")
                with self._lock:
                    if gen != self._generation:
                        self.busy = False
                        return
                    self._graph.add_edge(edge)
                edges.append(edge)
                lc = LoopClosure(j, k, m.score, m.inlier_ratio,
                                 math.hypot(innov[0], innov[1]), abs(innov[2]))
                closures.append(lc)
                self.events.append(
                    f"loop closure: keyframe {k} <-> {j} | score {m.score:.2f} "
                    f"inliers {m.inlier_ratio:.0%} | disagreement "
                    f"{lc.innovation_xy * 100:.1f} cm, {math.degrees(lc.innovation_theta):.1f} deg")
                if lc.innovation_xy > cfg.trigger_xy or lc.innovation_theta > cfg.trigger_theta:
                    run_opt = True
                else:
                    self._pending += 1
        if self._pending >= cfg.max_pending_closures:
            run_opt = True
        if not run_opt:
            self.busy = False
            return
        self._pending = 0

        # ---- Optimize the snapshot ----
        X, stats = optimize(poses, edges, huber_delta=cfg.huber_delta,
                            outlier_chi2=cfg.outlier_chi2)
        if stats.rejected_edges:
            with self._lock:
                if gen == self._generation:
                    self._graph.remove_edges(stats.rejected_edges)
            self.events.append(f"rejected {len(stats.rejected_edges)} inconsistent loop "
                               f"closure(s) as outliers")

        if stats.max_correction < cfg.rebuild_min_correction and not stats.rejected_edges:
            self.events.append(f"pose graph optimized: map already consistent "
                               f"(max correction {stats.max_correction * 100:.1f} cm), "
                               f"no rebuild needed")
            self.busy = False
            return

        # ---- Rebuild the global map from every keyframe ----
        t0 = time.perf_counter()
        grid = OccupancyGrid(**self._grid_kwargs)
        for i in range(n):
            insert_keyframe(grid, kfs[i], X[i])
        local = None
        if self._local_map_factory is not None:
            local = self._local_map_factory()
            path_K = kfs[n - 1].path_dist
            first = n - 1
            while (first > 0 and n - first < local.max_keyframes
                   and path_K - kfs[first - 1].path_dist <= local.window_m):
                first -= 1
            for i in range(first, n):
                local.add(kfs[i], X[i])
        rebuild = time.perf_counter() - t0
        self.events.append(
            f"pose graph optimized: {n} keyframes, {len(edges) - len(stats.rejected_edges)} "
            f"edges, max correction {stats.max_correction * 100:.1f} cm, "
            f"{stats.iterations} iterations in {stats.time_s * 1000:.0f} ms; "
            f"map rebuilt in {rebuild:.2f} s")

        result = BackendResult(gen, n - 1, X, grid, local, stats, closures, rebuild)
        with self._result_lock:
            if gen == self._generation:
                self._result = result
        self.busy = False

    # ------------------------------------------------------------------
    def _detect(self, k, poses, kfs):
        cfg = self.cfg
        path = np.array([kf.path_dist for kf in kfs])
        pk = poses[k]
        eligible = np.flatnonzero(path[k] - path >= cfg.min_path_m)
        if eligible.size == 0:
            return []
        dist = np.hypot(poses[eligible, 0] - pk[0], poses[eligible, 1] - pk[1])
        near = eligible[dist <= cfg.search_radius]
        near = near[np.argsort(dist[dist <= cfg.search_radius])]

        chosen = []
        for j in near:                   # closest first, one per separate pass
            if all(abs(path[j] - path[c]) > 2.0 * cfg.submap_path_m for c in chosen):
                chosen.append(int(j))
            if len(chosen) >= cfg.max_candidates:
                break

        out = []
        query = kfs[k].loop_points.astype(np.float64)
        for j in chosen:
            ref = self._submap(j, poses, kfs, path)
            guess = relative(poses[j], pk)
            m = self.matcher.match(ref, query, guess, cfg.window, cfg.window_angle)
            if m is None or m.score < cfg.min_score or m.inlier_ratio < cfg.min_inliers:
                continue
            info = information_from_covariance(
                m.cov, (cfg.sigma_xy_min, cfg.sigma_xy_min, cfg.sigma_theta_min))
            out.append((j, m.pose, info, m, relative(guess, m.pose)))
        return out

    def _submap(self, j, poses, kfs, path):
        """Loop points of keyframe j and its path neighbours, in j's frame."""
        cfg = self.cfg
        sel = np.flatnonzero(np.abs(path - path[j]) <= cfg.submap_path_m)
        if sel.size > cfg.submap_max_keyframes:
            sel = sel[np.linspace(0, sel.size - 1, cfg.submap_max_keyframes).astype(int)]
        rel = relative_many(poses[j], poses[sel])
        clouds = [transform_points(rel[t], kfs[i].loop_points.astype(np.float64))
                  for t, i in enumerate(sel)]
        return np.vstack(clouds)
