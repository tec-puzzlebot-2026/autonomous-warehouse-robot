#!/usr/bin/env python3
"""
Offline test of the SLAM core — no ROS, no Isaac Sim.

A 2D warehouse (walls, rack rows, pillars) is ray-cast to simulate a LiDAR
mounted with an offset on a differential-drive robot. Wheel odometry is
corrupted with systematic scale errors and noise, like a real robot with
slightly wrong wheel radii. The robot drives a long tour and comes back to
places it has already mapped, which is where loop closure matters.

Several SLAM configurations are run on exactly the same data and compared
against ground truth:

  odometry   wheel odometry only (no SLAM)
  v1         front-end matched against the global map, no loop closure
  local      front-end matched against the local map, no loop closure
  v2         local front-end + pose graph + loop closure (synchronous)
  v2-thread  same as v2, back-end running in a background thread

Metrics
  ATE        absolute trajectory error: RMS position error vs ground truth
  ghost      fraction of occupied map cells more than 15 cm from any real
             wall (doubled, bent or globally scaled walls all count)
  doubled    the same after removing the global per-axis scale error, which
             loop closure cannot observe: what is left is local inconsistency
             (walls drawn twice), the error loop closure is meant to remove
  scale      per-axis scale error of the trajectory [%]

Run from the mapping/ directory:
    python3 tests/offline_sim_test.py                       # large world, all modes
    python3 tests/offline_sim_test.py --world small
    python3 tests/offline_sim_test.py --modes v1 v2 --beams 3600
    python3 tests/offline_sim_test.py --modes v2-thread --speed 2   # 2x real time
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from slam_core import (SlamSystem, SlamConfig, ScanPreprocessor, OdometryBuffer,  # noqa: E402
                       LoopClosureConfig, bounded_distance_transform,
                       compose, relative, wrap_angle)
from slam_core.se2 import relative_many  # noqa: E402


# ----------------------------------------------------------------------
# Worlds
# ----------------------------------------------------------------------
def rect(x0, y0, x1, y1):
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def build_world():
    """Small 20 x 12 m warehouse (the v1 test)."""
    seg = rect(0, 0, 20, 12)
    for x0 in (4.0, 9.0, 14.0):
        seg += rect(x0, 3.5, x0 + 3.0, 4.5)
        seg += rect(x0, 7.5, x0 + 3.0, 8.5)
    seg += rect(1.0, 10.0, 2.0, 11.0)
    seg += rect(18.0, 1.0, 19.0, 2.0)
    seg += [(10.0, 0.0, 10.0, 1.2)]
    return np.array(seg, dtype=np.float64)


def build_large_world():
    """40 x 26 m warehouse: four long rack rows with a cross aisle, pillars."""
    seg = rect(0, 0, 40, 26)
    for y0 in (5.0, 10.0, 15.0, 20.0):
        seg += rect(6.0, y0, 18.5, y0 + 1.0)            # rack row, west half
        seg += rect(21.5, y0, 34.0, y0 + 1.0)           # rack row, east half
    for x in (8.0, 16.0, 24.0, 32.0):                   # pillars on the long walls
        seg += rect(x, 0.0, x + 0.4, 0.4)
        seg += rect(x, 25.6, x + 0.4, 26.0)
    for y in (8.0, 18.0):                               # pillars on the short walls
        seg += rect(0.0, y, 0.4, y + 0.4)
        seg += rect(39.6, y, 40.0, y + 0.4)
    seg += rect(1.0, 24.3, 2.2, 25.3)                   # boxes, off the robot's route
    seg += rect(38.4, 5.0, 39.4, 6.0)
    seg += rect(38.5, 16.0, 39.3, 17.2)
    return np.array(seg, dtype=np.float64)


SCENARIOS = {
    "small": dict(world=build_world, start=(2.0, 2.0, 0.0), max_range=12.0,
                  waypoints=[(18.5, 2.2), (18.5, 6.0), (2.5, 6.0), (2.5, 10.2), (18.0, 10.2),
                             (18.0, 6.0), (2.0, 6.0), (2.0, 2.0), (8.0, 2.0)]),
    "large": dict(world=build_large_world, start=(3.0, 2.5, 0.0), max_range=8.0,
                  waypoints=[(37.0, 2.5), (37.0, 13.0), (3.0, 13.0), (3.0, 23.5), (37.0, 23.5),
                             (37.0, 8.0), (3.0, 8.0), (3.0, 2.5), (20.0, 2.5), (20.0, 13.0)]),
}


def raycast(world, origin, angles, max_range):
    """Distance along each ray to the first segment (inf if none)."""
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x1, y1, x2, y2 = (world[:, i][None, :] for i in range(4))
    ex, ey = x2 - x1, y2 - y1
    denom = dx * ey - dy * ex
    ox, oy = x1 - origin[0], y1 - origin[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ox * ey - oy * ex) / denom
        u = (ox * dy - oy * dx) / denom
    ok = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf).min(axis=1)
    t[t > max_range] = np.inf
    return t


# ----------------------------------------------------------------------
# Robot motion
# ----------------------------------------------------------------------
def drive(waypoints, start, dt=0.02, v_max=0.4):
    pose = np.array(start, dtype=np.float64)
    traj = [pose.copy()]
    for wx, wy in waypoints:
        for _ in range(100000):
            ex, ey = wx - pose[0], wy - pose[1]
            if math.hypot(ex, ey) < 0.1:
                break
            err = float(wrap_angle(math.atan2(ey, ex) - pose[2]))
            w = float(np.clip(2.5 * err, -1.2, 1.2))
            v = v_max * max(0.0, math.cos(err)) ** 2
            pose = pose + np.array([v * math.cos(pose[2] + w * dt / 2) * dt,
                                    v * math.sin(pose[2] + w * dt / 2) * dt, w * dt])
            pose[2] = float(wrap_angle(pose[2]))
            traj.append(pose.copy())
    return np.array(traj)


def corrupt_odometry(true_traj, rng, scale_err=0.03, yaw_err=0.04, noise=0.002):
    """Integrate true increments with systematic scale/yaw errors and noise."""
    odom = [np.zeros(3)]
    for a, b in zip(true_traj[:-1], true_traj[1:]):
        d = relative(a, b)
        ds = math.hypot(d[0], d[1]) * math.copysign(1.0, d[0] if abs(d[0]) > 1e-9 else 1.0)
        dth = d[2]
        ds_o = ds * (1.0 + scale_err) + rng.normal(0, noise * abs(ds) + 1e-5)
        dth_o = dth * (1.0 + yaw_err) + 0.01 * ds + rng.normal(0, noise * abs(dth) + 1e-5)
        step = np.array([ds_o * math.cos(dth_o / 2), ds_o * math.sin(dth_o / 2), dth_o])
        odom.append(compose(odom[-1], step))
    return np.array(odom)


# ----------------------------------------------------------------------
# Data generation (shared by every mode, so they see identical inputs)
# ----------------------------------------------------------------------
def generate(scenario, n_beams, seed, dt=0.02):
    sc = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    world = sc["world"]()
    true_traj = drive(sc["waypoints"], sc["start"], dt)
    odom_traj = corrupt_odometry(true_traj, rng)
    laser_mount = np.array([0.08, 0.02, math.radians(3.0)])
    angle_min, angle_inc = -math.pi, 2 * math.pi / n_beams
    angles = angle_min + angle_inc * np.arange(n_beams)
    scan_every = int(round(0.1 / dt))     # 10 Hz LiDAR
    scans = []
    for k in range(0, len(true_traj), scan_every):
        sensor = compose(true_traj[k], laser_mount)
        r = raycast(world, sensor[:2], sensor[2] + angles, sc["max_range"])
        r = r + rng.normal(0, 0.01, r.shape)
        r[rng.random(r.shape) < 0.01] = np.nan
        scans.append((k, r))
    return dict(world=world, true=true_traj, odom=odom_traj, mount=laser_mount,
                angle_min=angle_min, angle_inc=angle_inc, scans=scans, dt=dt,
                max_range=sc["max_range"], n_beams=n_beams)


# ----------------------------------------------------------------------
LOCAL_WINDOW = SlamConfig().local_map_window_m


def make_config(mode, particles, seed):
    loop = LoopClosureConfig(enabled=mode in ("v2", "v2-thread"),
                             threaded=(mode == "v2-thread"))
    return SlamConfig(num_particles=particles, seed=seed, local_map_window_m=LOCAL_WINDOW,
                      frontend_local_map=(mode != "v1"), loop=loop)


def run_mode(data, mode, particles=200, seed=1, speed=1.0):
    true_traj, odom_traj, dt = data["true"], data["odom"], data["dt"]
    gt_rel = relative_many(true_traj[0], true_traj)
    if mode == "odometry":
        idx = [k for k, _ in data["scans"]]
        return dict(mode=mode, est=odom_traj[idx], gt=gt_rel[idx], grid=None, times=None,
                    slam=None, events=[])

    pre = ScanPreprocessor(map_beam_step=max(1, data["n_beams"] // 900), match_beams=180,
                           map_max_range=min(10.0, data["max_range"]),
                           match_max_range=min(10.0, data["max_range"]))
    m = data["mount"]
    c, s = math.cos(m[2]), math.sin(m[2])
    pre.set_mounting([[c, -s], [s, c]], m[:2])

    slam = SlamSystem(make_config(mode, particles, seed))
    buf = OdometryBuffer()
    est_live, gt, times, events = [], [], [], []
    k_last = 0
    kf_scan_index = []
    # The threaded back-end is paced like the real robot: one 10 Hz scan every
    # 0.1 s of wall time (divided by `speed`), so it gets the same time budget.
    pace = 0.1 / speed if (mode == "v2-thread" and speed > 0) else 0.0
    wall0 = time.perf_counter()
    for n, (k, r) in enumerate(data["scans"]):
        if pace:
            wait = wall0 + n * pace - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        for kk in range(k_last, k + 1):
            buf.add(kk * dt, odom_traj[kk])
        k_last = k + 1
        t = k * dt - 0.01
        odom_pose = buf.pose_at(t)
        if odom_pose is None:
            continue
        t0 = time.perf_counter()
        prepared = pre.prepare(r, data["angle_min"], data["angle_inc"], 0.12,
                               data["max_range"])
        res = slam.process(t, odom_pose, prepared)
        times.append(time.perf_counter() - t0)
        if res.keyframe:
            kf_scan_index.append(k)
        est_live.append(res.pose)
        gt.append(gt_rel[k])
        events += slam.backend.drain_events()

    # Let a background optimization still running finish, then apply it.
    if mode == "v2-thread":
        for _ in range(600):
            if not slam.backend.busy and slam.backend._result is None:
                break
            time.sleep(0.05)
        slam._apply_backend_result()
        events += slam.backend.drain_events()

    # Final trajectory = keyframe poses after all corrections (what the map uses).
    kf_poses = slam.keyframe_poses()
    out = dict(mode=mode, est_live=np.array(est_live), gt_live=np.array(gt),
               est=kf_poses, gt=gt_rel[np.array(kf_scan_index)], grid=slam.grid,
               times=np.array(times) * 1000.0, slam=slam, events=events)
    slam.shutdown()
    return out


# ----------------------------------------------------------------------
def ghost_ratio(grid, world, start, scale=(1.0, 1.0), res=0.05, tol=0.15):
    """
    Fraction of occupied cells farther than tol from any true wall.
    With scale=(sx, sy) the map is first un-scaled about the start, which
    removes a global scale error and leaves only local inconsistencies
    (doubled or bent walls).
    """
    if grid is None:
        return float("nan")
    values, origin = grid.to_occupancy_values()
    occ_r, occ_c = np.nonzero(values >= 65)
    if occ_r.size == 0:
        return float("nan")
    cells = np.column_stack((origin[0] + (occ_c + 0.5) * grid.res,
                             origin[1] + (occ_r + 0.5) * grid.res)) / np.asarray(scale)
    # True walls, sampled densely and expressed in the map (start) frame.
    pts = []
    for x1, y1, x2, y2 in world:
        n = int(math.hypot(x2 - x1, y2 - y1) / 0.01) + 2
        tt = np.linspace(0, 1, n)
        pts.append(np.column_stack((x1 + tt * (x2 - x1), y1 + tt * (y2 - y1))))
    pts = np.vstack(pts)
    inv = relative(np.array(start, float), np.zeros(3))
    c, s = math.cos(inv[2]), math.sin(inv[2])
    pm = np.column_stack((inv[0] + c * pts[:, 0] - s * pts[:, 1],
                          inv[1] + s * pts[:, 0] + c * pts[:, 1]))
    lo = np.minimum(pm.min(axis=0), cells.min(axis=0)) - 1.0
    hi = np.maximum(pm.max(axis=0), cells.max(axis=0)) + 1.0
    w, h = (np.ceil((hi - lo) / res).astype(int) + 1)
    truth = np.zeros((h, w), dtype=bool)
    ix = np.floor((pm[:, 0] - lo[0]) / res).astype(int)
    iy = np.floor((pm[:, 1] - lo[1]) / res).astype(int)
    truth[iy, ix] = True
    R = int(math.ceil(tol / res)) + 1
    dist = bounded_distance_transform(truth, R) * res
    cx = np.floor((cells[:, 0] - lo[0]) / res).astype(int)
    cy = np.floor((cells[:, 1] - lo[1]) / res).astype(int)
    return float(np.mean(dist[cy, cx] > tol))


def axis_scale(est, gt):
    """Least-squares per-axis scale of the trajectory about the start."""
    sx = float(np.dot(est[:, 0], gt[:, 0]) / max(np.dot(gt[:, 0], gt[:, 0]), 1e-9))
    sy = float(np.dot(est[:, 1], gt[:, 1]) / max(np.dot(gt[:, 1], gt[:, 1]), 1e-9))
    return sx, sy


def ate(est, gt):
    return math.sqrt(np.mean(np.sum((est[:, :2] - gt[:, :2]) ** 2, axis=1)))


# ----------------------------------------------------------------------
def main():
    global LOCAL_WINDOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=sorted(SCENARIOS), default="large")
    ap.add_argument("--modes", nargs="+",
                    default=["odometry", "v1", "local", "v2", "v2-thread"])
    ap.add_argument("--beams", type=int, default=720)
    ap.add_argument("--particles", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="v2-thread replay speed vs real time (0 = as fast as possible)")
    ap.add_argument("--local-window", type=float, default=LOCAL_WINDOW,
                    help="front-end local map window [m of path]")
    ap.add_argument("--out", default="offline_slam_result.png")
    args = ap.parse_args()
    LOCAL_WINDOW = args.local_window

    sc = SCENARIOS[args.world]
    print(f"generating '{args.world}' data ({args.beams} beams)...", flush=True)
    data = generate(args.world, args.beams, args.seed)
    path_len = np.sum(np.hypot(*np.diff(data["true"][:, :2], axis=0).T))
    print(f"path {path_len:.0f} m, {len(data['scans'])} scans, LiDAR range "
          f"{data['max_range']:.0f} m\n")

    results = []
    print(f"{'mode':10s} {'ATE [m]':>8s} {'max err':>8s} {'final err':>9s} "
          f"{'ghost':>6s} {'doubled':>8s} {'scale x/y [%]':>14s} "
          f"{'ms/scan':>8s} {'p95 ms':>7s} {'loops':>6s} {'opts':>5s}")
    for mode in args.modes:
        r = run_mode(data, mode, args.particles, args.seed, args.speed)
        err = np.hypot(*(r["est"][:, :2] - r["gt"][:, :2]).T)
        r["ate"] = ate(r["est"], r["gt"])
        r["ghost"] = ghost_ratio(r["grid"], data["world"], sc["start"])
        r["scale"] = axis_scale(r["est"], r["gt"])
        r["doubled"] = ghost_ratio(r["grid"], data["world"], sc["start"], r["scale"])
        loops = sum(1 for e in r["events"] if e.startswith("loop closure"))
        opts = r["slam"].corrections if r["slam"] else 0
        tm = (f"{r['times'].mean():8.1f} {np.percentile(r['times'], 95):7.1f}"
              if r["times"] is not None else f"{'-':>8s} {'-':>7s}")
        sx, sy = r["scale"]
        mapcols = (f"{r['ghost'] * 100:5.1f}% {r['doubled'] * 100:7.1f}% "
                   f"{(sx - 1) * 100:+6.2f}/{(sy - 1) * 100:+5.2f} ") if r["grid"] is not None \
            else f"{'-':>6s} {'-':>8s} {'-':>14s} "
        print(f"{mode:10s} {r['ate']:8.3f} {err.max():8.3f} {err[-1]:9.3f} "
              f"{mapcols}{tm} {loops:6d} {opts:5d}", flush=True)
        results.append(r)

    for r in results:
        for e in r["events"]:
            if "optimized" in e or "rejected" in e or "error" in e:
                print(f"  [{r['mode']}] {e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    show = [r for r in results if r["grid"] is not None]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(6.5 * len(show), 5.2), squeeze=False)
    odo = next((r for r in results if r["mode"] == "odometry"), None)
    for ax, r in zip(axes[0], show):
        values, origin = r["grid"].to_occupancy_values()
        img = np.where(values < 0, 0.5, 1.0 - values / 100.0)
        h, w = values.shape
        res = r["grid"].res
        ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=[origin[0], origin[0] + w * res, origin[1], origin[1] + h * res])
        ax.plot(r["gt"][:, 0], r["gt"][:, 1], "g-", lw=1.2, label="ground truth")
        if odo is not None:
            ax.plot(odo["est"][:, 0], odo["est"][:, 1], "r--", lw=0.8, label="odometry")
        ax.plot(r["est"][:, 0], r["est"][:, 1], "b-", lw=0.9, label="SLAM")
        if r["slam"] is not None:
            poses, edges = r["slam"].graph_snapshot()
            for i, j, kind in edges:
                if kind == "loop":
                    ax.plot(poses[[i, j], 0], poses[[i, j], 1], "m-", lw=0.6, alpha=0.6)
        ax.set_title(f"{r['mode']}: ATE {r['ate']:.2f} m, ghost {r['ghost'] * 100:.1f}%, "
                     f"doubled {r['doubled'] * 100:.1f}%")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
