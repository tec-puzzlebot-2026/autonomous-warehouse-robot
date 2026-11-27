"""
route.py — the inspection route: look at every obstacle edge once (no ROS).

A box can only stand next to something: on the floor against an obstacle or a
wall, or on a rack. So instead of covering the floor, the route covers the EDGES
of the obstacles on the map, each seen once by the downward-tilted camera from a
distance where it sees the floor and both rack levels.

  1. Edge points: obstacle cells that touch free floor, thinned to one every
     `edge_step` m (tiny specks of LiDAR noise are dropped first).
  2. Candidate viewpoints: reachable cells in a band `standoff` m from the
     obstacles, every `spacing` m, each looking at its nearest edge.
  3. Selection: greedy set cover, i.e. repeatedly take the viewpoint that sees the
     most edge points not seen yet, until `target_coverage` or no gain.
  4. Order: the shortest tour from the robot's position, nearest neighbour then
     2-opt, with driving distances on the map (not straight lines). With
     closed=True the tour is shortest INCLUDING the way back to the start, so it
     goes round the arena as one loop instead of ending far away (the map tour).

The result reports the coverage, so the monitor can show how much of the map the
camera will have looked at.
"""
# Author: Raymundo Gomez

import math
from dataclasses import dataclass, field

import numpy as np

from main_ros.restock.geometry import wrap


@dataclass
class RouteConfig:
    robot_radius: float = 0.15       # m, the robot fits where the clearance is larger
    clearance_margin: float = 0.02   # m
    standoff_min: float = 0.40       # m, viewpoint distance to the nearest obstacle
    standoff_max: float = 0.80
    spacing: float = 0.20            # m between candidate viewpoints
    edge_step: float = 0.10          # m between edge points
    view_range: float = 1.2          # m, edges farther than this are not counted as seen
    fov: float = math.radians(60.0)  # horizontal field of view used for coverage
    min_obstacle_cells: int = 3      # smaller occupied blobs are noise
    target_coverage: float = 0.98
    min_gain: int = 2                # a viewpoint must add at least this many edge points


@dataclass
class Route:
    viewpoints: list = field(default_factory=list)   # [(x, y, theta)] in visiting order
    coverage: float = 0.0                            # fraction of reachable edge points seen
    n_edges: int = 0                                 # edge points that can be seen at all
    n_unreachable: int = 0                           # edge points no viewpoint can see
    length: float = 0.0                              # driving length of the tour [m]

    def to_dict(self):
        return {'viewpoints': [[round(float(v), 4) for v in p] for p in self.viewpoints],
                'coverage': round(self.coverage, 4), 'n_edges': self.n_edges,
                'n_unreachable': self.n_unreachable, 'length_m': round(self.length, 2)}


def _thin(cells, res, step):
    """Keep one cell per step x step square."""
    k = max(1, int(round(step / res)))
    seen, out = set(), []
    for r, c in cells:
        key = (r // k, c // k)
        if key not in seen:
            seen.add(key)
            out.append((r, c))
    return out


def edge_points(grid, cfg):
    """Obstacle cells that touch free floor, thinned to one per edge_step."""
    obs = grid.obstacles(cfg.min_obstacle_cells)
    free = grid.free()
    touch = np.zeros_like(obs)
    touch[1:, :] |= free[:-1, :]
    touch[:-1, :] |= free[1:, :]
    touch[:, 1:] |= free[:, :-1]
    touch[:, :-1] |= free[:, 1:]
    rr, cc = np.nonzero(obs & touch)
    return _thin(list(zip(rr.tolist(), cc.tolist())), grid.res, cfg.edge_step), obs


def _in_sight(grid, blocked, view_xy, edge_xy, cfg):
    """Edge points within view range and line of sight: [(index, bearing)]."""
    out = []
    for k, (ex, ey) in enumerate(edge_xy):
        dx, dy = ex - view_xy[0], ey - view_xy[1]
        d = math.hypot(dx, dy)
        if d > cfg.view_range or d < 1e-6:
            continue
        if grid.line_of_sight(view_xy, (ex, ey), blocked, ignore_last=1):
            out.append((k, math.atan2(dy, dx)))
    return out


def _visible(grid, blocked, view_xy, heading, edges, edge_xy, cfg):
    """Indices of the edge points seen from a viewpoint looking along `heading`."""
    half = cfg.fov / 2.0
    return [k for k, b in _in_sight(grid, blocked, view_xy, edge_xy, cfg)
            if abs(wrap(b - heading)) <= half]


def _tour_order(dist, start_d, closed=False):
    """Nearest neighbour from the start, then 2-opt on the open (or closed) tour."""
    n = len(start_d)
    if n == 0:
        return []
    left = set(range(n))
    cur = int(np.argmin(start_d))
    order = [cur]
    left.remove(cur)
    while left:
        cur = min(left, key=lambda j: dist[order[-1]][j])
        order.append(cur)
        left.remove(cur)

    def length(o):
        back = start_d[o[-1]] if closed else 0.0
        return start_d[o[0]] + sum(dist[o[i]][o[i + 1]] for i in range(len(o) - 1)) + back

    improved = True
    best = length(order)
    while improved:
        improved = False
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                L = length(cand)
                if L + 1e-9 < best:
                    order, best, improved = cand, L, True
    return order


def plan_route(grid, start, cfg=None, closed=False):
    """Inspection route from the robot's position `start` (x, y[, theta]).

    closed: order it as a loop that ends back at the start (the length includes the
    way back), instead of the shortest path through all viewpoints.
    """
    cfg = cfg or RouteConfig()
    edges, obs = edge_points(grid, cfg)
    if not edges:
        return Route()
    edge_xy = [grid.cell_to_world(r, c) for r, c in edges]
    trav = grid.traversable(cfg.robot_radius, cfg.clearance_margin)
    reach = np.isfinite(grid.path_lengths(start[:2], trav))
    clear_obs = grid.clearance(treat_unknown_as_obstacle=False)

    # Candidate viewpoints in the standoff band, thinned to `spacing`.
    band = reach & (clear_obs >= cfg.standoff_min) & (clear_obs <= cfg.standoff_max)
    rr, cc = np.nonzero(band)
    cands = _thin(list(zip(rr.tolist(), cc.tolist())), grid.res, cfg.spacing)
    ex = np.array(edge_xy)
    views, sees = [], []
    half = cfg.fov / 2.0
    for r, c in cands:
        x, y = grid.cell_to_world(r, c)
        in_sight = _in_sight(grid, obs, (x, y), edge_xy, cfg)
        if not in_sight:
            continue
        # Several headings per spot: toward the nearest edge, and every 45 degrees.
        d = np.hypot(ex[:, 0] - x, ex[:, 1] - y)
        k = int(np.argmin(d))
        headings = [math.atan2(ex[k, 1] - y, ex[k, 0] - x)]
        headings += [wrap(headings[0] + i * math.pi / 4) for i in range(1, 8)]
        for hd in headings:
            seen = {k2 for k2, b in in_sight if abs(wrap(b - hd)) <= half}
            if seen:
                views.append((x, y, hd))
                sees.append(seen)

    coverable = set().union(*sees) if sees else set()
    covered, chosen = set(), []
    while len(covered) < cfg.target_coverage * max(len(coverable), 1):
        gains = [len(s - covered) for s in sees]
        if not gains:
            break
        k = int(np.argmax(gains))
        if gains[k] < cfg.min_gain:
            break
        chosen.append(k)
        covered |= sees[k]

    route = Route(n_edges=len(coverable), n_unreachable=len(edges) - len(coverable))
    route.coverage = len(covered) / max(len(coverable), 1)
    if not chosen:
        return route
    pts = [views[k] for k in chosen]
    # Driving distances between the chosen viewpoints and from the start.
    fields = [grid.path_lengths(p[:2], trav) for p in pts]
    cells = [grid.world_to_cell(p[0], p[1]) for p in pts]
    n = len(pts)
    dist = [[float(fields[i][cells[j]]) if np.isfinite(fields[i][cells[j]]) else 1e6
             for j in range(n)] for i in range(n)]
    start_cell = grid.world_to_cell(*start[:2])
    start_d = [float(f[start_cell]) if grid.inside(*start_cell) and np.isfinite(f[start_cell])
               else math.hypot(p[0] - start[0], p[1] - start[1]) for f, p in zip(fields, pts)]
    order = _tour_order(dist, start_d, closed)
    route.viewpoints = [pts[i] for i in order]
    route.length = start_d[order[0]] + sum(dist[order[i]][order[i + 1]]
                                           for i in range(len(order) - 1))
    if closed:
        route.length += start_d[order[-1]]
    return route
