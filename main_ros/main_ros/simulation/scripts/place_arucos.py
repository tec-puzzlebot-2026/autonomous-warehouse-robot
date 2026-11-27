#!/usr/bin/env python3
"""
place_arucos.py — put the ArUco plates on the sim walls where they are in the real arena
========================================================================================
The real positions come from the repo, so the sim and the robot always agree:

  config/arena_map.yaml          landmark_ids / landmark_x / landmark_y / landmark_yaw
                                 (centers measured with a tape, arena map frame)
  config/aruco_localization.yaml aruco_experiment per_marker_height (center above floor)
  assets/aruco/aruco_plate_<ID>.usda  (make_wall_signs.py)

Arena map frame: origin at the inner corner of the dock wall and the wall next to the
docks; +x along that wall, away from the docks; +y along the dock wall. In the Isaac
warehouse (build_warehouse_lab.py) the docks are on the east wall, near the north end,
so the arena origin is the inner north-east corner and

    sim x = east inner face  - map x
    sim y = north inner face - map y          (a 180 deg turn: map yaw + 180 deg)

The sim walls are not exactly the measured ones (a few cm), so each marker is snapped
onto the inner face of the sim wall it belongs to (the one its yaw faces away from),
1 mm in front of it, keeping its position along the wall. The script prints, per
marker, how far it had to move. Running it again replaces /World/ArUcoMarkers.

Isaac Sim (open warehouse_lab_forklift.usd first): Window > Script Editor, open this
file, Run. Then Ctrl+S.
Without Isaac (pxr only):  python3 place_arucos.py path/to/warehouse_lab_forklift.usd
"""
# Author: Raymundo Gomez

import math
import os
import sys

import yaml
from pxr import Gf, Usd, UsdGeom

ROOT = '/World/ArUcoMarkers'
WALLS = '/World/WarehouseLab/Perimeter'
GAP = 0.001                       # m in front of the wall (no z-fighting)


def package_dir(layer_dir):
    """main_ros package folder (has config/), found upwards from the stage file."""
    d = layer_dir
    for _ in range(6):
        if os.path.isfile(os.path.join(d, 'config', 'arena_map.yaml')):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f'no config/arena_map.yaml above {layer_dir}: is the stage in the repo?')


def params(path, node):
    with open(path) as f:
        return yaml.safe_load(f)[node]['ros__parameters']


def load_markers(pkg):
    a = params(os.path.join(pkg, 'config', 'arena_map.yaml'), 'ekf_localization')
    e = params(os.path.join(pkg, 'config', 'aruco_localization.yaml'), 'aruco_experiment')
    heights = dict(zip(e['per_marker_ids'], e['per_marker_height']))
    out = []
    for i, mid in enumerate(a['landmark_ids']):
        out.append({'id': int(mid), 'x': float(a['landmark_x'][i]), 'y': float(a['landmark_y'][i]),
                    'yaw': float(a['landmark_yaw'][i]),
                    'z': float(heights.get(mid, e.get('marker_height', 0.09)))})
    return out


def wall_faces(stage):
    """Inner face of every perimeter wall: (name, axis, coord, normal, lo, hi)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    walls = stage.GetPrimAtPath(WALLS)
    if not walls.IsValid():
        raise RuntimeError(f'{WALLS} not found: is this the warehouse stage?')
    boxes = []
    for p in walls.GetChildren():
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        if not r.IsEmpty():
            boxes.append((p.GetName(), r.GetMin(), r.GetMax()))
    cx = (min(b[1][0] for b in boxes) + max(b[2][0] for b in boxes)) / 2
    cy = (min(b[1][1] for b in boxes) + max(b[2][1] for b in boxes)) / 2
    faces = []
    for name, lo, hi in boxes:
        if hi[0] - lo[0] > hi[1] - lo[1]:           # runs along x: face at a y
            inner_hi = (lo[1] + hi[1]) / 2 < cy
            faces.append((name, 'y', hi[1] if inner_hi else lo[1], 1 if inner_hi else -1,
                          lo[0], hi[0]))
        else:                                        # runs along y: face at an x
            inner_hi = (lo[0] + hi[0]) / 2 < cx
            faces.append((name, 'x', hi[0] if inner_hi else lo[0], 1 if inner_hi else -1,
                          lo[1], hi[1]))
    return faces


def arena_origin(faces):
    """Inner north-east corner: east inner face x, north inner face y."""
    east = min(f[2] for f in faces if f[1] == 'x' and f[3] < 0)
    north = min(f[2] for f in faces if f[1] == 'y' and f[3] < 0)
    return east, north


def place_one(m, faces, origin):
    sx, sy = origin[0] - m['x'], origin[1] - m['y']
    yaw = m['yaw'] + math.pi                         # facing direction in the sim
    nx, ny = round(math.cos(yaw)), round(math.sin(yaw))
    best = None
    for name, axis, coord, normal, lo, hi in faces:
        if (axis, normal) != (('x', nx) if nx else ('y', ny)):
            continue                                 # a wall facing the other way
        along = sy if axis == 'x' else sx
        off = max(lo - along, 0.0, along - hi)       # beyond the wall's ends
        d = abs((sx if axis == 'x' else sy) - coord) + off
        if best is None or d < best[0]:
            best = (d, name, axis, coord, normal)
    if best is None:
        raise RuntimeError(f'ArUco {m["id"]}: no sim wall faces yaw {m["yaw"]:.4f}')
    d, name, axis, coord, normal = best
    if axis == 'x':
        px, py = coord + normal * GAP, sy
    else:
        px, py = sx, coord + normal * GAP
    return (px, py, m['z']), math.degrees(math.atan2(ny, nx)), name, math.hypot(px - sx, py - sy)


def place(stage, layer_dir):
    pkg = package_dir(layer_dir)
    assets = os.path.join(pkg, 'main_ros', 'simulation', 'assets', 'aruco')
    faces = wall_faces(stage)
    origin = arena_origin(faces)
    if stage.GetPrimAtPath(ROOT).IsValid():
        stage.RemovePrim(ROOT)
    UsdGeom.Xform.Define(stage, ROOT)
    print(f'arena origin (map 0,0) at sim x={origin[0]:.3f} y={origin[1]:.3f}')
    for m in load_markers(pkg):
        usda = os.path.join(assets, f'aruco_plate_{m["id"]:02d}.usda')
        if not os.path.isfile(usda):
            print(f'ArUco {m["id"]}: {usda} missing, run make_wall_signs.py')
            continue
        pos, yaw_deg, wall, moved = place_one(m, faces, origin)
        prim = UsdGeom.Xform.Define(stage, f'{ROOT}/ArUco_{m["id"]:02d}')
        prim.GetPrim().GetReferences().AddReference(os.path.relpath(usda, layer_dir))
        api = UsdGeom.XformCommonAPI(prim)
        api.SetTranslate(Gf.Vec3d(*pos))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, yaw_deg))
        print(f'ArUco {m["id"]:2d}: {wall:12s} x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.4f} '
              f'facing {yaw_deg:+.0f} deg (moved {100 * moved:.1f} cm onto the sim wall)')


def main():
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.usd', '.usda', '.usdc')):
        stage = Usd.Stage.Open(sys.argv[1])
        place(stage, os.path.dirname(os.path.abspath(sys.argv[1])))
        stage.GetRootLayer().Save()
        print('saved', sys.argv[1])
        return
    import omni.usd                                  # Isaac Script Editor
    stage = omni.usd.get_context().get_stage()
    path = stage.GetRootLayer().realPath if stage else ''
    if not path:
        raise RuntimeError('open (and save) warehouse_lab_forklift.usd first')
    place(stage, os.path.dirname(path))
    print('done: check the markers, then Ctrl+S')


if __name__ == '__main__' or 'omni' in sys.modules:
    main()
