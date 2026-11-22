"""A synthetic arena like the real one, as an occupancy grid, for the restocking tests.

    4.85 x 3.66 m floor, walls all around. Three docks (trailer recesses 0.40 m wide,
    0.50 m deep) in the top wall, each with its company logo on the wall just to the
    LEFT of the opening (seen from inside): D4 soriene, D3 emezon, D2 walnut.
    A rack (0.60 x 0.25 m) and a free-standing obstacle in the room.
"""
import os

import numpy as np

from main_ros.restock.grid import OccupancyMap

RES = 0.05
W, H = 4.85, 3.66
WALL = 0.05
# Dock openings along the top wall: (x_left_edge, x_right_edge) seen from above.
# Seen from INSIDE (facing +y, the top wall), "right" is +x... the robot faces +y,
# so its right hand points to +x. Logo sits just left (-x) of each opening.
DOCKS = {'soriene': (0.90, 1.30), 'emezon': (1.90, 2.30), 'walnut': (2.90, 3.30)}
DOCK_DEPTH = 0.50
RACK = (2.0, 2.6, 1.20, 1.45)            # x0, x1, y0, y1
PILLAR = (3.8, 4.0, 0.6, 0.8)


def logo_positions(with_view=True):
    """Logo on the wall face, 0.15 m left (-x) of each opening; seen from the floor
    (view direction from the logo toward the robot: -y, roughly)."""
    if with_view:
        return {c: (x0 - 0.15, H, -1.75) for c, (x0, _) in DOCKS.items()}
    return {c: (x0 - 0.15, H) for c, (x0, _) in DOCKS.items()}


def make_world(unknown_trailers=False):
    """The arena as an OccupancyMap (map frame = arena corner, x right, y up)."""
    margin = 0.3
    w = int(round((W + 2 * margin) / RES))
    h = int(round((H + DOCK_DEPTH + 2 * margin) / RES))
    origin = (-margin, -margin)
    v = np.full((h, w), -1, dtype=np.int16)

    def fill(x0, x1, y0, y1, val):
        c0 = int(round((x0 - origin[0]) / RES))
        c1 = int(round((x1 - origin[0]) / RES))
        r0 = int(round((y0 - origin[1]) / RES))
        r1 = int(round((y1 - origin[1]) / RES))
        v[r0:r1, c0:c1] = val

    fill(0, W, 0, H, 0)                                  # floor
    fill(-WALL, W + WALL, -WALL, 0, 100)                 # bottom wall
    fill(-WALL, W + WALL, H, H + WALL, 100)              # top wall
    fill(-WALL, 0, -WALL, H + WALL, 100)                 # left wall
    fill(W, W + WALL, -WALL, H + WALL, 100)              # right wall
    for x0, x1 in DOCKS.values():                        # trailers: openings + recesses
        fill(x0, x1, H, H + DOCK_DEPTH, -1 if unknown_trailers else 0)
        if unknown_trailers:
            fill(x0, x1, H, H + 0.1, 0)                  # only the mouth was seen
        fill(x0 - WALL, x0, H, H + DOCK_DEPTH + WALL, 100)
        fill(x1, x1 + WALL, H, H + DOCK_DEPTH + WALL, 100)
        fill(x0 - WALL, x1 + WALL, H + DOCK_DEPTH, H + DOCK_DEPTH + WALL, 100)
    fill(*RACK, 100)
    fill(*PILLAR, 100)
    return OccupancyMap(v, origin, RES)


def write_map_files(base, grid):
    """What Planisferio's save_map writes: <base>.pgm/.yaml/_session.npz."""
    img = np.full(grid.values.shape, 205, np.uint8)
    img[(grid.values >= 0) & (grid.values <= 50)] = 254
    img[grid.values >= 65] = 0
    img = np.flipud(img)
    with open(base + '.pgm', 'wb') as f:
        f.write(f'P5\n{img.shape[1]} {img.shape[0]}\n255\n'.encode() + img.tobytes())
    with open(base + '.yaml', 'w') as f:
        f.write(f'image: {os.path.basename(base)}.pgm\nmode: trinary\nresolution: {grid.res}\n'
                f'origin: [{grid.origin[0]}, {grid.origin[1]}, 0.0]\nnegate: 0\n'
                'occupied_thresh: 0.65\nfree_thresh: 0.25\n')
    np.savez_compressed(base + '_session.npz', version=np.array(1))
