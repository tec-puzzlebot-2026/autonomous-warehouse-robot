"""Tests of the ArUco zones (no ROS): a size_m square in front of every marker, labeled
by its ID (21 = drop zone), kept inside the map, and used by restocking.

Run:  cd main_ros && python3 -m pytest test/test_zones.py -v
"""
# Author: Raymundo Gomez

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import restock_world as RW  # noqa: E402

from main_ros.restock.zones import (DROP_ZONE, Zone, ZoneConfig, drop_zone,  # noqa: E402
                                    zones_from_markers)


def all_inside(grid, z):
    free = grid.free()
    for x, y in z.corners:            # corners pulled 2 cm in: on free floor
        cx, cy = z.center
        r, c = grid.world_to_cell(x + 0.02 * (cx - x) / 0.5, y + 0.02 * (cy - y) / 0.5)
        if not (grid.inside(r, c) and free[r, c]):
            return False
    return True


def test_marker_21_is_the_drop_zone_and_the_others_keep_their_id():
    grid = RW.make_world()
    markers = [{'id': 21, 'x': 1.6, 'y': 0.0, 'yaw': math.pi / 2},       # bottom wall
               {'id': 7, 'x': 0.0, 'y': 2.0, 'yaw': 0.0}]                # left wall
    zones = zones_from_markers(grid, markers)
    assert [z.label for z in zones] == [DROP_ZONE, 'aruco_7']
    dz = drop_zone(zones)
    assert dz.marker_id == 21 and dz.size == 1.0
    assert abs(dz.center[0] - 1.6) < 0.06 and 0.45 < dz.center[1] < 0.8   # in front of it
    assert dz.contains(1.6, 0.5) and not dz.contains(1.6, -0.2)
    assert all(all_inside(grid, z) for z in zones)


def test_a_zone_near_a_corner_slides_along_the_wall_to_stay_inside():
    grid = RW.make_world()
    z = zones_from_markers(grid, [{'id': 21, 'x': 0.15, 'y': 0.0, 'yaw': math.pi / 2}])[0]
    assert z.inside > 0.97 and all_inside(grid, z)
    assert z.center[0] >= 0.5 - 0.06                 # pushed off the left wall


def test_unknown_yaw_faces_the_open_floor():
    grid = RW.make_world()
    z = zones_from_markers(grid, [{'id': 5, 'x': RW.W, 'y': 1.0, 'yaw': None}])[0]
    assert abs(math.cos(z.yaw) + 1.0) < 0.1          # right wall: faces -x
    assert all_inside(grid, z)


def test_config_labels_and_round_trip():
    cfg = ZoneConfig.from_dict({'size_m': 0.8, 'labels': {'13': 'drop_zone'}})
    grid = RW.make_world()
    z = zones_from_markers(grid, [{'id': 13, 'x': 2.0, 'y': 0.0, 'yaw': math.pi / 2},
                                  {'id': 21, 'x': 3.0, 'y': 0.0, 'yaw': math.pi / 2}], cfg)
    assert [q.label for q in z] == [DROP_ZONE, 'aruco_21'] and z[0].size == 0.8
    back = Zone.from_dict(z[0].to_dict())
    assert back.label == DROP_ZONE and abs(back.center[0] - z[0].center[0]) < 1e-3


def test_restocking_sets_boxes_down_inside_the_drop_zone():
    from main_ros.restock.restock_core import World
    from main_ros.restock.route import RouteConfig
    grid = RW.make_world()
    zones = zones_from_markers(grid, [{'id': 21, 'x': 1.6, 'y': 0.0, 'yaw': math.pi / 2}])
    w = World('/tmp', grid, {}, {}, [], RouteConfig(), zones)
    spot = w.free_spot(None)
    assert spot is not None and w.drop_zone.contains(*spot)
    assert w.in_dock(*w.drop_zone.center)           # boxes there count as delivered


def test_restocking_takes_the_drop_zone_when_it_sees_marker_21(tmp_path):
    """A mission folder without the drop zone: restocking learns it from the landmark
    mapper as soon as ArUco 21 is seen, shows it and saves zones.yaml."""
    import types
    from main_ros.restock import mission_store as ms
    from main_ros.restock.restock_core import RestockCore, World
    from main_ros.restock.route import RouteConfig
    grid = RW.make_world()
    shown = []
    seen = {'markers': []}
    ports = types.SimpleNamespace(landmarks=lambda: seen, show_zones=shown.append)
    core = types.SimpleNamespace(world=World(str(tmp_path), grid, {}, {}, [], RouteConfig()),
                                 zone_cfg=ZoneConfig(), p=ports, folder=str(tmp_path),
                                 _event=lambda text: None)
    RestockCore._watch_drop_zone(core, 0.0)
    assert core.world.drop_zone is None
    seen['markers'] = [{'id': 21, 'x': 1.6, 'y': 0.0, 'yaw': math.pi / 2}]
    RestockCore._watch_drop_zone(core, 1.0)            # checked every 2 s
    assert core.world.drop_zone is None
    RestockCore._watch_drop_zone(core, 2.5)
    assert core.world.drop_zone is not None and core.world.drop_zone.marker_id == 21
    assert shown and (tmp_path / ms.ZONES).exists()
