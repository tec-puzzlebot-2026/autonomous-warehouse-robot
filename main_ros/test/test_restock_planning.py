"""Restocking planners on a synthetic arena: docks, route, box faces, geometry (no ROS)."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import restock_world as RW  # noqa: E402

from main_ros.restock import geometry as G  # noqa: E402
from main_ros.restock.docks import find_docks, free_area_near  # noqa: E402
from main_ros.restock.route import RouteConfig, plan_route  # noqa: E402


# ----------------------------------------------------------------------------- docks
def test_each_logo_finds_the_dock_to_its_right():
    grid = RW.make_world()
    docks = find_docks(grid, RW.logo_positions())
    assert set(docks) == set(RW.DOCKS)
    for company, (x0, x1) in RW.DOCKS.items():
        d = docks[company]
        assert abs(d.entrance[0] - 0.5 * (x0 + x1)) < 0.06        # centered on the opening
        assert d.entrance[1] < RW.H                                # in front, on the floor
        assert abs(G.wrap(d.entrance[2] - math.pi / 2)) < 0.15     # facing into the trailer
        assert RW.H < d.inside[1] < RW.H + RW.DOCK_DEPTH           # inside the recess
        assert abs(d.width - (x1 - x0)) < 0.08


def test_docks_are_found_even_if_the_trailers_were_barely_seen():
    grid = RW.make_world(unknown_trailers=True)
    docks = find_docks(grid, RW.logo_positions())
    assert set(docks) == set(RW.DOCKS)


def test_a_logo_with_no_opening_to_its_right_gives_no_dock():
    grid = RW.make_world()
    docks = find_docks(grid, {'nobody': (4.6, RW.H)})        # right end of the top wall
    assert docks == {}


def test_free_area_is_open_floor_near_the_docks():
    grid = RW.make_world()
    trav = grid.traversable(0.15)
    docks = find_docks(grid, RW.logo_positions())
    xy = free_area_near(grid, trav, [d.entrance for d in docks.values()])
    assert xy is not None
    clear = grid.clearance()
    assert clear[grid.world_to_cell(*xy)] >= 0.45


# ----------------------------------------------------------------------------- route
def test_route_sees_almost_every_obstacle_edge():
    grid = RW.make_world()
    route = plan_route(grid, (2.4, 0.4, 0.0), RouteConfig())
    assert route.coverage >= 0.95
    assert 5 <= len(route.viewpoints) <= 80
    trav = grid.traversable(0.15)
    for x, y, _ in route.viewpoints:
        assert trav[grid.world_to_cell(x, y)]                  # the robot fits everywhere
    # Both long sides of the rack are looked at.
    rx0, rx1, ry0, ry1 = RW.RACK
    below = [v for v in route.viewpoints if rx0 - 0.3 < v[0] < rx1 + 0.3 and v[1] < ry0]
    above = [v for v in route.viewpoints if rx0 - 0.3 < v[0] < rx1 + 0.3 and v[1] > ry1]
    assert below and above


def test_route_order_is_shorter_than_a_random_order():
    grid = RW.make_world()
    route = plan_route(grid, (2.4, 0.4, 0.0), RouteConfig())
    pts = np.array([v[:2] for v in route.viewpoints])
    straight = np.hypot(*np.diff(pts, axis=0).T).sum()
    rng = np.random.default_rng(0)
    rand = pts[rng.permutation(len(pts))]
    assert straight < np.hypot(*np.diff(rand, axis=0).T).sum()
    assert route.length >= straight - 1e-6                     # driving >= straight lines


# ----------------------------------------------------------------------------- box faces
def test_box_on_open_floor_shows_all_four_faces():
    grid = RW.make_world()
    trav = grid.traversable(0.15)
    box = G.BoxFootprint(1.2, 1.5, -math.pi / 2)
    views = dict(G.face_viewpoints(box, grid, trav, standoff=0.30))
    assert all(views[f] is not None for f in ('front', 'left', 'back', 'right'))


def test_box_against_the_rack_hides_its_back():
    grid = RW.make_world()
    trav = grid.traversable(0.15)
    rx0, rx1, ry0, _ = RW.RACK
    # On the floor against the rack's lower side, pick face toward -y (the aisle).
    box = G.BoxFootprint(0.5 * (rx0 + rx1), ry0 - 0.06, -math.pi / 2)
    views = dict(G.face_viewpoints(box, grid, trav, standoff=0.30))
    assert views['front'] is not None
    assert views['back'] is None                                # facing the rack
    assert views['left'] is not None or views['right'] is not None   # seen off-axis


def test_viewpoints_are_visited_nearest_first():
    named = [('a', (3.0, 0.0, 0.0)), ('b', (1.0, 0.0, 0.0)), ('c', None), ('d', (2.0, 0.0, 0.0))]
    order = G.order_viewpoints((0.0, 0.0), named)
    assert [n for n, _ in order] == ['b', 'd', 'a']


# ----------------------------------------------------------------------------- geometry
def _camera():
    return G.Camera(fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                    mount=(0.10, 0.0, 0.178, 0.0, 0.26, -0.0124), base_height=0.047)


def _project(cam, p_base):
    """base_link point -> pixel (u, v)."""
    Tinv = np.linalg.inv(cam.T)
    q = Tinv @ np.array([p_base[0], p_base[1], p_base[2], 1.0])
    return cam.fx * q[0] / q[2] + cam.cx, cam.fy * q[1] / q[2] + cam.cy


def _image_box(cam, dist, lateral, bottom_z, spec=G.BoxSpec()):
    """Image bounding box of a box face `dist` m ahead (base_link heights: minus axle)."""
    zb = bottom_z - cam.base_height
    pts = [_project(cam, (dist, lateral + s * spec.width / 2, zb + h))
           for s in (-1, 1) for h in (0.0, spec.height)]
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    return [min(us), min(vs), max(us), max(vs)]


def test_locate_box_on_the_floor():
    cam = _camera()
    box = _image_box(cam, 0.60, 0.05, 0.0)
    s = G.locate_box(cam, (1.0, 2.0, 0.0), box)
    assert s.level == 0
    assert abs(s.distance - math.hypot(0.60, 0.05)) < 0.06
    assert abs(s.x - (1.0 + 0.60 + 0.05)) < 0.07                 # center = face + half depth
    assert abs(s.y - (2.0 + 0.05)) < 0.03


def test_locate_box_on_rack_levels():
    cam = _camera()
    for level, z in ((1, 0.19), (2, 0.38)):
        s = G.locate_box(cam, (0.0, 0.0, 0.0), _image_box(cam, 0.70, 0.0, z))
        assert s.level == level, (level, s.bottom_z)


def test_bearing_sign():
    cam = _camera()
    assert cam.bearing(100.0) > 0 > cam.bearing(540.0)          # left of center = positive


def test_qr_inside_the_box_is_accepted_and_the_neighbors_is_not():
    box = {'rect': {'center': [300, 250], 'size': [120, 120], 'angle_deg': 0.0}}
    poly = G.box_polygon(box)
    own = [(275, 225), (325, 225), (325, 275), (275, 275)]
    neighbor = [(370, 225), (420, 225), (420, 275), (370, 275)]   # beside the box
    straddling = [(330, 225), (380, 225), (380, 275), (330, 275)]
    assert G.qr_belongs_to_box(own, poly)
    assert not G.qr_belongs_to_box(neighbor, poly)
    assert not G.qr_belongs_to_box(straddling, poly)
    tiny = [(299, 249), (301, 249), (301, 251), (299, 251)]
    assert not G.qr_belongs_to_box(tiny, poly)                    # implausibly small


def test_face_from_scan_gives_distance_and_angle():
    rng = np.random.default_rng(1)
    ys = np.linspace(-0.06, 0.06, 25)
    a = math.radians(8.0)                                          # face turned 8 degrees
    xs = 0.20 + ys * math.tan(a)
    pts = np.column_stack((xs, ys)) + rng.normal(0, 0.002, (25, 2))
    pts = np.vstack((pts, [[0.5, 0.3], [0.1, -0.2]]))              # clutter outside the face
    d, yaw_err = G.face_from_scan(pts)
    assert abs(d - 0.20) < 0.01
    assert abs(abs(yaw_err) - a) < math.radians(2.0)


def test_pose_helpers():
    a = (1.0, 2.0, 0.5)
    b = (0.3, -0.2, 0.1)
    c = G.compose(a, b)
    r = G.relative(a, c)
    assert np.allclose(r, b)
    p = G.pose_facing((0.0, 0.0), math.pi / 2, 0.25)
    assert np.allclose(p[:2], (0.0, 0.25)) and abs(G.wrap(p[2] + math.pi / 2)) < 1e-9
