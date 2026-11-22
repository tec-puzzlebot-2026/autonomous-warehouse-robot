"""The mission folder and the EKF parameters made from discovered markers (no ROS)."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import restock_world as RW  # noqa: E402

from main_ros.restock import mission_store as MS  # noqa: E402
from main_ros.restock.docks import Dock, find_docks  # noqa: E402
from main_ros.restock.grid import OccupancyMap  # noqa: E402
from main_ros.restock.landmarks import LandmarkSet, marker_yaw_from_optical  # noqa: E402


_write_planisferio_files = RW.write_map_files


def test_mission_folder_round_trip(tmp_path):
    store = MS.MissionStore(str(tmp_path / 'missions'))
    grid = RW.make_world()
    base = str(tmp_path / 'map_20261121_120000')
    _write_planisferio_files(base, grid)
    msg = f'Saved {base}.pgm and {base}.yaml and session {base}_session.npz'
    assert MS.parse_saved_base(msg) == base

    folder = store.new_folder('20261121_120000')
    copied = MS.copy_map_files(MS.parse_saved_base(msg), folder)
    assert set(copied) == {MS.MAP_PGM, MS.MAP_YAML, MS.SESSION}
    store.mark_latest(folder)
    assert store.resolve() == os.path.realpath(folder)
    assert store.resolve('20261121_120000') == folder

    # The copied map reads back as the same grid.
    back = OccupancyMap.from_files(os.path.join(folder, MS.MAP_YAML))
    assert back.values.shape == grid.values.shape
    assert np.mean((back.values >= 65) == (grid.values >= 65)) > 0.999

    docks = find_docks(back, RW.logo_positions())
    MS.write_yaml(os.path.join(folder, MS.DOCKS), {'docks': [d.to_dict() for d in docks.values()]})
    loaded = [Dock.from_dict(d) for d in MS.read_yaml(os.path.join(folder, MS.DOCKS))['docks']]
    assert {d.company for d in loaded} == set(RW.DOCKS)


def test_latest_falls_back_to_the_newest_complete_folder(tmp_path):
    store = MS.MissionStore(str(tmp_path))
    for name in ('20261101_100000', '20261102_100000'):
        f = store.new_folder(name)
        for fn in (MS.MAP_YAML, MS.SESSION):
            open(os.path.join(f, fn), 'w').close()
    os.makedirs(os.path.join(str(tmp_path), '20261103_100000'))     # incomplete
    assert store.resolve().endswith('20261102_100000')


def test_ekf_parameters_use_the_measured_markers():
    template = {'frame_id': 'map', 'landmark_ids': [5], 'landmark_x': [0.0], 'landmark_y': [0.0],
                'landmark_yaw': [0.0], 'localization_mode': 'fixed', 'search_rotation': False,
                'latency_s': 0.2}
    lms = [{'id': 21, 'x': 0.8, 'y': 3.6, 'yaw': -1.57}, {'id': 7, 'x': 4.5, 'y': 0.0, 'yaw': None}]
    p = MS.ekf_parameters(template, lms, (0, 4.85, 0, 3.66), {'search_rotation': True})
    q = p['ekf_localization']['ros__parameters']
    assert q['landmark_ids'] == [7, 21]
    assert q['landmark_x'] == [4.5, 0.8] and q['landmark_yaw'] == [999.0, -1.57]
    assert q['localization_mode'] == 'auto' and q['search_rotation'] is True
    assert q['latency_s'] == 0.2 and q['arena_bounds'] == [0.0, 4.85, 0.0, 3.66]
    assert template['landmark_ids'] == [5]                          # template untouched


def test_landmarks_are_robust_to_outliers():
    s = LandmarkSet(min_count=5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s.add(21, 0.8 + rng.normal(0, 0.01), 3.6 + rng.normal(0, 0.01), -1.57)
    s.add(21, 3.0, 1.0, 0.0)                                          # a bad sighting
    s.add(5, 1.0, 1.0)                                                # seen only once
    conf = s.confirmed()
    assert set(conf) == {21}
    x, y, yaw, n, _ = conf[21]
    assert abs(x - 0.8) < 0.01 and abs(y - 3.6) < 0.01 and n == 20
    assert abs(yaw + 1.57) < 0.01


def test_marker_yaw_facing_the_camera():
    # Camera looking straight ahead (optical z = base x); a marker facing the camera
    # has its z axis pointing back at it: optical -z, i.e. base -x.
    from main_ros.vision.aruco import handeye
    T = handeye.mount_to_X(0.1, 0.0, 0.2, 0.0, 0.0, 0.0)
    q = (0.0, 1.0, 0.0, 0.0)                                          # 180 deg about y
    yaw = marker_yaw_from_optical(q, T, robot_yaw=math.pi / 2)
    assert abs(math.atan2(math.sin(yaw + math.pi / 2), math.cos(yaw + math.pi / 2))) < 1e-6


def test_dock_contains_points_inside_the_trailer_only():
    grid = RW.make_world()
    d = find_docks(grid, RW.logo_positions())['emezon']
    x0, x1 = RW.DOCKS['emezon']
    assert d.contains(0.5 * (x0 + x1), RW.H + 0.3)
    assert not d.contains(0.5 * (x0 + x1), RW.H - 0.4)              # in front of it
    assert not d.contains(x1 + 0.4, RW.H + 0.3)                      # beside it


def test_save_mission_folder_end_of_mapping(tmp_path):
    grid = RW.make_world()
    base = str(tmp_path / 'maps' / 'map_0001')
    os.makedirs(os.path.dirname(base))
    _write_planisferio_files(base, grid)
    lm = {'markers': [{'id': 7, 'x': 1.5, 'y': RW.H, 'yaw': -1.5708},
                      {'id': 99, 'x': 0.0, 'y': 1.0, 'yaw': 0.0}],
          'logos': [{'id': c, 'x': v[0], 'y': v[1], 'yaw': v[2]}
                    for c, v in RW.logo_positions().items()] + [{'id': 'yinyang', 'x': 0, 'y': 0}]}
    store = MS.MissionStore(str(tmp_path / 'missions'))
    template = {'localization_mode': 'fixed', 'search_rotation': False, 'init_frames': 30}
    folder, summary = MS.save_mission_folder(store, base, lm, template,
                                             ('soriene', 'emezon', 'walnut'), [5, 7, 13])
    assert store.resolve() == os.path.realpath(folder)
    assert summary['markers'] == [7] and summary['docks'] == ['emezon', 'soriene', 'walnut']
    assert summary['missing_docks'] == [] and summary['session']
    ekf = MS.read_yaml(os.path.join(folder, MS.EKF))['ekf_localization']['ros__parameters']
    assert ekf['landmark_ids'] == [7] and ekf['localization_mode'] == 'auto'
    assert ekf['search_rotation'] is False and ekf['init_frames'] == 30
    logos = MS.read_yaml(os.path.join(folder, MS.LOGOS))['logos']
    assert sorted(lg['id'] for lg in logos) == ['emezon', 'soriene', 'walnut']
