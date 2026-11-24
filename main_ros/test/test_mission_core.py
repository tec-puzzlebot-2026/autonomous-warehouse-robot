"""Tests of the mission state machine with fake ports (no ROS needed).

Run:  cd main_ros && python3 -m pytest test/test_mission_core.py -v
"""
# Author: Raymundo Gomez

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main_ros.control import mission_core as mc  # noqa: E402


class Handle:
    def __init__(self, result, delay, ports):
        self._result, self._t, self._p = result, ports.t + delay, ports

    def done(self):
        return self._p.t >= self._t

    def result(self):
        return self._result


class Job:
    def __init__(self, ports, delay=0.5):
        self._t, self._p = ports.t + delay, ports

    def done(self):
        return self._p.t >= self._t


class FakePorts:
    """Scripts become ready `ready_after` seconds after they start."""

    def __init__(self, ready_after=None):
        self.t = 0.0
        self.running = {}            # name -> start time
        self.launched = set()
        self.stopped = set()
        self.ready_after = ready_after or {}
        self.external_ready = True
        self.explorer = ''
        self.slam = {'backend_busy': '0'}
        self.sent = []               # (target, command)
        self.services = []
        self.zero = 0
        self.launch_ok = True
        self.logs = []

    # time
    def now(self):
        return self.t

    def prepare_bringup(self):
        pass

    # processes
    def launch(self, step):
        if not self.launch_ok:
            return False, 'boom'
        self.running[step.name] = self.t
        self.launched.add(step.name)
        self.stopped.discard(step.name)
        return True, 'ok'

    def alive(self, step):
        return step.external or step.name in self.running

    def ready(self, step):
        if step.external:
            return self.external_ready
        t0 = self.running.get(step.name)
        return t0 is not None and self.t - t0 >= self.ready_after.get(step.name, 1.0)

    def exit_code(self, step):
        return None if step.name in self.running else 1

    def log_tail(self, step, n):
        return ['last line']

    def log_path(self, step):
        return f'/tmp/{step.name}.log'

    def was_launched(self, step):
        return step.name in self.launched and step.name not in self.stopped

    def stop_steps(self, steps):
        for s in steps:
            self.running.pop(s.name, None)
            self.stopped.add(s.name)
        return Job(self)

    # commands
    def send_explorer(self, cmd):
        self.sent.append(('explorer', cmd))

    def send_navigation(self, cmd):
        self.sent.append(('navigation', cmd))

    def zero_velocity(self):
        self.zero += 1

    def call_service(self, key):
        self.services.append(key)
        msg = {'slam_save': 'Saved /ws/maps/m.pgm and /ws/maps/m.yaml'}.get(key, 'ok')
        return Handle((True, msg), 0.3, self)

    # status
    def explorer_status(self):
        return self.explorer, 0.1

    def slam_status(self):
        return self.slam

    def publish_map_save_status(self, text):
        self.logs.append(text)

    def on_state(self, state):
        pass

    def log(self, text):
        self.logs.append(text)

    # phase 2
    def save_mission(self, raw):
        self.saved_missions = getattr(self, 'saved_missions', []) + [raw]
        return True, '/ws/maps/missions/20260101_120000', '/ws/maps/missions/20260101_120000'

    def make_restock(self, mission, folder):
        self.restock = FakeRestock(mission, folder)
        return self.restock

    def make_route_drive(self, folder):
        self.restock = FakeRestock('route_only', folder)
        return self.restock

    def make_tour(self):
        self.tour = FakeTour()
        return self.tour

    def load_grid(self, folder):
        return 'grid of ' + folder

    def plan_route(self, folder):
        self.planned = folder
        return Handle((True, '12 viewpoints, 9.5 m, 98 % of the edges'), 0.5, self)


class FakeTour:
    """Stands in for map_tour.MapTour: finishes when .done is set."""

    def __init__(self):
        self.done = self.paused = False
        self.outcome, self.detail = '', ''
        self.calls = []

    def start(self, grid):
        self.calls.append(('start', grid))

    def tick(self):
        pass

    def pause(self):
        self.paused = True
        self.calls.append('pause')

    def resume(self):
        self.paused = False
        self.calls.append('resume')

    def finish_early(self):
        self.done, self.outcome = True, 'stopped by the operator'
        self.calls.append('finish_early')

    def snapshot(self):
        return {'goal': 1}


class FakeRestock:
    """Stands in for restock_core.RestockCore."""

    def __init__(self, mission, folder):
        self.mission, self.folder = mission, folder
        self.started = self.done = self.paused = False
        self.error = ''
        self.outcome = ''
        self.state, self.phase, self.detail = 'LOAD', 'LOAD', ''
        self.calls = []

    def start(self):
        self.started = True
        self.calls.append('start')

    def tick(self):
        self.calls.append('tick')

    def pause(self):
        self.paused = True
        self.calls.append('pause')

    def resume(self):
        self.paused = False
        self.error = ''
        self.calls.append('resume')

    def abort(self, why=''):
        self.calls.append('abort')

    def snapshot(self):
        return {'state': self.state}


STEPS = [
    mc.Step('isaac_sim', '', [{'clock': '/clock'}], 30.0),
    mc.Step('wheel_odometry', 'ros2 run main_ros wheel_odometry', [{'message': '/wheel/odom'}], 10.0),
    mc.Step('slam', 'ros2 run planisferio slam', [{'message': '/map'}], 10.0, role='slam'),
    mc.Step('sendero', 'ros2 run sendero navigator', [{'action': '/navigate_to_pose'}], 10.0),
    mc.Step('frontera', 'ros2 run frontera exploration_manager',
            [{'message': '/exploration/status'}], 10.0, role='explorer', stage='mapping'),
    mc.Step('camera', 'ros2 launch main_ros vision.launch.py', [{'message': '/camera/image'}],
            10.0, stage='quality'),
    mc.Step('landmark_mapper', 'ros2 run main_ros landmark_mapper', [{'node': '/lm'}], 10.0,
            stage='quality'),
    mc.Step('ekf', 'ros2 launch main_ros ekf_localization.launch.py',
            [{'message': '/ekf/status'}], 10.0, stage='restocking'),
    mc.Step('forklift', 'ros2 run main_ros forklift_placeholder',
            [{'action': '/forklift/pick_box'}], 10.0, stage='restocking'),
]
CORE = {'wheel_odometry', 'slam', 'sendero'}


def run(core, ports, seconds, dt=0.1):
    for _ in range(int(round(seconds / dt))):
        ports.t += dt
        core.tick()


def ready_system(**kw):
    p = FakePorts()
    core = mc.MissionCore(STEPS, p, **kw)
    assert core.command('S')[0]
    run(core, p, 10)
    return core, p


def map_stage_1(core, p):
    """Stage 1 to its end: S, Frontera explores, COMPLETE -> MAPPING_DONE."""
    assert core.state == mc.READY, core.detail
    p.explorer = 'WAITING_FOR_START'
    assert core.command('s')[0] and core.state == mc.EXPLORING
    p.explorer = 'EXPLORING'
    run(core, p, 2)
    p.explorer = 'COMPLETE'
    run(core, p, 10)
    assert core.state == mc.MAPPING_DONE, core.detail


def test_bringup_starts_only_stage_1_and_never_moves():
    core, p = ready_system()
    assert core.state == mc.READY
    assert ('explorer', 'START') not in p.sent
    assert set(p.running) == CORE | {'frontera'}          # no camera, EKF or forklift


def test_stage_1_saves_the_map_and_a_mission_folder_then_waits():
    core, p = ready_system()
    map_stage_1(core, p)
    assert p.services[:2] == ['slam_optimize', 'slam_save']
    assert core.last_map['ok'] and core.last_map['message'] == '/ws/maps/m'
    assert core.mission_folder == '/ws/maps/missions/20260101_120000'
    assert p.saved_missions == ['Saved /ws/maps/m.pgm and /ws/maps/m.yaml']
    snap = core.snapshot()
    assert {'START', 'SKIP'} <= set(snap['allowed'])
    assert snap['stages'][0][2] == 'done' and snap['stages'][1][2] == 'todo'
    run(core, p, 30)
    assert core.state == mc.MAPPING_DONE                   # waits for a key


def test_all_four_stages_with_a_key_after_each():
    core, p = ready_system()
    map_stage_1(core, p)
    # stage 2: Frontera closed, camera + landmark mapper started, the tour drives
    assert core.command('S')[0] and core.state == mc.STAGE_BRINGUP
    assert 'frontera' not in p.running
    run(core, p, 5)
    assert core.state == mc.QUALITY, core.detail
    assert {'camera', 'landmark_mapper'} <= set(p.running)
    assert p.tour.calls[0] == ('start', 'grid of /ws/maps/missions/20260101_120000')
    assert core.command('P')[0] and core.state == mc.QUALITY_PAUSED and p.tour.paused
    assert core.command('S')[0] and core.state == mc.QUALITY and not p.tour.paused
    p.tour.done, p.tour.outcome = True, 'tour complete (0 goals skipped)'
    run(core, p, 10)
    assert core.state == mc.QUALITY_DONE, core.detail
    assert len(p.saved_missions) == 2                     # a new folder with the landmarks
    assert 'SAVE_MAP' not in core.allowed()
    # stage 3: the route, then wait
    assert core.command('S')[0] and core.state == mc.PLANNING
    run(core, p, 1)
    assert core.state == mc.ROUTE_READY and '12 viewpoints' in core.detail
    run(core, p, 10)
    assert core.state == mc.ROUTE_READY
    # stage 4: EKF + forklift started, Planisferio localization only, restocking runs
    assert core.command('S')[0] and core.state == mc.STAGE_BRINGUP
    run(core, p, 5)
    assert core.state == mc.RESTOCKING, core.detail
    assert {'ekf', 'forklift'} <= set(p.running) and 'frontera' not in p.running
    assert 'slam_localization_only' in p.services
    r = p.restock
    assert r.started and r.mission == 'full' and r.folder == core.mission_folder
    assert core.command('P')[0] and core.state == mc.RESTOCK_PAUSED and r.paused
    assert core.command('S')[0] and core.state == mc.RESTOCKING and not r.paused
    r.error = 'no ArUco found'
    run(core, p, 0.2)
    assert core.state == mc.RESTOCK_PAUSED and 'no ArUco' in core.detail
    core.command('S')
    assert core.state == mc.RESTOCKING and r.error == ''
    r.done, r.outcome = True, 'COMPLETE: route finished'
    run(core, p, 0.2)
    assert core.state == mc.RESTOCK_DONE
    assert [m for _, _, m in core.stage_progress()] == ['done'] * 4
    core.command('X')
    run(core, p, 5)
    assert 'abort' in r.calls and not p.running
    assert p.services.count('slam_save') == 2              # no save once the map is final


def test_k_skips_stage_2_and_restocking_starts_the_camera_itself():
    core, p = ready_system()
    map_stage_1(core, p)
    assert core.command('K')[0] and core.state == mc.PLANNING
    assert 'frontera' not in p.running and p.planned == core.mission_folder
    run(core, p, 1)
    assert core.state == mc.ROUTE_READY
    assert [m for _, _, m in core.stage_progress()] == ['done', 'skipped', 'done', 'todo']
    core.command('S')
    run(core, p, 6)
    assert core.state == mc.RESTOCKING, core.detail
    assert {'camera', 'landmark_mapper', 'ekf', 'forklift'} <= set(p.running)


def test_route_only_drives_the_route_without_starting_anything():
    core, p = ready_system(settings=mc.MissionSettings(restock_mode='route_only'))
    map_stage_1(core, p)
    assert core.command('K')[0]
    run(core, p, 1)
    assert core.state == mc.ROUTE_READY
    assert 'follow the route' in core.snapshot()['start_meaning']
    assert core.command('S')[0] and core.state == mc.RESTOCKING, core.detail
    assert set(p.running) == CORE                          # no camera, EKF or forklift
    assert 'slam_localization_only' in p.services
    r = p.restock
    assert r.started and r.mission == 'route_only' and r.folder == core.mission_folder
    assert core.command('P')[0] and core.state == mc.RESTOCK_PAUSED and r.paused
    assert core.command('S')[0] and core.state == mc.RESTOCKING and not r.paused
    r.done, r.outcome = True, 'COMPLETE: route finished (0 viewpoints skipped)'
    run(core, p, 0.2)
    assert core.state == mc.RESTOCK_DONE and 'ROUTE DONE' in core.detail


def test_finish_ends_the_tour_early_and_saves():
    core, p = ready_system()
    map_stage_1(core, p)
    core.command('S')
    run(core, p, 5)
    assert core.command('F')[0] and core.state == mc.FINISHING
    assert 'finish_early' in p.tour.calls
    run(core, p, 10)
    assert core.state == mc.QUALITY_DONE


def test_saved_map_waits_for_the_pose_then_offers_stage_2():
    core, p = ready_system(start_folder='/ws/maps/missions/old')
    assert 'frontera' not in p.running and set(p.running) == CORE
    assert core.state == mc.LOCALIZING
    run(core, p, 5)
    assert core.state == mc.LOCALIZING                    # no pose yet
    p.slam['localized'] = '1'                             # RViz 2D Pose Estimate
    run(core, p, 0.2)
    assert core.state == mc.MAPPING_DONE
    assert mc.RESET not in core.allowed()
    assert core.command('K')[0]
    run(core, p, 1)
    assert core.state == mc.ROUTE_READY and p.planned == '/ws/maps/missions/old'


def test_an_error_in_stage_2_resumes_the_tour_after_the_bringup():
    core, p = ready_system()
    map_stage_1(core, p)
    core.command('S')
    run(core, p, 5)
    del p.running['camera']                               # the camera crashed
    run(core, p, 0.2)
    assert core.state == mc.ERROR and p.tour.paused
    core.command('S')
    run(core, p, 6)
    assert core.state == mc.QUALITY_PAUSED, core.detail
    assert 'frontera' not in p.running                    # still closed
    core.command('S')
    assert core.state == mc.QUALITY


def test_pause_resume_and_finish_by_operator():
    core, p = ready_system()
    core.command('S')
    p.explorer = 'EXPLORING'
    run(core, p, 1)
    assert core.command('P')[0] and core.state == mc.PAUSED
    assert ('explorer', 'PAUSE') in p.sent and ('navigation', 'PAUSE') in p.sent
    assert p.zero >= 1
    assert core.command('S')[0] and core.state == mc.EXPLORING
    assert p.sent[-2:] == [('navigation', 'RESUME'), ('explorer', 'START')]
    assert core.command('F')[0] and core.state == mc.FINISHING
    run(core, p, 10)
    assert core.state == mc.MAPPING_DONE and core.end_class == 'FINISHED_BY_OPERATOR'


def test_frontera_error_end_still_saves_the_map():
    core, p = ready_system()
    core.command('S')
    p.explorer = 'ERROR_NAV_UNRESPONSIVE'
    run(core, p, 10)
    assert core.state == mc.MAPPING_DONE and core.end_class == 'ERROR_NAV_UNRESPONSIVE'
    assert 'slam_save' in p.services


def test_reset_restarts_frontera_and_clears_slam():
    core, p = ready_system()
    core.command('S')
    p.explorer = 'EXPLORING'
    run(core, p, 1)
    assert core.command('N')[0] and core.state == mc.RESETTING
    run(core, p, 10)
    assert core.state == mc.READY, core.detail
    assert 'slam_reset' in p.services
    assert ('navigation', 'CLEAR_COSTMAPS') in p.sent
    assert 'frontera' in p.running            # relaunched


def test_stop_saves_map_and_closes_everything():
    core, p = ready_system()
    core.command('S')
    assert core.command('X')[0]
    run(core, p, 5)
    assert core.state == mc.STOPPED
    assert 'slam_save' in p.services
    assert not p.running
    assert core.command('S')[0] and core.state == mc.BRINGUP


def test_bringup_timeout_and_retry():
    p = FakePorts()
    p.external_ready = False                  # Isaac Sim not playing
    core = mc.MissionCore(STEPS, p)
    core.command('S')
    run(core, p, 31)
    assert core.state == mc.ERROR and 'isaac_sim' in core.detail
    p.external_ready = True
    assert core.command('S')[0]
    run(core, p, 10)
    assert core.state == mc.READY


def test_a_dead_script_halts_the_robot():
    core, p = ready_system()
    core.command('S')
    p.explorer = 'EXPLORING'
    run(core, p, 1)
    del p.running['slam']                     # SLAM crashed
    run(core, p, 0.2)
    assert core.state == mc.ERROR and 'slam' in core.detail
    assert ('navigation', 'CANCEL') in p.sent


def test_commands_rejected_in_wrong_states():
    p = FakePorts()
    core = mc.MissionCore(STEPS, p)
    assert not core.command('P')[0]
    assert not core.command('K')[0]
    assert not core.command('hello')[0]
    core, p = ready_system()
    assert not core.command('K')[0]           # nothing to skip before stage 1 is done
    assert 'SKIP' not in core.snapshot()['allowed']


@pytest.mark.parametrize('word', ['S', 'start', 'C', 'resume'])
def test_aliases(word):
    core, p = ready_system()
    assert core.command(word)[0] and core.state == mc.EXPLORING


def test_with_previous_steps_start_in_parallel():
    steps = [
        mc.Step('slam', 'ros2 run planisferio slam', [{'message': '/map'}], 10.0, role='slam'),
        mc.Step('sendero', 'ros2 run sendero navigator', [{'action': '/navigate_to_pose'}], 10.0,
                with_previous=True),
        mc.Step('frontera', 'ros2 run frontera exploration_manager',
                [{'message': '/exploration/status'}], 10.0, role='explorer', with_previous=True),
    ]
    p = FakePorts()
    core = mc.MissionCore(steps, p)
    core.command('S')
    core.tick()                                   # one tick: all three are launched
    assert {'slam', 'sendero', 'frontera'} <= set(p.running)
    run(core, p, 10)
    assert core.state == mc.READY
    assert sum(1 for e in core.events if 'in parallel' in e[1]) == 2
