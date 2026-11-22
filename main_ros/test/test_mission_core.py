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
    mc.Step('sendero', 'ros2 launch sendero sendero.launch.py', [{'action': '/navigate_to_pose'}], 10.0),
    mc.Step('frontera', 'ros2 run frontera exploration_manager',
            [{'message': '/exploration/status'}], 10.0, role='explorer', phase='mapping'),
]


def run(core, ports, seconds, dt=0.1):
    for _ in range(int(round(seconds / dt))):
        ports.t += dt
        core.tick()


def ready_system():
    p = FakePorts()
    core = mc.MissionCore(STEPS, p)
    assert core.command('S')[0]
    run(core, p, 10)
    assert core.state == mc.READY, core.detail
    return core, p


def test_bringup_never_moves_the_robot():
    core, p = ready_system()
    assert ('explorer', 'START') not in p.sent
    assert set(p.running) == {'wheel_odometry', 'slam', 'sendero', 'frontera'}


def test_full_mapping_run_to_restocking_stub():
    core, p = ready_system()
    p.explorer = 'WAITING_FOR_START'
    assert core.command('s')[0]
    assert core.state == mc.EXPLORING
    assert p.sent[-1] == ('explorer', 'START')
    p.explorer = 'EXPLORING'
    run(core, p, 5)
    p.explorer = 'COMPLETE'
    run(core, p, 1)
    assert core.state == mc.FINISHING
    run(core, p, 10)
    assert core.state == mc.MAPPING_DONE, core.detail
    assert p.services[:2] == ['slam_optimize', 'slam_save']
    assert core.last_map['ok'] and core.last_map['message'] == '/ws/maps/m'
    assert 'mission:=restocking' in core.detail          # mission:=mapping
    assert core.mission_folder == '/ws/maps/missions/20260101_120000'
    assert p.saved_missions == ['Saved /ws/maps/m.pgm and /ws/maps/m.yaml']
    ok, msg = core.command('R')
    assert not ok and 'mission:=full' in msg


PHASE2 = STEPS + [
    mc.Step('ekf', 'ros2 launch main_ros ekf_localization.launch.py', [{'message': '/ekf/status'}],
            10.0, phase='restocking'),
    mc.Step('forklift', 'ros2 run main_ros forklift_placeholder',
            [{'action': '/forklift/pick_box'}], 10.0, phase='restocking'),
    mc.Step('landmark_mapper', 'ros2 run main_ros landmark_mapper', [{'node': '/lm'}], 10.0),
]


def test_full_mission_maps_then_restocks():
    p = FakePorts()
    core = mc.MissionCore(PHASE2, p, mission='full')
    core.command('S')
    run(core, p, 10)
    assert core.state == mc.READY
    assert 'ekf' not in p.running and 'landmark_mapper' in p.running
    core.command('S')
    p.explorer = 'COMPLETE'
    run(core, p, 12)
    assert core.state == mc.MAPPING_DONE and 'Press R' in core.detail
    assert core.command('R')[0] and core.state == mc.PHASE2_BRINGUP
    assert 'frontera' not in p.running                 # Frontera closed on purpose ...
    run(core, p, 5)
    assert core.state == mc.RESTOCKING, core.detail   # ... and that is not a failure
    assert {'ekf', 'forklift'} <= set(p.running)
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
    assert core.snapshot()['restock'] == {'state': 'LOAD'}


def test_restocking_mission_brings_up_phase2_steps_and_starts_on_s():
    p = FakePorts()
    core = mc.MissionCore(PHASE2, p, mission='restocking')
    assert 'frontera' not in [s.name for s in core.steps]
    core.command('S')
    run(core, p, 10)
    assert core.state == mc.READY and 'start restocking' in core.detail
    assert {'ekf', 'forklift', 'landmark_mapper'} <= set(p.running)
    assert mc.RESET not in core.allowed() and mc.SAVE_MAP not in core.allowed()
    core.command('S')
    assert core.state == mc.RESTOCKING and p.restock.mission == 'restocking'
    # A script dies: the restocking pauses, the bring-up restarts it, S resumes.
    p.running.pop('ekf')
    run(core, p, 0.2)
    assert core.state == mc.ERROR and p.restock.paused
    core.command('S')
    run(core, p, 10)
    assert core.state == mc.READY
    core.command('S')
    assert core.state == mc.RESTOCKING and p.restock.calls.count('start') == 1
    core.command('X')
    assert 'abort' in p.restock.calls


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
    assert not core.command('R')[0]
    assert not core.command('hello')[0]
    core, p = ready_system()
    assert not core.command('R')[0]           # restocking only after mapping
    assert 'R' not in core.snapshot()['allowed']


@pytest.mark.parametrize('word', ['S', 'start', 'C', 'resume'])
def test_aliases(word):
    core, p = ready_system()
    assert core.command(word)[0] and core.state == mc.EXPLORING
