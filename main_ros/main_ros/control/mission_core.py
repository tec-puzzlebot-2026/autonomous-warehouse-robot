"""
mission_core.py — the mission state machine (no ROS)
=====================================================
The Program Manager decides; other scripts act. This module is the decision
part: a state machine driven by operator commands and by what the other
scripts report. It never computes a path, a map or a velocity. It starts and
stops scripts, sends them short commands and waits for their answers through
a `ports` object (implemented by the ROS node in program_manager.py, and by
a fake in the tests).

One run goes through four stages. After each stage the robot stops and the
monitor waits for a key: S starts the next stage, K skips the optional one.
Each stage starts only the scripts it needs, and closes the ones it no longer
needs (Frontera after stage 1), so the Jetson is never running everything.

  Stage 1  MAPPING       bring-up, odometry, Planisferio, Sendero, Frontera
           IDLE ─S─> BRINGUP ─> READY ─S─> EXPLORING ⇄P/S⇄ PAUSED
                     ─F, or Frontera ends─> FINISHING (optimize, save) ─> MAPPING_DONE
  Stage 2  MAP QUALITY   (optional, K skips it) + camera, ArUco, logos, landmark mapper;
           Frontera closed. The robot drives along every edge of the map and back to
           its start (map_tour.py) while Planisferio keeps mapping: loop closures,
           ArUco markers and logos placed on the map.
           MAPPING_DONE ─S─> STAGE_BRINGUP ─> QUALITY ⇄P/S⇄ QUALITY_PAUSED
                     ─tour ends, or F─> FINISHING ─> QUALITY_DONE
  Stage 3  ROUTE         the restocking patrol route on the final map (no motion)
           QUALITY_DONE ─S─> PLANNING ─> ROUTE_READY      (MAPPING_DONE ─K─> PLANNING)
  Stage 4  RESTOCKING    + EKF, box detectors, QR, forklift; Planisferio switches to
           localization only (restock/restock_core.py does the work)
           ROUTE_READY ─S─> STAGE_BRINGUP ─> RESTOCKING ⇄P/S⇄ RESTOCK_PAUSED ─> RESTOCK_DONE

  A saved map (map:=<mission folder>): BRINGUP loads it into Planisferio, then
  LOCALIZING waits for the robot's pose (RViz 2D Pose Estimate), then MAPPING_DONE:
  S improves that map (stage 2), K goes straight to the route (stage 3).

  Every mission folder (end of stage 1 and of stage 2) holds the map, the
  Planisferio session, the markers, logos and docks found so far, and the EKF file.

  N (reset, stage 1 only): stop the robot, restart Frontera, clear SLAM -> READY
  X (stop) from any active state: stop, save the map (stages 1-2), close everything
  M (save map): whenever SLAM is running and the map is still changing.
  An error stops in ERROR; S brings the scripts up again and returns to where the
  run was (a paused tour or restocking continues with S).

Safety rules kept from the CALL-M program manager:
  * bringing the system up never moves the robot; motion starts only on S;
  * every state change that ends motion halts the robot first (Frontera,
    Sendero and a zero velocity);
  * a script that dies while the mission runs halts the robot (ERROR).
"""
# Author: Raymundo Gomez

import collections
import time
from dataclasses import dataclass, field

IDLE = 'IDLE'
BRINGUP = 'BRINGUP'
LOCALIZING = 'LOCALIZING'
READY = 'READY'
EXPLORING = 'EXPLORING'
PAUSED = 'PAUSED'
FINISHING = 'FINISHING'
MAPPING_DONE = 'MAPPING_DONE'
STAGE_BRINGUP = 'STAGE_BRINGUP'
QUALITY = 'QUALITY'
QUALITY_PAUSED = 'QUALITY_PAUSED'
QUALITY_DONE = 'QUALITY_DONE'
PLANNING = 'PLANNING'
ROUTE_READY = 'ROUTE_READY'
RESTOCKING = 'RESTOCKING'
RESTOCK_PAUSED = 'RESTOCK_PAUSED'
RESTOCK_DONE = 'RESTOCK_DONE'
RESETTING = 'RESETTING'
STOPPING = 'STOPPING'
STOPPED = 'STOPPED'
ERROR = 'ERROR'

START = 'START'
PAUSE = 'PAUSE'
FINISH = 'FINISH'
SKIP = 'SKIP'
SAVE_MAP = 'SAVE_MAP'
RESET = 'RESET'
STOP = 'STOP'

# Keys / words accepted on the command topic.
ALIASES = {
    'S': START, 'START': START, 'C': START, 'CONFIRM': START, 'RESUME': START, 'NEXT': START,
    'P': PAUSE, 'PAUSE': PAUSE,
    'F': FINISH, 'FINISH': FINISH, 'FINISH_MAPPING': FINISH,
    'K': SKIP, 'SKIP': SKIP,
    'M': SAVE_MAP, 'SAVE_MAP': SAVE_MAP, 'SAVE': SAVE_MAP,
    'N': RESET, 'RESET': RESET,
    'X': STOP, 'STOP': STOP,
}

ALLOWED = {
    IDLE: [START],
    BRINGUP: [STOP],
    LOCALIZING: [STOP],
    READY: [START, SAVE_MAP, RESET, STOP],
    EXPLORING: [PAUSE, FINISH, SAVE_MAP, RESET, STOP],
    PAUSED: [START, FINISH, SAVE_MAP, RESET, STOP],
    FINISHING: [STOP],
    MAPPING_DONE: [START, SKIP, SAVE_MAP, RESET, STOP],
    STAGE_BRINGUP: [STOP],
    QUALITY: [PAUSE, FINISH, SAVE_MAP, STOP],
    QUALITY_PAUSED: [START, FINISH, SAVE_MAP, STOP],
    QUALITY_DONE: [START, SAVE_MAP, STOP],
    PLANNING: [STOP],
    ROUTE_READY: [START, STOP],
    RESTOCKING: [PAUSE, STOP],
    RESTOCK_PAUSED: [START, STOP],
    RESTOCK_DONE: [STOP],
    RESETTING: [STOP],
    STOPPING: [],
    STOPPED: [START],
    ERROR: [START, SAVE_MAP, RESET, STOP],
}

# What S and K do in each state (shown by the monitor next to the keys).
START_MEANING = {
    IDLE: 'bring up', STOPPED: 'bring up', READY: 'start mapping', PAUSED: 'resume mapping',
    ERROR: 'retry the bring-up', MAPPING_DONE: 'improve the map (stage 2)',
    QUALITY_PAUSED: 'resume the map tour', QUALITY_DONE: 'plan the route (stage 3)',
    ROUTE_READY: 'start restocking (stage 4)', RESTOCK_PAUSED: 'resume restocking',
}
SKIP_MEANING = {MAPPING_DONE: 'skip to the route'}

# Frontera end states (its /exploration/status once a run is over).
EXPLORATION_END = ('COMPLETE', 'COMPLETE_PARTIAL', 'STAGNATED', 'TIME_LIMIT')

# States in which the launched scripts must stay alive.
SUPERVISED = (LOCALIZING, READY, EXPLORING, PAUSED, FINISHING, MAPPING_DONE, QUALITY,
              QUALITY_PAUSED, QUALITY_DONE, PLANNING, ROUTE_READY, RESTOCKING,
              RESTOCK_PAUSED, RESTOCK_DONE)

STAGES = ('core', 'mapping', 'quality', 'restocking')
STAGE_TITLES = (('mapping', '1 map'), ('quality', '2 quality'), ('route', '3 route'),
                ('restocking', '4 restock'))


@dataclass
class Step:
    """One script of the system (or an external precondition such as Isaac Sim).

    Attributes:
        name: Short name shown in the monitor.
        cmd: Shell command that starts it ('' = external: only wait).
        ready: List of readiness checks, e.g. [{'message': '/map'}].
        timeout: Seconds to wait for readiness.
        critical: If it dies while the mission runs, the robot is halted.
        role: 'explorer' (restarted on RESET), 'slam' (map services) or ''.
        hint: Extra text for the operator when it does not become ready.
        stage: 'core' (started at bring-up, runs the whole time), 'mapping'
            (stage 1 only: Frontera), 'quality' (from stage 2 on: camera, ArUco,
            logos, landmark mapper) or 'restocking' (stage 4).
        with_previous: True = started together with the step before it instead
            of after it is ready (steps that do not need each other start in
            parallel); its readiness is still checked in order.
    """
    name: str
    cmd: str = ''
    ready: list = field(default_factory=list)
    timeout: float = 60.0
    critical: bool = True
    role: str = ''
    hint: str = ''
    stage: str = 'core'
    with_previous: bool = False

    @property
    def external(self):
        """True if the Program Manager does not start this step itself."""
        return not self.cmd


@dataclass
class MissionSettings:
    """Timing of the composite operations (see program_manager.yaml)."""
    start_ack_timeout_s: float = 5.0
    start_retries: int = 3
    optimize_settle_min_s: float = 2.0
    optimize_timeout_s: float = 20.0
    service_timeout_s: float = 5.0
    save_timeout_s: float = 15.0
    save_map_on_stop: bool = True
    stop_timeout_s: float = 15.0
    exit_on_stop: bool = True       # after X: the Program Manager (and the launch) closes
    plan_timeout_s: float = 120.0   # stage 3 on the Jetson


def describe_checks(checks):
    """Human-readable list of readiness checks."""
    parts = []
    for c in checks:
        for k, v in c.items():
            parts.append({'message': f'data on {v}', 'topic': f'publisher of {v}',
                          'clock': f'{v} advancing', 'action': f'action {v}',
                          'service': f'service {v}', 'node': f'node {v}',
                          'status': f'{v}'}.get(k, f'{k} {v}'))
    return ', '.join(parts) or 'nothing'


class MissionCore:
    """The four-stage mission state machine."""

    def __init__(self, steps, ports, settings=None, mode='sim', start_folder=None):
        """Create the state machine in IDLE.

        Args:
            steps: List of Step, in start-up order (all stages).
            ports: Object with the methods listed in program_manager.py.
            settings: MissionSettings.
            mode: 'sim' or 'real' (informational).
            start_folder: A saved mission folder to start from (map:=). Stage 1 is
                skipped: Planisferio loads that map and waits for the robot's pose.
        """
        for s in steps:
            if s.stage not in STAGES:
                raise ValueError(f'step {s.name}: stage must be one of {STAGES}')
        self.all_steps = list(steps)
        self.loaded_map = bool(start_folder)
        self.mission_folder = start_folder or None
        self.p = ports
        self.s = settings or MissionSettings()
        self.mode = mode
        self.started_stages = {'core'} | (set() if self.loaded_map else {'mapping'})
        self.retired = set()            # steps closed on purpose (Frontera after stage 1)
        self.done_stages = set()        # 'mapping', 'quality', 'route', 'restocking'
        self.skipped_stages = set()
        if self.loaded_map:
            self.done_stages.add('mapping')
        self.tour = None                # map_tour.MapTour during stage 2
        self.restock = None             # restock_core.RestockCore during stage 4
        self.route_info = ''
        self.was_localized = False      # map:= run: Planisferio got the robot's pose
        self.state = IDLE
        self.detail = 'Press S to bring the system up.'
        self.exit_requested = False     # X finished and exit_on_stop: close the manager
        self.state_since = self.p.now()
        self.started_at = self.p.now()
        self.end_class = ''
        self.last_map = None            # {'ok', 'message', 'time', 'reason', 'raw'}
        self.events = collections.deque(maxlen=60)
        self._op = {}                   # data of the composite operation in progress
        self._save = None               # map save in progress
        self._event(f'Program Manager ready ({mode} mode'
                    + (f', map {start_folder}' if start_folder else '')
                    + '). Press S to bring the system up.')

    # ── steps ────────────────────────────────────────────────────────────
    @property
    def steps(self):
        """Steps of the stages started so far (minus the ones closed on purpose)."""
        return [s for s in self.all_steps
                if s.stage in self.started_stages and s.name not in self.retired]

    def _stage_steps(self, stage):
        return [s for s in self.all_steps if s.stage == stage and s.name not in self.retired]

    # =====================================================================
    # operator commands
    # =====================================================================
    def allowed(self):
        """Commands accepted in the current state."""
        allowed = list(ALLOWED[self.state])
        if SAVE_MAP in allowed and (not self._slam_alive() or 'quality' in self.done_stages
                                    or self.state in (ROUTE_READY,)):
            allowed.remove(SAVE_MAP)
        if RESET in allowed and ('mapping' not in self.started_stages or self.retired
                                 or self.loaded_map):
            allowed.remove(RESET)
        return allowed

    def command(self, text):
        """Handle an operator command.

        Args:
            text: A key (S, P, F, K, M, N, X) or a word (START, PAUSE, ...).

        Returns:
            (accepted, message).
        """
        word = str(text).strip().upper()
        cmd = ALIASES.get(word)
        if cmd is None:
            return self._reply(False, f'unknown command "{text}"')
        if cmd not in self.allowed():
            return self._reply(False, f'{cmd} is not possible in {self.state}')
        st = self.state
        if cmd == START:
            if st in (IDLE, STOPPED, ERROR):
                self._begin_bringup(f'{word}: bringing the system up')
            elif st == READY:
                self._start_exploring('START')
            elif st == PAUSED:
                self._start_exploring('RESUME')
            elif st == MAPPING_DONE:
                self._begin_quality()
            elif st == QUALITY_PAUSED:
                self.p.send_navigation('RESUME')
                self.tour.resume()
                self._set(QUALITY, 'Map tour resumed.')
            elif st == QUALITY_DONE:
                self._begin_planning()
            elif st == ROUTE_READY:
                self._begin_restocking()
            elif st == RESTOCK_PAUSED:
                self.p.send_navigation('RESUME')
                self.restock.resume()
                self._set(RESTOCKING, 'Restocking resumed.')
        elif cmd == SKIP:
            self.skipped_stages.add('quality')
            self._event('Stage 2 (map quality) skipped')
            self._begin_planning()
        elif cmd == PAUSE:
            if st == RESTOCKING:
                self.restock.pause()
                self._halt(explorer='PAUSE', navigation='CANCEL')
                self._set(RESTOCK_PAUSED, 'Restocking paused: the robot holds (a forklift '
                          'action was canceled). S resumes.')
            elif st == QUALITY:
                self.tour.pause()
                self._halt(explorer='PAUSE', navigation='CANCEL')
                self._set(QUALITY_PAUSED, 'Map tour paused: the robot holds. S resumes, '
                          'F finishes the tour and saves the map.')
            else:
                self._halt(explorer='PAUSE', navigation='PAUSE')
                self._set(PAUSED, 'Paused: the robot holds. S resumes, F finishes mapping.')
        elif cmd == FINISH:
            if st in (QUALITY, QUALITY_PAUSED):
                self.tour.finish_early()
                self._begin_finish('FINISHED_BY_OPERATOR', 'quality')
            else:
                self._begin_finish('FINISHED_BY_OPERATOR', 'mapping')
        elif cmd == SAVE_MAP:
            if not self._request_save('operator'):
                return self._reply(False, 'a map save is already running')
        elif cmd == RESET:
            self._begin_reset()
        elif cmd == STOP:
            self._begin_stop('operator')
        return self._reply(True, f'{cmd} accepted')

    def _reply(self, ok, message):
        """Record the outcome of a command and return it."""
        self._event(('' if ok else 'rejected: ') + message)
        return ok, message

    # =====================================================================
    # inputs from other scripts
    # =====================================================================
    def on_map_save_request(self, source='frontera'):
        """Frontera asked for a map save (its SAVE_MAP or auto-save)."""
        if self._slam_alive() and self.state in (EXPLORING, PAUSED):
            self._request_save(source)

    # =====================================================================
    # tick
    # =====================================================================
    def tick(self):
        """Advance the state machine; call it at about 10 Hz."""
        now = self.p.now()
        self._poll_save(now)
        handler = getattr(self, '_tick_' + self.state.lower(), None)
        if handler is not None:
            handler(now)
        if self.state in SUPERVISED:
            self._check_health()

    # =====================================================================
    # bring-up (also used to start the scripts of a later stage)
    # =====================================================================
    def _begin_bringup(self, why):
        """Start (or verify) every step of the stages started so far."""
        self._op = {'i': 0, 'phase': 'check', 't0': self.p.now(), 'steps': self.steps,
                    'then': self._after_bringup}
        self._set(BRINGUP, 'Bringing the system up...')
        self._event(why)
        self.p.prepare_bringup()

    def _begin_stage_steps(self, steps, then, text):
        self._op = {'i': 0, 'phase': 'check', 't0': self.p.now(), 'steps': list(steps),
                    'then': then}
        self._set(STAGE_BRINGUP, text)

    def _tick_stage_bringup(self, now):
        self._tick_bringup(now)

    def _tick_bringup(self, now):
        op = self._op
        steps = op['steps']
        if op['i'] >= len(steps):
            op['then']()
            return
        step = steps[op['i']]
        if op['phase'] == 'check':
            if self.p.alive(step) and self.p.ready(step):
                op['i'] += 1
                return
            if not step.external and not self.p.alive(step):
                ok, msg = self.p.launch(step)
                if not ok:
                    self._step_failed(step, f'cannot start {step.name}: {msg}')
                    return
                self._event(f'started {step.name}')
                # Start the steps that go together with this one right away.
                j = op['i'] + 1
                while j < len(steps) and steps[j].with_previous:
                    nxt = steps[j]
                    if not nxt.external and not self.p.alive(nxt):
                        ok, msg = self.p.launch(nxt)
                        if ok:
                            self._event(f'started {nxt.name} (in parallel)')
                    j += 1
            op['phase'], op['t0'] = 'wait', now
            self.detail = f'{step.name}: waiting for {describe_checks(step.ready)}'
            return
        if self.p.ready(step):
            self._event(f'{step.name} ready ({now - op["t0"]:.1f} s)')
            op['i'] += 1
            op['phase'] = 'check'
            return
        if not step.external and not self.p.alive(step):
            self._step_failed(step, f'{step.name} exited during start-up '
                              f'(code {self.p.exit_code(step)}); see '
                              f'{self.p.log_path(step)}', show_log=True)
            return
        if now - op['t0'] > step.timeout:
            hint = f' {step.hint}' if step.hint else ''
            self._step_failed(step, f'{step.name}: no {describe_checks(step.ready)} '
                              f'after {step.timeout:.0f} s.{hint}', show_log=True)

    def _step_failed(self, step, reason, show_log=False):
        """A step did not come up: stop (critical) or warn and go on (optional)."""
        if show_log and not step.external:
            for line in self.p.log_tail(step, 4):
                self._event(f'  {step.name}| {line}')
        if step.critical:
            self._error(reason)
            return
        self._event(f'WARNING (optional step, continuing): {reason}')
        self._op['i'] += 1
        self._op['phase'] = 'check'

    def _after_bringup(self):
        """All scripts up: go where the run is (first start, or after an error)."""
        self._event('All scripts ready.')
        if self.restock is not None and not self.restock.done:
            self._set(RESTOCK_PAUSED, 'Scripts ready again. S resumes restocking.')
        elif 'route' in self.done_stages:
            self._set(ROUTE_READY, f'Route ready: {self.route_info}. S starts restocking.')
        elif self.tour is not None and not self.tour.done:
            self._set(QUALITY_PAUSED, 'Scripts ready again. S resumes the map tour.')
        elif 'quality' in self.done_stages:
            self._set(QUALITY_DONE, 'Scripts ready again. S plans the restocking route.')
        elif 'mapping' in self.done_stages and not self.loaded_map:
            self._set(MAPPING_DONE, 'Scripts ready again. S improves the map, K skips to '
                      'the route.')
        elif self.loaded_map and not self.was_localized:
            self._set(LOCALIZING, 'Map loaded. Give the robot its pose on the map: RViz '
                      '"2D Pose Estimate" (/initialpose).')
        elif self.loaded_map:
            self._set(MAPPING_DONE, 'Map loaded and localized. S improves the map, K skips '
                      'to the route.')
        else:
            self._set(READY, 'System ready. The robot is still. Press S to start mapping.')

    # ── loaded map: wait for the pose ────────────────────────────────────
    def _tick_localizing(self, now):
        st = self.p.slam_status()
        if str(st.get('localized', '0')) in ('1', 'true', 'True'):
            self.was_localized = True
            self._event('Planisferio localized on the loaded map')
            self._set(MAPPING_DONE, 'Map loaded and localized. S improves the map '
                      '(stage 2), K skips to the route (stage 3).')

    # =====================================================================
    # stage 1: exploring
    # =====================================================================
    def _start_exploring(self, how):
        """Release the robot: Sendero first, then Frontera."""
        self.p.send_navigation('RESUME')
        self.p.send_explorer('START')
        self._op = {'sent_at': self.p.now(), 'retries': 0, 'acked': False}
        self.end_class = ''
        self._set(EXPLORING, 'Stage 1: exploring. Frontera picks frontiers, Sendero drives.')
        self._event('Exploration ' + ('started' if how == 'START' else 'resumed'))

    def _tick_exploring(self, now):
        status, age = self.p.explorer_status()
        op = self._op
        if status in EXPLORATION_END or status.startswith('ERROR'):
            self._event(f'Frontera finished: {status}')
            self._begin_finish(status, 'mapping')
            return
        if status in ('WAITING_FOR_START', 'PAUSED') and not op.get('acked'):
            if now - op['sent_at'] > self.s.start_ack_timeout_s:
                if op['retries'] >= self.s.start_retries:
                    self._fault('Frontera does not accept START '
                                f'(status {status}).')
                    return
                op['retries'] += 1
                op['sent_at'] = now
                self.p.send_explorer('START')
                self._event(f'START repeated ({op["retries"]})')
        elif status:
            op['acked'] = True
        self.detail = f'Stage 1: exploring (Frontera: {status or "no status"}).'

    def _tick_paused(self, now):
        status, _ = self.p.explorer_status()
        if status in EXPLORATION_END or status.startswith('ERROR'):
            self._begin_finish(status, 'mapping')

    # =====================================================================
    # finishing stage 1 or 2: optimize, save the map, save a mission folder
    # =====================================================================
    def _begin_finish(self, end_class, stage):
        self.end_class = end_class
        self._op = {'phase': 'halt', 't0': self.p.now(), 'stage': stage}
        what = 'Mapping' if stage == 'mapping' else 'Map tour'
        self._set(FINISHING, f'{what} finished ({end_class}): optimizing the map...')

    def _tick_finishing(self, now):
        op = self._op
        if op['phase'] == 'halt':
            self._halt(explorer='STOP', navigation='CANCEL')
            self.p.send_navigation('RESUME')        # clear a pause flag
            op['handle'] = self.p.call_service('slam_optimize')
            op['phase'], op['t0'] = 'optimize', now
            return
        if op['phase'] == 'optimize':
            h = op['handle']
            if h is None or h.done() or now - op['t0'] > self.s.service_timeout_s:
                ok, msg = h.result() if (h is not None and h.done()) else (False, 'no answer')
                self._event(('Pose-graph optimization requested' if ok
                             else f'Optimization skipped: {msg}'))
                op['phase'] = 'settle' if ok else 'save'
                op['t0'] = now
            return
        if op['phase'] == 'settle':
            busy = str(self.p.slam_status().get('backend_busy', '0')) == '1'
            waited = now - op['t0']
            if (waited >= self.s.optimize_settle_min_s and not busy) \
                    or waited > self.s.optimize_timeout_s:
                if busy:
                    self._event('Optimization still running; saving anyway.')
                op['phase'] = 'save'
            return
        if op['phase'] == 'save':
            self.detail = 'Saving the map...'
            self._request_save(f'end of {op["stage"]}', force=True)
            op['phase'] = 'saving'
            return
        if op['phase'] == 'saving' and self._save is None:
            m = self.last_map or {}
            stage = op['stage']
            saved = ''
            if m.get('ok'):
                ok, info, folder = self.p.save_mission(m.get('raw', ''))
                self._event(('Mission folder saved: ' if ok else 'Mission folder NOT saved: ')
                            + info)
                if ok:
                    self.mission_folder = folder
                saved = f'Map saved: {m.get("message", "")}.'
            else:
                saved = f'Map NOT saved ({m.get("message", "?")}).'
            self.done_stages.add(stage)
            if stage == 'mapping':
                self._set(MAPPING_DONE, f'STAGE 1 DONE ({self.end_class}). {saved} '
                          'S improves the map (stage 2), K skips to the route (stage 3).')
            else:
                self._set(QUALITY_DONE, f'STAGE 2 DONE ({self.end_class}). {saved} '
                          'S plans the restocking route (stage 3).')
            self._event(f'Stage {"1" if stage == "mapping" else "2"} complete: '
                        f'{self.end_class}')

    # =====================================================================
    # stage 2: map quality
    # =====================================================================
    def _begin_quality(self):
        if not self.mission_folder:
            self._event('rejected: no mission folder saved yet (M saves the map first)')
            return
        self._retire_explorer()
        self.started_stages.add('quality')
        self._begin_stage_steps(self._stage_steps('quality'), self._start_tour,
                                'Stage 2: starting the camera, ArUco, logo detector and '
                                'landmark mapper (Frontera closed)...')
        self._event('Stage 2: map quality')

    def _start_tour(self):
        self.p.send_navigation('RESUME')
        self.tour = self.p.make_tour()
        grid = self.p.load_grid(self.mission_folder)
        if grid is None:
            self._error(f'cannot read the map in {self.mission_folder}')
            return
        self.tour.start(grid)
        self._set(QUALITY, 'Stage 2: map tour started.')

    def _tick_quality(self, now):
        t = self.tour
        t.tick()
        if t.done:
            self._begin_finish(f'TOUR: {t.outcome}', 'quality')
            return
        self.detail = 'Stage 2: ' + (t.detail or 'map tour')

    # =====================================================================
    # stage 3: the restocking route
    # =====================================================================
    def _begin_planning(self):
        self._halt(explorer='STOP', navigation='CANCEL')
        self._retire_explorer()
        self._op = {'handle': self.p.plan_route(self.mission_folder), 't0': self.p.now()}
        self._set(PLANNING, 'Stage 3: planning the restocking route on the final map...')

    def _tick_planning(self, now):
        op = self._op
        h = op['handle']
        if h is not None and not h.done() and now - op['t0'] < self.s.plan_timeout_s:
            return
        ok, msg = h.result() if (h is not None and h.done()) else (False, 'timed out')
        if not ok:
            self._error(f'route planning failed: {msg}')
            return
        self.route_info = msg
        self.done_stages.add('route')
        self._event(f'Stage 3 complete: {msg}')
        self._set(ROUTE_READY, f'STAGE 3 DONE. Route: {msg}. S starts restocking (stage 4).')

    # =====================================================================
    # stage 4: restocking
    # =====================================================================
    def _begin_restocking(self):
        self._retire_explorer()
        steps = []
        if 'quality' not in self.started_stages:       # stage 2 skipped: camera now
            self.started_stages.add('quality')
            steps += self._stage_steps('quality')
        self.started_stages.add('restocking')
        steps += self._stage_steps('restocking')
        self._begin_stage_steps(steps, self._start_restock,
                                'Stage 4: starting the EKF, box detectors, QR reader and '
                                'forklift...')
        self._event('Stage 4: restocking')

    def _start_restock(self):
        h = self.p.call_service('slam_localization_only')
        self._event('Planisferio: localization only (the map no longer changes)' if h
                    else 'WARNING: Planisferio has no localization_only service; it keeps '
                    'mapping')
        self.p.send_navigation('RESUME')
        self.restock = self.p.make_restock('full', self.mission_folder)
        self.restock.start()
        self._set(RESTOCKING, 'Stage 4: restocking...')

    def _tick_restocking(self, now):
        r = self.restock
        r.tick()
        if r.done:
            self.done_stages.add('restocking')
            self._set(RESTOCK_DONE, f'STAGE 4 DONE: {r.outcome}. X closes everything.')
            self.p.zero_velocity()
            return
        if r.error:
            self._halt(explorer='PAUSE', navigation='CANCEL')
            self._set(RESTOCK_PAUSED, f'{r.error} — S retries, X stops.')
            return
        self.detail = f'Stage 4 [{r.state} / {r.phase}]: {r.detail}'

    def _retire_explorer(self):
        """Close Frontera for good (stages 2-4 do not explore)."""
        explorers = [s for s in self.all_steps if s.role == 'explorer' and not s.external
                     and s.name not in self.retired]
        if not explorers:
            return
        self._halt(explorer='STOP', navigation='CANCEL')
        self.retired |= {s.name for s in explorers}
        self.p.stop_steps(explorers)
        self._event('Frontera closed (no more exploring)')

    # =====================================================================
    # reset (stage 1)
    # =====================================================================
    def _begin_reset(self):
        """New mapping run without closing SLAM, odometry or Sendero."""
        self._op = {'phase': 'halt', 't0': self.p.now()}
        self._set(RESETTING, 'Resetting: stopping the robot...')

    def _tick_resetting(self, now):
        op = self._op
        if op['phase'] == 'halt':
            self._halt(explorer='STOP', navigation='CANCEL')
            self.p.send_navigation('RESUME')
            self.p.send_navigation('CLEAR_COSTMAPS')
            explorers = [s for s in self.steps if s.role == 'explorer' and not s.external]
            op['job'] = self.p.stop_steps(explorers)
            op['phase'], op['t0'] = 'kill_explorer', now
            self.detail = 'Resetting: closing Frontera...'
            return
        if op['phase'] == 'kill_explorer':
            if op['job'].done() or now - op['t0'] > self.s.stop_timeout_s:
                op['handle'] = self.p.call_service('slam_reset')
                op['phase'], op['t0'] = 'slam_reset', now
                self.detail = 'Resetting: clearing the SLAM map...'
            return
        if op['phase'] == 'slam_reset':
            h = op['handle']
            if h is None or h.done() or now - op['t0'] > self.s.service_timeout_s:
                ok, msg = h.result() if (h is not None and h.done()) else (False, 'no answer')
                self._event('SLAM map cleared' if ok else f'SLAM reset failed: {msg}')
                self.end_class = ''
                self.done_stages.discard('mapping')
                self._begin_bringup('Reset: restarting Frontera and checking every script')

    # =====================================================================
    # stop
    # =====================================================================
    def _begin_stop(self, why):
        """Halt the robot, save the map, close every script."""
        if self.restock is not None:
            self.restock.abort(f'X ({why})')
        if self.tour is not None and not self.tour.done:
            self.tour.pause()
        self._op = {'phase': 'halt', 't0': self.p.now(), 'why': why}
        self._set(STOPPING, 'Stopping: the robot is halting...')

    def _tick_stopping(self, now):
        op = self._op
        if op['phase'] == 'halt':
            self._halt(explorer='STOP', navigation='CANCEL')
            map_changing = 'restocking' not in self.started_stages
            if (self.s.save_map_on_stop and self._slam_alive() and self._save is None
                    and map_changing):
                self._request_save('stop', force=True)
                op['phase'] = 'saving'
                self.detail = 'Stopping: saving the map...'
            else:
                op['phase'] = 'kill'
            op['t0'] = now
            return
        if op['phase'] == 'saving':
            if self._save is None or now - op['t0'] > self.s.save_timeout_s:
                op['phase'] = 'kill'
            return
        if op['phase'] == 'kill':
            launched = [s for s in self.all_steps if not s.external]
            op['job'] = self.p.stop_steps(list(reversed(launched)))
            op['phase'], op['t0'] = 'killing', now
            self.detail = 'Stopping: closing every script...'
            return
        if op['phase'] == 'killing':
            if op['job'].done() or now - op['t0'] > self.s.stop_timeout_s + 10.0:
                self.p.zero_velocity()
                if self.s.exit_on_stop:
                    self._set(STOPPED, 'Stopped: every script is closed. '
                              'Closing the Program Manager...')
                    self.exit_requested = True
                else:
                    self._set(STOPPED, 'Stopped: every script is closed. '
                              'Press S to bring the system up again.')
                self._event('All scripts closed.')

    # =====================================================================
    # errors
    # =====================================================================
    def _error(self, reason):
        """Halt and wait for the operator."""
        if self.restock is not None and not self.restock.done:
            self.restock.pause()
        if self.tour is not None and not self.tour.done:
            self.tour.pause()
        self._halt(explorer='PAUSE', navigation='CANCEL')
        self._set(ERROR, f'{reason} — S retries, X stops.')
        self._event('ERROR: ' + reason)

    def _fault(self, reason):
        """Runtime failure: halt the robot immediately."""
        self._error(reason)

    def _check_health(self):
        """A critical script that died halts the robot."""
        for step in self.steps:
            if step.external or not step.critical:
                continue
            if self.p.was_launched(step) and not self.p.alive(step):
                self._fault(f'{step.name} stopped (exit code {self.p.exit_code(step)}); '
                            f'see {self.p.log_path(step)}')
                return

    # =====================================================================
    # map saving
    # =====================================================================
    def _request_save(self, reason, force=False):
        """Ask SLAM to save the map (asynchronous)."""
        if self._save is not None:
            return force
        handle = self.p.call_service('slam_save')
        self._save = {'handle': handle, 't0': self.p.now(), 'reason': reason}
        self.p.publish_map_save_status('Saving map...')
        self._event(f'Saving the map ({reason})')
        return True

    def _poll_save(self, now):
        sv = self._save
        if sv is None:
            return
        h = sv['handle']
        if h is not None and not h.done() and now - sv['t0'] < self.s.save_timeout_s:
            return
        ok, msg = h.result() if (h is not None and h.done()) else (False, 'SLAM did not answer')
        path = msg.replace('Saved ', '').split(' and ')[0].rsplit('.', 1)[0] if ok else msg
        self.last_map = {'ok': ok, 'message': path, 'time': time.strftime('%H:%M:%S'),
                         'reason': sv['reason'], 'raw': msg}
        self._save = None
        self.p.publish_map_save_status(('Map saved: ' if ok else 'Map save failed: ') + path)
        self._event(('Map saved: ' if ok else 'Map save failed: ') + path)

    # =====================================================================
    # helpers
    # =====================================================================
    def _halt(self, explorer, navigation):
        """Stop robot motion: Frontera, Sendero and a zero velocity."""
        if self._explorer_alive():
            self.p.send_explorer(explorer)
        self.p.send_navigation(navigation)
        self.p.zero_velocity()

    def _explorer_alive(self):
        return any(self.p.alive(s) for s in self.steps if s.role == 'explorer')

    def _slam_alive(self):
        return any(self.p.alive(s) for s in self.steps if s.role == 'slam')

    def _set(self, state, detail):
        """Change state."""
        if state != self.state:
            self._event(f'{self.state} -> {state}')
            self.state = state
            self.state_since = self.p.now()
        self.detail = detail
        self.p.on_state(state)

    def _event(self, text):
        """Remember an event for the monitor and the log."""
        self.events.append((time.strftime('%H:%M:%S'), text))
        self.p.log(text)

    def stage_progress(self):
        """[(stage, title, 'done' | 'skipped' | 'active' | 'todo')] for the monitor."""
        active = {EXPLORING: 'mapping', PAUSED: 'mapping', READY: 'mapping',
                  QUALITY: 'quality', QUALITY_PAUSED: 'quality', PLANNING: 'route',
                  RESTOCKING: 'restocking', RESTOCK_PAUSED: 'restocking'}.get(self.state)
        if self.state == FINISHING:
            active = self._op.get('stage')
        if self.state == STAGE_BRINGUP:
            active = 'restocking' if 'restocking' in self.started_stages else 'quality'
        out = []
        for stage, title in STAGE_TITLES:
            if stage in self.done_stages:
                mark = 'done'
            elif stage in self.skipped_stages:
                mark = 'skipped'
            elif stage == active:
                mark = 'active'
            else:
                mark = 'todo'
            out.append((stage, title, mark))
        return out

    def snapshot(self):
        """Everything the status topic needs."""
        now = self.p.now()
        return {
            'mode': self.mode,
            'stages': self.stage_progress(),
            'loaded_map': self.loaded_map,
            'mission_folder': self.mission_folder,
            'route': self.route_info,
            'tour': self.tour.snapshot() if self.tour is not None else None,
            'restock': self.restock.snapshot() if self.restock is not None else None,
            'state': self.state,
            'detail': self.detail,
            'allowed': self.allowed(),
            'start_meaning': START_MEANING.get(self.state, ''),
            'skip_meaning': SKIP_MEANING.get(self.state, ''),
            'state_age_s': round(now - self.state_since, 1),
            'uptime_s': round(now - self.started_at, 1),
            'end_class': self.end_class,
            'last_map': self.last_map,
            'saving_map': self._save is not None,
            'exiting': self.exit_requested,
            'events': list(self.events)[-10:],
        }
