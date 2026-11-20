"""
mission_core.py — the mission state machine (no ROS)
=====================================================
The Program Manager decides; other scripts act. This module is the decision
part: a state machine driven by operator commands and by what the other
scripts report. It never computes a path, a map or a velocity. It starts and
stops scripts, sends them short commands and waits for their answers through
a `ports` object (implemented by the ROS node in program_manager.py, and by
a fake in the tests).

Phase 1 — autonomous mapping (implemented)

  IDLE ──S──> BRINGUP ──all scripts ready──> READY ──S──> EXPLORING
                 │                              ^            │  ^
              failure                           │          P │  │ S
                 v                              │            v  │
               ERROR <── a script died ─────────┼──────── PAUSED
                                                │            │
       EXPLORING / PAUSED ──F, or Frontera ends─┴──> FINISHING
                        (stop exploring, optimize the pose graph, save the map)
                                                         │
                                                         v
                                                    MAPPING_DONE ──R──> RESTOCKING
                                                                     (phase 2: stub)

  N (reset) from READY / EXPLORING / PAUSED / MAPPING_DONE / RESTOCKING / ERROR:
      stop the robot, restart Frontera, clear SLAM and Sendero -> BRINGUP -> READY
  X (stop) from any active state:
      stop the robot, save the map, close every script -> STOPPED (S brings it up again)
  M (save map): whenever SLAM is running.

Safety rules kept from the CALL-M program manager:
  * bringing the system up never moves the robot; exploration starts only on
    an explicit S in READY;
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
READY = 'READY'
EXPLORING = 'EXPLORING'
PAUSED = 'PAUSED'
FINISHING = 'FINISHING'
MAPPING_DONE = 'MAPPING_DONE'
RESTOCKING = 'RESTOCKING'
RESETTING = 'RESETTING'
STOPPING = 'STOPPING'
STOPPED = 'STOPPED'
ERROR = 'ERROR'

START = 'START'
PAUSE = 'PAUSE'
FINISH = 'FINISH'
SAVE_MAP = 'SAVE_MAP'
RESTOCK = 'RESTOCK'
RESET = 'RESET'
STOP = 'STOP'

# Keys / words accepted on the command topic.
ALIASES = {
    'S': START, 'START': START, 'C': START, 'CONFIRM': START, 'RESUME': START,
    'P': PAUSE, 'PAUSE': PAUSE,
    'F': FINISH, 'FINISH': FINISH, 'FINISH_MAPPING': FINISH,
    'M': SAVE_MAP, 'SAVE_MAP': SAVE_MAP, 'SAVE': SAVE_MAP,
    'R': RESTOCK, 'RESTOCK': RESTOCK, 'START_RESTOCKING': RESTOCK,
    'N': RESET, 'RESET': RESET,
    'X': STOP, 'STOP': STOP,
}

ALLOWED = {
    IDLE: [START],
    BRINGUP: [STOP],
    READY: [START, SAVE_MAP, RESET, STOP],
    EXPLORING: [PAUSE, FINISH, SAVE_MAP, RESET, STOP],
    PAUSED: [START, FINISH, SAVE_MAP, RESET, STOP],
    FINISHING: [STOP],
    MAPPING_DONE: [RESTOCK, SAVE_MAP, RESET, STOP],
    RESTOCKING: [SAVE_MAP, RESET, STOP],
    RESETTING: [STOP],
    STOPPING: [],
    STOPPED: [START],
    ERROR: [START, SAVE_MAP, RESET, STOP],
}

# What START does in each state (shown by the monitor next to the S key).
START_MEANING = {IDLE: 'bring up', STOPPED: 'bring up', READY: 'start exploring',
                 PAUSED: 'resume', ERROR: 'retry bring-up'}

# Frontera end states (its /exploration/status once a run is over).
EXPLORATION_END = ('COMPLETE', 'COMPLETE_PARTIAL', 'STAGNATED', 'TIME_LIMIT')

# States in which the launched scripts must stay alive.
SUPERVISED = (READY, EXPLORING, PAUSED, FINISHING, MAPPING_DONE, RESTOCKING)


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
    """
    name: str
    cmd: str = ''
    ready: list = field(default_factory=list)
    timeout: float = 60.0
    critical: bool = True
    role: str = ''
    hint: str = ''

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
    """Phase-1 mission state machine (autonomous mapping)."""

    def __init__(self, steps, ports, settings=None, mode='sim'):
        """Create the state machine in IDLE.

        Args:
            steps: List of Step, in start-up order.
            ports: Object with the methods listed in the module docstring of
                program_manager.py (launch, alive, ready, ...).
            settings: MissionSettings.
            mode: 'sim' or 'real' (informational).
        """
        self.steps = list(steps)
        self.p = ports
        self.s = settings or MissionSettings()
        self.mode = mode
        self.state = IDLE
        self.detail = 'Press S to bring the system up.'
        self.exit_requested = False     # X finished and exit_on_stop: close the manager
        self.state_since = self.p.now()
        self.started_at = self.p.now()
        self.end_class = ''
        self.last_map = None            # {'ok', 'message', 'time', 'reason'}
        self.events = collections.deque(maxlen=60)
        self._op = {}                   # data of the composite operation in progress
        self._save = None               # map save in progress
        self._event(f'Program Manager ready ({mode} mode). Press S to bring the system up.')

    # =====================================================================
    # operator commands
    # =====================================================================
    def allowed(self):
        """Commands accepted in the current state."""
        allowed = list(ALLOWED[self.state])
        if SAVE_MAP in allowed and not self._slam_alive():
            allowed.remove(SAVE_MAP)
        return allowed

    def command(self, text):
        """Handle an operator command.

        Args:
            text: A key (S, P, F, M, R, N, X) or a word (START, PAUSE, ...).

        Returns:
            (accepted, message).
        """
        word = str(text).strip().upper()
        cmd = ALIASES.get(word)
        if cmd is None:
            return self._reply(False, f'unknown command "{text}"')
        if cmd not in self.allowed():
            return self._reply(False, f'{cmd} is not possible in {self.state}')
        if cmd == START:
            if self.state in (IDLE, STOPPED, ERROR):
                self._begin_bringup(f'{word}: bringing the system up')
            elif self.state == READY:
                self._start_exploring('START')
            elif self.state == PAUSED:
                self._start_exploring('RESUME')
        elif cmd == PAUSE:
            self._halt(explorer='PAUSE', navigation='PAUSE')
            self._set(PAUSED, 'Paused: the robot holds. S resumes, F finishes mapping.')
        elif cmd == FINISH:
            self._begin_finish('FINISHED_BY_OPERATOR')
        elif cmd == SAVE_MAP:
            if not self._request_save('operator'):
                return self._reply(False, 'a map save is already running')
        elif cmd == RESTOCK:
            self._set(RESTOCKING, 'Restocking (phase 2) is not implemented yet. '
                      'N starts a new mapping run, X stops.')
            self._event('Phase 2 requested: restocking is a placeholder for now.')
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
        if self._slam_alive():
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

    # ── bring-up ─────────────────────────────────────────────────────────
    def _begin_bringup(self, why):
        """Start (or verify) every step in order."""
        self._op = {'i': 0, 'phase': 'check', 't0': self.p.now()}
        self._set(BRINGUP, 'Bringing the system up...')
        self._event(why)
        self.p.prepare_bringup()

    def _tick_bringup(self, now):
        op = self._op
        if op['i'] >= len(self.steps):
            self._set(READY, 'System ready. The robot is still. '
                      'Press S to start exploring.')
            self._event('All scripts ready.')
            return
        step = self.steps[op['i']]
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

    # ── exploring ────────────────────────────────────────────────────────
    def _start_exploring(self, how):
        """Release the robot: Sendero first, then Frontera."""
        self.p.send_navigation('RESUME')
        self.p.send_explorer('START')
        self._op = {'sent_at': self.p.now(), 'retries': 0, 'acked': False}
        self.end_class = ''
        self._set(EXPLORING, 'Exploring: Frontera picks frontiers, Sendero drives.')
        self._event('Exploration ' + ('started' if how == 'START' else 'resumed'))

    def _tick_exploring(self, now):
        status, age = self.p.explorer_status()
        op = self._op
        if status in EXPLORATION_END or status.startswith('ERROR'):
            self._event(f'Frontera finished: {status}')
            self._begin_finish(status)
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
        self.detail = f'Exploring (Frontera: {status or "no status"}).'

    def _tick_paused(self, now):
        status, _ = self.p.explorer_status()
        if status in EXPLORATION_END or status.startswith('ERROR'):
            self._begin_finish(status)

    # ── finishing ────────────────────────────────────────────────────────
    def _begin_finish(self, end_class):
        """End exploration, optimize the pose graph and save the map."""
        self.end_class = end_class
        self._op = {'phase': 'halt', 't0': self.p.now()}
        self._set(FINISHING, f'Mapping finished ({end_class}): optimizing the map...')

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
            self.detail = f'Mapping finished ({self.end_class}): saving the map...'
            self._request_save('end of mapping', force=True)
            op['phase'] = 'saving'
            return
        if op['phase'] == 'saving' and self._save is None:
            m = self.last_map or {}
            if m.get('ok'):
                text = (f'EXPLORATION DONE ({self.end_class}). Map saved: '
                        f'{m.get("message", "")}. Press R for restocking.')
            else:
                text = (f'EXPLORATION DONE ({self.end_class}). Map NOT saved '
                        f'({m.get("message", "?")}); press M to retry. '
                        'R starts restocking.')
            self._set(MAPPING_DONE, text)
            self._event(f'Mapping phase complete: {self.end_class}')

    # ── reset ────────────────────────────────────────────────────────────
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
                self._begin_bringup('Reset: restarting Frontera and checking every script')

    # ── stop ─────────────────────────────────────────────────────────────
    def _begin_stop(self, why):
        """Halt the robot, save the map, close every script."""
        self._op = {'phase': 'halt', 't0': self.p.now(), 'why': why}
        self._set(STOPPING, 'Stopping: the robot is halting...')

    def _tick_stopping(self, now):
        op = self._op
        if op['phase'] == 'halt':
            self._halt(explorer='STOP', navigation='CANCEL')
            if self.s.save_map_on_stop and self._slam_alive() and self._save is None:
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
            launched = [s for s in self.steps if not s.external]
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

    # ── errors ───────────────────────────────────────────────────────────
    def _error(self, reason):
        """Bring-up failure: halt and wait for the operator."""
        self._halt(explorer='PAUSE', navigation='CANCEL')
        self._set(ERROR, f'{reason} — S retries, N resets, X stops.')
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
                         'reason': sv['reason']}
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

    def snapshot(self):
        """Everything the status topic needs."""
        now = self.p.now()
        return {
            'mode': self.mode,
            'state': self.state,
            'detail': self.detail,
            'allowed': self.allowed(),
            'start_meaning': START_MEANING.get(self.state, ''),
            'state_age_s': round(now - self.state_since, 1),
            'uptime_s': round(now - self.started_at, 1),
            'end_class': self.end_class,
            'last_map': self.last_map,
            'saving_map': self._save is not None,
            'exiting': self.exit_requested,
            'events': list(self.events)[-10:],
        }
