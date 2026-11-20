"""
process_supervisor.py — start, watch and stop the robot's scripts (no ROS)
==========================================================================
Every script runs as its own process group (so `ros2 launch` and everything
it spawns stop together). Output goes to

    <log_dir>/<name>.txt    one file per script
    <log_dir>/all.txt       everything, each line prefixed with [name]

Stopping sends SIGINT (ROS 2 nodes shut down cleanly on it), then SIGTERM,
then SIGKILL, each after a grace period.
"""
# Author: Raymundo Gomez

import os
import re
import signal
import subprocess
import threading
import time


def main_ros_logs_dir():
    """main_ros/logs in the source tree (created if needed).

    With colcon --symlink-install the installed files are links into the
    source tree, so following them leads to main_ros/ (the folder with
    package.xml). Falls back to ~/main_ros_logs if that is not possible.
    """
    candidates = [os.path.realpath(__file__)]
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(os.path.realpath(os.path.join(
            get_package_share_directory('main_ros'), 'package.xml')))
    except Exception:
        pass
    for path in candidates:
        d = os.path.dirname(path)
        for _ in range(5):
            if os.path.isfile(os.path.join(d, 'package.xml')) and \
                    os.path.isfile(os.path.join(d, 'setup.py')):
                logs = os.path.join(d, 'logs')
                try:
                    os.makedirs(logs, exist_ok=True)
                    return logs
                except OSError:
                    break
            d = os.path.dirname(d)
    logs = os.path.expanduser('~/main_ros_logs')
    os.makedirs(logs, exist_ok=True)
    return logs


def run_log_dir(prefix, root=''):
    """Create <root or main_ros/logs>/<prefix>_<date>_<time>/ and return it."""
    root = os.path.expanduser(root) if root else main_ros_logs_dir()
    d = os.path.join(root, f'{prefix}_{time.strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(d, exist_ok=True)
    return d


class _Job:
    """Completion flag of a background stop operation."""

    def __init__(self):
        self._event = threading.Event()

    def done(self):
        return self._event.is_set()

    def set(self):
        self._event.set()


class ManagedProcess:
    """One launched script."""

    def __init__(self, name, cmd, proc, log_path):
        self.name = name
        self.cmd = cmd
        self.proc = proc
        self.log_path = log_path
        self.started_at = time.time()
        self.stopping = False

    @property
    def pid(self):
        return self.proc.pid

    def alive(self):
        return self.proc.poll() is None

    def exit_code(self):
        return self.proc.poll()


class ProcessSupervisor:
    """Launches scripts and keeps track of them."""

    def __init__(self, log_dir, env=None, logger=None):
        """Create the supervisor.

        Args:
            log_dir: Directory for this session's logs (created).
            env: Environment for the scripts (default: this process's).
            logger: Callable(text) for supervisor messages.
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.env = dict(env if env is not None else os.environ)
        self.procs = {}
        self.starts = {}
        self._lock = threading.Lock()
        self._log = logger or (lambda text: None)
        self._all = open(os.path.join(log_dir, 'all.txt'), 'a', buffering=1)

    # ── start ────────────────────────────────────────────────────────────
    def start(self, name, cmd):
        """Start a script in its own process group.

        Args:
            name: Script name (log file name).
            cmd: Shell command.

        Returns:
            (ok, message).
        """
        if not cmd or not cmd.strip():
            return False, 'no command configured'
        with self._lock:
            old = self.procs.get(name)
            if old is not None and old.alive():
                return True, f'already running (pid {old.pid})'
            log_path = os.path.join(self.log_dir, f'{name}.txt')
            fh = open(log_path, 'a', buffering=1)
            fh.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")} $ {cmd}\n')
            try:
                proc = subprocess.Popen(
                    ['bash', '-c', cmd], env=self.env, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                    errors='replace')
            except OSError as exc:
                fh.close()
                return False, str(exc)
            mp = ManagedProcess(name, cmd, proc, log_path)
            self.procs[name] = mp
            self.starts[name] = self.starts.get(name, 0) + 1
        threading.Thread(target=self._pump, args=(mp, fh), daemon=True,
                         name=f'log_{name}').start()
        self._log(f'started {name} (pid {proc.pid}): {cmd}')
        return True, f'pid {proc.pid}'

    def _pump(self, mp, fh):
        """Copy a script's output to its log file and to all.txt."""
        prefix = f'[{mp.name}] '
        try:
            for line in mp.proc.stdout:
                fh.write(line)
                try:
                    self._all.write(prefix + line)
                except ValueError:      # all.txt closed at shutdown
                    pass
        except (ValueError, OSError):
            pass
        finally:
            code = mp.proc.wait()
            fh.write(f'===== exited with code {code}\n')
            fh.close()

    # ── queries ──────────────────────────────────────────────────────────
    def get(self, name):
        return self.procs.get(name)

    def alive(self, name):
        mp = self.procs.get(name)
        return mp is not None and mp.alive()

    def exit_code(self, name):
        mp = self.procs.get(name)
        return None if mp is None else mp.exit_code()

    def was_launched(self, name):
        """Launched and not deliberately stopped since."""
        mp = self.procs.get(name)
        return mp is not None and not mp.stopping

    def log_path(self, name):
        mp = self.procs.get(name)
        return mp.log_path if mp is not None else os.path.join(self.log_dir, f'{name}.txt')

    # ── stop ─────────────────────────────────────────────────────────────
    def stop(self, name, sigint_wait=5.0, sigterm_wait=3.0):
        """Stop one script (blocking): SIGINT, then SIGTERM, then SIGKILL."""
        mp = self.procs.get(name)
        if mp is None:
            return
        mp.stopping = True
        if not mp.alive():
            return
        for sig, wait in ((signal.SIGINT, sigint_wait), (signal.SIGTERM, sigterm_wait),
                          (signal.SIGKILL, 2.0)):
            try:
                os.killpg(mp.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                mp.proc.wait(timeout=wait)
                self._log(f'stopped {name} ({sig.name})')
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            self._log(f'WARNING: {name} (pid {mp.pid}) did not stop')
        self._reap_group(mp.pid, name)

    def _reap_group(self, pgid, name):
        """Terminate children left in the process group after its leader exited."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                return
            deadline = time.time() + 2.0
            while time.time() < deadline:
                try:
                    os.killpg(pgid, 0)
                except (ProcessLookupError, PermissionError):
                    return
                time.sleep(0.05)
        self._log(f'WARNING: processes of {name} (group {pgid}) survived SIGKILL')

    def stop_many_async(self, names):
        """Stop scripts one after another in a background thread.

        Returns:
            A job whose done() tells when all are stopped.
        """
        job = _Job()

        def run():
            try:
                for n in names:
                    self.stop(n)
            finally:
                job.set()
        threading.Thread(target=run, daemon=True, name='stopper').start()
        return job

    def stop_all(self):
        """Stop every script (blocking), newest first."""
        for name in reversed(list(self.procs)):
            self.stop(name)

    # ── stale processes ──────────────────────────────────────────────────
    def kill_stale(self, patterns):
        """Kill leftovers of a previous run that this supervisor did not start.

        Args:
            patterns: Regular expressions matched against full command lines.

        Returns:
            List of (pid, command line) that were killed.
        """
        mine = {mp.pid for mp in self.procs.values()}
        me = os.getpid()
        regexes = [re.compile(p) for p in patterns if p]
        killed = []
        if not regexes:
            return killed
        for pid_s in os.listdir('/proc'):
            if not pid_s.isdigit():
                continue
            pid = int(pid_s)
            if pid in mine or pid == me:
                continue
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as f:
                    cmdline = f.read().replace(b'\x00', b' ').decode(errors='replace').strip()
            except OSError:
                continue
            if cmdline and any(r.search(cmdline) for r in regexes):
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append((pid, cmdline))
                except (ProcessLookupError, PermissionError):
                    pass
        if killed:
            time.sleep(0.5)
        return killed

    def report(self):
        """Lines describing how every script ended (for the shutdown report)."""
        lines = []
        for name, mp in self.procs.items():
            code = mp.exit_code()
            state = 'STILL RUNNING' if code is None else f'exited ({code})'
            lines.append(f'{name:<18} starts={self.starts.get(name, 0):<3} {state:<16} {mp.log_path}')
        return lines

    def close(self):
        """Close all.txt."""
        try:
            self._all.close()
        except Exception:
            pass
