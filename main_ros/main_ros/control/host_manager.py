#!/usr/bin/env python3
# =============================================================================
# host_manager.py — Team Zero warehouse robot: host-side manager (laptop)
# =============================================================================
# The host PC only watches and commands the robot. It does it WITHOUT joining
# the robot's ROS 2 network: the laptop runs ROS 2 Humble, the robot side runs
# Jazzy, and Humble processes that receive Jazzy discovery data fail to decode
# it ("Bad alloc ... ParticipantEntitiesInfo") and can be killed for running
# out of memory. So:
#
#   1. the mission monitor runs ON THE ROBOT SIDE (Spark container / Jetson)
#      and is shown here in a terminal window through ssh (+ docker exec);
#   2. Foxglove Studio connects to the foxglove_bridge that the robot's
#      Program Manager starts (a websocket, not DDS);
#   3. optionally RViz (visualization rviz) — this one DOES need DDS.
#
# This script is plain Python (no rclpy). In this terminal it prints whether
# the robot's Foxglove bridge is reachable. Ctrl+C closes the monitor window
# and Foxglove Studio; the robot is not affected. When the monitor window
# closes (X finished on the robot, or Q) this side closes too.
#
# Start:  ros2 launch main_ros host.launch.py                 (sim)
#         ros2 launch main_ros host.launch.py mode:=real
#    or:  ros2 run main_ros host_manager --mode sim
#
# Logs: main_ros/logs/host_manager_<mode>_<date>_<time>/  manager.txt, all.txt,
#       one .txt per process
# =============================================================================
# Author: Raymundo Gomez

import argparse
import os
import shlex
import shutil
import signal
import socket
import threading
import time

import yaml

from main_ros.control.process_supervisor import ProcessSupervisor, run_log_dir


def find_config_dir():
    """main_ros/config: installed share first, then the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory('main_ros'), 'config')
        if os.path.isfile(os.path.join(d, 'host_manager.yaml')):
            return d
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, '..', '..', 'config'))


def parse_ws_url(url):
    """'ws://host:port' -> (host, port)."""
    rest = url.split('://', 1)[-1].strip('/')
    host, _, port = rest.partition(':')
    return host, int(port or 8765)


class HostManager:
    """Opens the monitor window and Foxglove; reports the robot link."""

    def __init__(self, mode, visualization, monitor, config_file):
        if mode not in ('sim', 'real'):
            raise ValueError(f'mode must be "sim" or "real", got "{mode}"')
        if visualization not in ('foxglove', 'rviz', 'none'):
            raise ValueError('visualization must be foxglove, rviz or none')
        self.mode = mode
        self.visualization = visualization
        self.open_monitor = monitor
        config_file = config_file or os.path.join(find_config_dir(), 'host_manager.yaml')
        with open(config_file) as f:
            self.cfg = yaml.safe_load(f)
        self.robot = (self.cfg.get('robot') or {}).get(mode) or {}
        self.log_dir = run_log_dir(f'host_manager_{mode}',
                                   str(self.cfg.get('log_root', '') or ''))
        self._mlog = open(os.path.join(self.log_dir, 'manager.txt'), 'a', buffering=1)
        self.sup = ProcessSupervisor(self.log_dir, logger=self._log)
        self.stopping = False
        self._link = None
        self._reported_exit = set()

        self._log('=' * 64)
        self._log(f'Host manager — {mode.upper()} mode, visualization: {visualization}')
        self._log(f'config: {config_file}')
        self._log(f'logs  : {self.log_dir}')
        self._log(f'robot : ssh {self.robot.get("ssh") or "(this PC)"}'
                  + (f', container {self.robot["container"]}'
                     if self.robot.get('container') else ''))
        self._log('=' * 64)

    # =====================================================================
    # start-up
    # =====================================================================
    def start(self):
        killed = self.sup.kill_stale(self.cfg.get('stale_process_patterns', []))
        for pid, cmd in killed:
            self._log(f'killed leftover process {pid}: {cmd[:100]}')
        if self.visualization == 'foxglove':
            self._start_foxglove()
        elif self.visualization == 'rviz':
            self._start_rviz()
        if self.open_monitor:
            self._start_monitor()
        self._log('Host side ready. Bring the robot up with S in the monitor.')

    # ── Foxglove ─────────────────────────────────────────────────────────
    def _start_foxglove(self):
        self._log('── Foxglove ' + '─' * 52)
        url = str(self.robot.get('foxglove', '') or '')
        if not url:
            self._log(f'ERROR: set robot.{self.mode}.foxglove in host_manager.yaml '
                      '(ws://<robot address>:8765)')
            return
        fx = self.cfg.get('foxglove', {})
        if not fx.get('open_studio', True):
            self._log(f'Connect Foxglove to {url}')
            return
        apps = [a for a in fx.get('studio_commands', ['foxglove-studio', 'foxglove'])
                if shutil.which(a)]
        if not apps:
            self._log('Foxglove Studio (desktop) not found. Open Foxglove (desktop or '
                      'https://app.foxglove.dev) -> Open connection -> Foxglove '
                      f'WebSocket -> {url}')
            return
        link = f'foxglove://open?ds=foxglove-websocket&ds.url={url}/'
        self.sup.start('foxglove_studio', f'{shlex.quote(apps[0])} {shlex.quote(link)}')
        self._log(f'Foxglove Studio opened ({apps[0]}), connecting to {url}')

    # ── RViz (needs DDS to the robot) ────────────────────────────────────
    def _start_rviz(self):
        self._log('── RViz ' + '─' * 56)
        self._log('WARNING: RViz joins the robot\'s ROS 2 network over DDS. With a '
                  'Humble laptop and a Jazzy robot this can fail (see the header).')
        cmd = self.cfg.get('rviz', {}).get('cmd', 'rviz2 -d "$(ros2 pkg prefix planisferio)/share/planisferio/rviz/planisferio.rviz"')
        ok, msg = self.sup.start('rviz', cmd)
        self._log('RViz started' if ok else f'ERROR: RViz not started: {msg}')

    # ── monitor (runs on the robot side, shown here) ─────────────────────
    def remote_monitor_command(self):
        """Shell command that runs the monitor on the robot side."""
        r = self.robot
        run = f'{r.get("setup", "true")} && exec ros2 run main_ros mission_monitor'
        container = str(r.get('container', '') or '')
        if container:
            cmd = (f'docker exec -it -u "$(id -u):$(id -g)" {shlex.quote(container)} '
                   f'bash -c {shlex.quote(run)}')
        else:
            cmd = f'bash -c {shlex.quote(run)}'
        ssh = str(r.get('ssh', '') or '')
        if ssh:
            return ' '.join(['ssh', '-t', '-o', 'ServerAliveInterval=5',
                             '-o', 'ServerAliveCountMax=3',
                             shlex.quote(ssh), shlex.quote(cmd)])
        return cmd

    def monitor_command(self):
        """Command that opens the monitor in a new terminal window, or None."""
        mon = self.cfg.get('monitor', {})
        title = mon.get('title', 'Team Zero Mission Monitor')
        geometry = mon.get('geometry', '84x41')
        # Exit code 0 (X finished, or Q): the window closes by itself.
        # Anything else (ssh failed, container not running...): it stays open
        # so the error can be read.
        inner = '; '.join([
            'printf "\\033]11;#101010\\007\\033]10;#cfcfcf\\007"',
            self.remote_monitor_command(),
            'rc=$?',
            'if [ $rc -ne 0 ]; then echo; echo "Monitor ended with code $rc. '
            'Press ENTER to close this window."; read; fi',
        ])
        if shutil.which('gnome-terminal'):
            argv = ['gnome-terminal', '--wait', f'--title={title}',
                    f'--geometry={geometry}', '--', 'bash', '-c', inner]
        elif shutil.which('xterm'):
            argv = ['xterm', '-T', title, '-geometry', geometry, '-bg', '#101010',
                    '-fg', '#cfcfcf', '-fa', 'Monospace', '-fs', '10',
                    '-e', 'bash', '-c', inner]
        else:
            return None
        return ' '.join(shlex.quote(a) for a in argv)

    def _start_monitor(self):
        self._log('── Monitor ' + '─' * 53)
        cmd = self.monitor_command()
        if cmd is None:
            self._log('No gnome-terminal or xterm found. In a terminal run:  '
                      + self.remote_monitor_command())
            return
        ok, msg = self.sup.start('monitor', cmd)
        if ok:
            self._log('Monitor window opened (it runs on the robot side'
                      + (f' through ssh {self.robot.get("ssh")}' if self.robot.get('ssh')
                         else '') + ')')
        else:
            self._log(f'ERROR: monitor not opened: {msg}')

    # =====================================================================
    # watching
    # =====================================================================
    def watch(self):
        """Report the robot link and closed windows; False = time to close."""
        url = str(self.robot.get('foxglove', '') or '')
        if url:
            host, port = parse_ws_url(url)
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    up = True
            except OSError:
                up = False
            if up != self._link:
                self._link = up
                self._log(f'robot Foxglove bridge {url}: '
                          + ('reachable' if up else 'NOT reachable (main.launch.py not '
                             'running yet, the bridge not started, or the network)'))
        for name in ('monitor', 'foxglove_studio', 'rviz'):
            mp = self.sup.get(name)
            if mp is not None and not mp.stopping and not mp.alive() \
                    and name not in self._reported_exit:
                self._reported_exit.add(name)
                if name == 'monitor':
                    if self.cfg.get('exit_with_monitor', True):
                        self._log('Monitor window closed: closing the host side.')
                        return False
                    self._log('Monitor window closed. Reopen it with:  '
                              + self.remote_monitor_command())
                else:
                    self._log(f'{name} closed (code {mp.exit_code()}); see {mp.log_path}')
        return True

    # =====================================================================
    def _log(self, text):
        line = f'[{time.strftime("%H:%M:%S")}] {text}'
        print(line, flush=True)
        try:
            self._mlog.write(line + '\n')
        except ValueError:
            pass

    def shutdown(self):
        """Close what this manager opened (the robot keeps running)."""
        self.stopping = True
        self._log('Shutting down the host side (the robot is not affected)...')
        if not self.cfg.get('foxglove', {}).get('close_studio_on_exit', True):
            self.sup.procs.pop('foxglove_studio', None)
        self.sup.stop_all()
        for line in self.sup.report():
            self._log(line)
        self._log(f'Logs: {self.log_dir}')
        self.sup.close()


def main(args=None):
    """Parse the options, open everything, wait for Ctrl+C."""
    ap = argparse.ArgumentParser(description='Team Zero host-side manager')
    ap.add_argument('--mode', default='sim', choices=['sim', 'real'])
    ap.add_argument('--visualization', default='foxglove',
                    choices=['foxglove', 'rviz', 'none'])
    ap.add_argument('--monitor', default='true')
    ap.add_argument('--config-file', default='')
    opts, _unknown = ap.parse_known_args(args)
    manager = HostManager(opts.mode, opts.visualization,
                          opts.monitor.strip().lower() in ('true', '1', 'yes'),
                          opts.config_file)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        manager.start()
        while not stop.is_set():
            if not manager.watch():
                break
            stop.wait(1.0)
    finally:
        manager.shutdown()


if __name__ == '__main__':
    main()