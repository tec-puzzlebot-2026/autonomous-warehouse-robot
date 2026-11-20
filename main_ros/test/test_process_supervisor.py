"""Tests of the process supervisor with real (harmless) processes."""
# Author: Raymundo Gomez

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main_ros.control.process_supervisor import ProcessSupervisor  # noqa: E402


def test_start_log_stop(tmp_path):
    sup = ProcessSupervisor(str(tmp_path))
    ok, _ = sup.start('talker', 'echo hello; trap "echo bye; exit 0" INT; sleep 30 & wait')
    assert ok and sup.alive('talker')
    time.sleep(0.5)
    sup.stop('talker')
    assert not sup.alive('talker') and not sup.was_launched('talker')
    time.sleep(0.2)
    log = open(sup.log_path('talker')).read()
    assert 'hello' in log and 'exited with code' in log
    assert '[talker] hello' in open(os.path.join(str(tmp_path), 'all.log')).read()
    sup.close()


def test_stubborn_process_is_killed_and_async_job(tmp_path):
    sup = ProcessSupervisor(str(tmp_path))
    sup.start('stubborn', 'trap "" INT TERM; sleep 60')
    sup.start('quiet', 'sleep 60')
    time.sleep(0.3)
    t0 = time.time()
    job = sup.stop_many_async(['stubborn', 'quiet'])
    while not job.done():
        time.sleep(0.05)
    assert not sup.alive('stubborn') and not sup.alive('quiet')
    assert time.time() - t0 < 15


def test_empty_command_and_stale_kill(tmp_path):
    sup = ProcessSupervisor(str(tmp_path))
    assert not sup.start('x', '')[0]
    stale = subprocess.Popen(['bash', '-c', 'exec -a stale_sendero_marker sleep 60'])
    time.sleep(0.3)
    killed = sup.kill_stale([r'stale_sendero_marker'])
    assert any(pid == stale.pid for pid, _ in killed)
    stale.wait(timeout=5)
