"""Lanzador: arranca el simulador y la interfaz web con un solo comando.

    python run_all.py              simulador + interfaz
    python run_all.py --no-sim     solo la interfaz (cuando ya hay un robot real publicando)
    python run_all.py --open       además abre el navegador
    python run_all.py --skip-check no verifica el broker antes de arrancar

El broker MQTT NO se arranca aquí (ver Paso 2 del manual). Ctrl+C detiene todo.
Funciona igual en macOS, Linux y Windows.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _pump(name, stream):
    """Copia la salida de un proceso hijo anteponiendo su nombre."""
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        if line:
            print(f"[{name}] {line}", flush=True)


def run(commands, env=None, poll=0.3, grace=5.0):
    """Ejecuta {nombre: argv} en paralelo.

    Termina cuando se pulsa Ctrl+C o cuando cualquiera de los procesos muere;
    en ambos casos detiene a los demás. Devuelve 0 si fue por Ctrl+C, o el
    código del primer proceso que terminó por su cuenta.
    """
    procs = {}
    try:
        for name, argv in commands.items():
            p = subprocess.Popen(argv, cwd=str(HERE), env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            procs[name] = p
            threading.Thread(target=_pump, args=(name, p.stdout), daemon=True).start()
        while True:
            for name, p in procs.items():
                code = p.poll()
                if code is not None:
                    print(f"[run_all] '{name}' terminó (código {code}); deteniendo el resto.", flush=True)
                    return code
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\n[run_all] Ctrl+C: deteniendo todo...", flush=True)
        return 0
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + grace
        for p in procs.values():
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-sim", action="store_true", help="no arrancar el simulador")
    ap.add_argument("--open", action="store_true", help="abrir el navegador al arrancar")
    ap.add_argument("--skip-check", action="store_true", help="no verificar el broker antes de arrancar")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(HERE))
    if not args.skip_check:
        import config
        import doctor
        if not doctor.check_broker(config.MQTT_HOST, config.MQTT_PORT, config.MQTT_USER, config.MQTT_PASS):
            print("\n[run_all] Arranca primero el broker (Paso 2 del manual) o usa --skip-check.")
            return 1

    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    commands = {}
    if not args.no_sim:
        commands["sim"] = [sys.executable, "simulator.py"]
    commands["web"] = [sys.executable, "app.py"]

    port = os.getenv("UI_PORT", "8080")
    print(f"[run_all] Interfaz en http://localhost:{port}  (Ctrl+C para detener)", flush=True)
    if args.open:
        threading.Timer(3.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    return run(commands, env=env)


if __name__ == "__main__":
    sys.exit(main())
