"""Diagnóstico del entorno: dice si todo está listo para correr la interfaz.

    python doctor.py

Usa solo la biblioteca estándar, así que funciona aunque todavía no hayas
instalado las dependencias. Salida en ASCII para que se vea bien en cualquier
terminal (macOS, Linux, Windows). Código de salida 0 = sin fallas.
"""
import importlib
import importlib.metadata as md
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_PY = (3, 9)

# --- salida ---------------------------------------------------------------
_fails = 0
_warns = 0


def ok(msg):
    print(f"  [OK]     {msg}")


def warn(msg, fix=None):
    global _warns
    _warns += 1
    print(f"  [AVISO]  {msg}")
    if fix:
        print(f"           -> {fix}")


def fail(msg, fix=None):
    global _fails
    _fails += 1
    print(f"  [FALLA]  {msg}")
    if fix:
        print(f"           -> {fix}")


# --- MQTT crudo (sin paho) ---------------------------------------------------
def _varint(n):
    out = bytearray()
    while True:
        byte = n % 128
        n //= 128
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _lp(s):  # cadena MQTT con prefijo de longitud
    b = s.encode("utf-8")
    return len(b).to_bytes(2, "big") + b


def mqtt_probe(host, port, user=None, password=None, timeout=3.0):
    """Abre una conexión MQTT 3.1.1 y devuelve (código_connack | None, error | None).

    Código 0 = aceptada, 4 = credenciales malas, 5 = no autorizado.
    """
    flags = 0x02  # clean session
    payload = _lp(f"puzzlebot-doctor-{os.getpid()}")
    if user:
        flags |= 0x80
        payload += _lp(user)
        if password:
            flags |= 0x40
            payload += _lp(password)
    var = _lp("MQTT") + bytes([4, flags]) + (10).to_bytes(2, "big")
    body = var + payload
    packet = b"\x10" + _varint(len(body)) + body
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(packet)
            resp = b""
            while len(resp) < 4:
                chunk = s.recv(4 - len(resp))
                if not chunk:
                    break
                resp += chunk
            if len(resp) < 4 or resp[0] != 0x20:
                return None, "el puerto responde, pero no parece un broker MQTT"
            try:
                s.sendall(b"\xe0\x00")  # DISCONNECT
            except OSError:
                pass
            return resp[3], None
    except socket.timeout:
        return None, "tiempo de espera agotado"
    except OSError as exc:
        return None, exc.strerror or str(exc)


def check_broker(host, port, user=None, password=None):
    """Imprime el resultado y devuelve True si el broker acepta la conexión."""
    code, err = mqtt_probe(host, port, user, password)
    where = f"{host}:{port}"
    if err:
        fail(f"No se pudo hablar con el broker en {where} ({err})",
             "arranca el broker (Paso 2 del manual) o revisa MQTT_HOST y MQTT_PORT")
        return False
    if code == 0:
        ok(f"Broker MQTT responde en {where}")
        return True
    if code in (4, 5):
        fail(f"El broker en {where} rechazó el acceso (código {code})",
             "revisa MQTT_USER y MQTT_PASS, o que mosquitto.conf tenga allow_anonymous true")
    else:
        fail(f"El broker en {where} rechazó la conexión (código {code})")
    return False


# --- verificaciones ------------------------------------------------------------
def check_python():
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})"
    if (v.major, v.minor) >= MIN_PY:
        ok(label)
    else:
        fail(label + f" es menor a {MIN_PY[0]}.{MIN_PY[1]}",
             "instala Python 3.9 o más reciente (sección Requisitos del manual)")
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        warn("No estás dentro de un entorno virtual",
             "actívalo: source .venv/bin/activate (macOS/Linux) o .venv\\Scripts\\Activate.ps1 (Windows)")
    else:
        ok(f"Entorno virtual activo ({sys.prefix})")


def check_files():
    missing = [n for n in ("config.py", "app.py", "simulator.py", "requirements.txt", "mosquitto.conf")
               if not (HERE / n).exists()]
    if missing:
        fail("Faltan archivos del proyecto: " + ", ".join(missing), "vuelve a descomprimir el zip completo")
    else:
        ok("Archivos del proyecto completos")


def check_modules():
    pkgs = [("nicegui", "nicegui"), ("paho.mqtt.client", "paho-mqtt"), ("PIL", "pillow")]
    all_ok = True
    for module, dist in pkgs:
        try:
            importlib.import_module(module)
        except ImportError:
            all_ok = False
            fail(f"Falta la biblioteca {dist}", "python -m pip install -r requirements.txt")
            continue
        try:
            version = md.version(dist)
        except md.PackageNotFoundError:
            version = "?"
        ok(f"{dist} {version}")
        if dist == "paho-mqtt":
            try:
                paho = importlib.import_module("paho.mqtt.client")
                if not hasattr(paho, "CallbackAPIVersion"):
                    fail("paho-mqtt es la versión 1.x; el proyecto necesita 2.0 o más",
                         "python -m pip install --upgrade \"paho-mqtt>=2.0\"")
            except ImportError:
                pass
    return all_ok


def check_port(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        ok(f"El puerto web {port} está libre")
    except OSError:
        warn(f"El puerto web {port} ya está en uso (¿la interfaz ya está corriendo?)",
             "cierra el otro proceso o cambia el puerto: UI_PORT=8081 (ver manual)")
    finally:
        s.close()


def main():
    sys.path.insert(0, str(HERE))
    print(f"Diagnóstico del proyecto en {HERE}\n")
    print("Python y entorno")
    check_python()
    print("\nArchivos")
    check_files()
    print("\nBibliotecas")
    check_modules()

    print("\nRed")
    try:
        import config
        host, port, user, pwd, prefix = (config.MQTT_HOST, config.MQTT_PORT,
                                         config.MQTT_USER, config.MQTT_PASS, config.PREFIX)
        print(f"  (broker configurado: {host}:{port}, prefijo '{prefix}')")
        check_broker(host, port, user, pwd)
        if host in ("broker.hivemq.com", "test.mosquitto.org", "broker.emqx.io") and prefix == "puzzlebot":
            warn("Broker público con el prefijo por defecto 'puzzlebot'",
                 "define TOPIC_PREFIX con algo único para no mezclarte con otras personas")
    except ImportError:
        fail("No se pudo cargar config.py")
    check_port("0.0.0.0", int(os.getenv("UI_PORT", "8080")))

    print()
    if _fails:
        print(f"Resultado: {_fails} falla(s), {_warns} aviso(s). Corrige las FALLAS y vuelve a correr doctor.py.")
        return 1
    if _warns:
        print(f"Resultado: todo listo, con {_warns} aviso(s).")
    else:
        print("Resultado: todo listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
