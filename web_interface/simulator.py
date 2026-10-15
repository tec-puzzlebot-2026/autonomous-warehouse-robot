"""Simulador del robot + state machine, para desarrollar la interfaz sin el Puzzlebot.

Publica la telemetría por MQTT con el mismo contrato de tópicos (config.py) que
usará el nodo puente real, y atiende los comandos que llegan por T_CMD.

Uso:
    python simulator.py

Flujo de la misión simulada (como el diagrama del proyecto):
    IDLE -> SEARCH_TRAILER -> ALIGN -> GRAB_PALLET -> TRANSPORT -> RELEASE -> DONE
    (ESTOP desde cualquier estado; reset vuelve a IDLE)
"""
import io
import json
import math
import os
import random
import threading
import time

from config import (
    MQTT_HOST, MQTT_PORT, MQTT_USER, MQTT_PASS,
    T_STATUS, T_STATE, T_POSE, T_VEL, T_PERCEPTION, T_GRIPPER,
    T_SYSTEM, T_LOG, T_PATH, T_CAMERA, T_CMD,
)

WHEEL_BASE = 0.19  # m (aprox. Puzzlebot)
V_MAX = 0.30       # m/s
W_MAX = 1.5        # rad/s


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class RobotSim:
    """Modelo mínimo: robot diferencial + state machine de la misión.

    No usa MQTT ni reloj real: se avanza con step(dt), así es fácil de probar.
    """

    def __init__(self, seed=None):
        self.rnd = random.Random(seed)
        self._clock = 0.0
        self._logs = []
        self.mode = "auto"
        self._reset_mission()

    # ------------------------------------------------------------------ util
    def _log(self, level, msg):
        self._logs.append({"ts": time.time(), "level": level, "msg": msg})

    def pop_logs(self):
        logs, self._logs = self._logs, []
        return logs

    def _reset_mission(self):
        self.state = "IDLE"
        self.x = self.y = self.th = 0.0
        self.enc_r = self.enc_l = 0.0
        self.sp_r = self.sp_l = 0.0
        self.v_cmd = self.w_cmd = 0.0
        self.v_teleop = self.w_teleop = 0.0
        self.teleop_until = 0.0
        self.gripper = "open"
        self._grip_target = "open"
        self._grip_timer = 0.0
        self.t_state = 0.0
        self.goal = (2.0, 1.0)
        self.path = []
        self.dist0 = 1.0
        self.progress = 0.0
        self.trailer_seen = False
        self.trailer_conf = 0.0
        self.aruco_id = None
        self.align_err = 0.35

    def _set_state(self, new):
        if new != self.state:
            self._log("info", f"Estado: {self.state} -> {new}")
            self.state = new
            self.t_state = 0.0

    def _set_gripper(self, target):
        """target: 'open' | 'close'. El gripper tarda 1.5 s en llegar."""
        want = "closed" if target == "close" else "open"
        if self.gripper == want:
            return
        self._grip_target = want
        self._grip_timer = 1.5
        self.gripper = "closing" if target == "close" else "opening"

    def _plan_path(self):
        gx, gy = self.goal
        n = 10
        self.path = [
            [round(self.x + (gx - self.x) * i / n, 3),
             round(self.y + (gy - self.y) * i / n, 3)]
            for i in range(n + 1)
        ]
        self.dist0 = max(0.05, math.hypot(gx - self.x, gy - self.y))

    # -------------------------------------------------------------- comandos
    def handle_command(self, msg):
        if not isinstance(msg, dict):
            return
        cmd = msg.get("cmd")

        if cmd == "estop":
            self._set_state("ESTOP")
            self._log("error", "PARO DE EMERGENCIA activado")
            return

        if self.state == "ESTOP" and cmd != "reset":
            self._log("warn", f"Comando '{cmd}' ignorado: robot en ESTOP (haz reset)")
            return

        if cmd == "reset":
            self._reset_mission()
            self._log("info", "Reset: misión y pose reiniciadas")
        elif cmd == "start":
            if self.mode != "auto":
                self._log("warn", "start ignorado: el robot está en modo manual")
            elif self.state in ("IDLE", "DONE"):
                self.trailer_seen = False
                self.trailer_conf = 0.0
                self.aruco_id = None
                self.align_err = 0.35
                self.progress = 0.0
                self._set_state("SEARCH_TRAILER")
            else:
                self._log("warn", f"start ignorado: misión en curso ({self.state})")
        elif cmd == "stop":
            self._set_state("IDLE")
            self._log("info", "Misión detenida")
        elif cmd == "mode":
            value = msg.get("value")
            if value in ("auto", "manual"):
                self.mode = value
                self._log("info", f"Modo: {value}")
        elif cmd == "goal":
            try:
                self.goal = (float(msg["x"]), float(msg["y"]))
            except (KeyError, TypeError, ValueError):
                self._log("warn", "goal inválido")
                return
            self._log("info", f"Nueva meta: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
            if self.state == "TRANSPORT":
                self._plan_path()
        elif cmd == "teleop":
            if self.mode != "manual":
                return  # se ignora en silencio para no llenar el log
            try:
                self.v_teleop = _clamp(float(msg.get("v", 0)), -V_MAX, V_MAX)
                self.w_teleop = _clamp(float(msg.get("w", 0)), -W_MAX, W_MAX)
            except (TypeError, ValueError):
                return
            self.teleop_until = self._clock + 0.5  # "hombre muerto": 0.5 s
        elif cmd == "gripper":
            if self.mode != "manual":
                self._log("warn", "gripper manual solo en modo manual")
                return
            value = msg.get("value")
            if value in ("open", "close"):
                self._set_gripper(value)
                self._log("info", f"Gripper: {value}")
        else:
            self._log("warn", f"Comando desconocido: {cmd}")

    # ------------------------------------------------------------------ paso
    def _auto_step(self):
        """Devuelve (v, w) deseados por la state machine."""
        s, t = self.state, self.t_state

        if s == "SEARCH_TRAILER":
            if t > 4.0:
                self.trailer_seen = True
                target = 0.93 + self.rnd.uniform(-0.03, 0.03)
                self.trailer_conf += (target - self.trailer_conf) * 0.15
                if self.trailer_conf > 0.85 and t > 6.0:
                    self._set_state("ALIGN")
            return 0.0, 0.4

        if s == "ALIGN":
            self.aruco_id = 7
            self.align_err = max(0.02, self.align_err - 0.05 * self._dt)
            self.trailer_conf = 0.9 + self.rnd.uniform(-0.03, 0.05)
            if self.align_err <= 0.02:
                self._set_state("GRAB_PALLET")
                self._set_gripper("close")
            return 0.05, 0.4 * self.align_err * math.sin(2.0 * self._clock)

        if s == "GRAB_PALLET":
            if self.gripper == "closed" and t > 2.0:
                self._plan_path()
                self._set_state("TRANSPORT")
            return 0.0, 0.0

        if s == "TRANSPORT":
            gx, gy = self.goal
            dx, dy = gx - self.x, gy - self.y
            dist = math.hypot(dx, dy)
            self.progress = _clamp(1.0 - dist / self.dist0, 0.0, 1.0)
            if dist < 0.08:
                self._set_state("RELEASE")
                self._set_gripper("open")
                return 0.0, 0.0
            err = _wrap(math.atan2(dy, dx) - self.th)
            v = min(0.25, 0.6 * dist) * max(0.0, math.cos(err))
            w = _clamp(1.8 * err, -1.0, 1.0)
            return v, w

        if s == "RELEASE":
            if self.gripper == "open" and t > 2.0:
                self.progress = 1.0
                self._set_state("DONE")
                self._log("info", "Misión completada")
            return 0.0, 0.0

        return 0.0, 0.0  # IDLE, DONE

    def step(self, dt):
        self._dt = dt
        self._clock += dt
        self.t_state += dt

        # gripper (temporizador de movimiento)
        if self._grip_timer > 0:
            self._grip_timer -= dt
            if self._grip_timer <= 0:
                self.gripper = self._grip_target

        # velocidad deseada
        if self.state == "ESTOP":
            v_t = w_t = 0.0
        elif self.mode == "manual":
            if self._clock < self.teleop_until:
                v_t, w_t = self.v_teleop, self.w_teleop
            else:
                v_t = w_t = 0.0
        else:
            v_t, w_t = self._auto_step()

        self.v_cmd, self.w_cmd = v_t, w_t
        self.sp_r = v_t + w_t * WHEEL_BASE / 2.0
        self.sp_l = v_t - w_t * WHEEL_BASE / 2.0

        # respuesta de los motores (primer orden, ~0.15 s)
        a = min(1.0, dt / 0.15)
        self.enc_r += (self.sp_r - self.enc_r) * a
        self.enc_l += (self.sp_l - self.enc_l) * a

        # cinemática
        v = (self.enc_r + self.enc_l) / 2.0
        w = (self.enc_r - self.enc_l) / WHEEL_BASE
        self.th = _wrap(self.th + w * dt)
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt

    # ------------------------------------------------------------- mensajes
    def state_msg(self):
        return {
            "state": self.state, "mode": self.mode,
            "progress": round(self.progress, 3),
            "t_state": round(self.t_state, 1),
            "goal": [self.goal[0], self.goal[1]],
            "ts": time.time(),
        }

    def pose_msg(self):
        return {"x": round(self.x, 4), "y": round(self.y, 4),
                "theta": round(self.th, 4), "ts": time.time()}

    def velocity_msg(self):
        n = lambda: self.rnd.gauss(0, 0.004)
        return {
            "cmd_v": round(self.v_cmd, 4), "cmd_w": round(self.w_cmd, 4),
            "setpoint_r": round(self.sp_r, 4), "setpoint_l": round(self.sp_l, 4),
            "enc_r": round(self.enc_r + n(), 4), "enc_l": round(self.enc_l + n(), 4),
            "ts": time.time(),
        }

    def perception_msg(self):
        return {
            "trailer_detected": self.trailer_seen,
            "trailer_conf": round(self.trailer_conf, 3) if self.trailer_seen else 0.0,
            "aruco_id": self.aruco_id,
            "align_error": round(self.align_err, 3),
            "ts": time.time(),
        }

    def gripper_msg(self):
        return {"state": self.gripper, "ts": time.time()}

    def system_msg(self):
        # "voice" va vacío: lo llenará el nodo NLP del robot real.
        return {"voice": "", "uptime": round(self._clock, 1), "ts": time.time()}

    def path_msg(self):
        return {"points": self.path, "ts": time.time()}


# ---------------------------------------------------------------------------
# Cámara sintética (reemplaza /cam_img) — solo requiere Pillow
# ---------------------------------------------------------------------------
def render_frame(sim, w=320, h=240):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), (38, 46, 58))
    d = ImageDraw.Draw(img)
    horizon = int(h * 0.5)
    d.rectangle([0, horizon, w, h], fill=(66, 70, 68))
    # franjas verticales que se desplazan al girar el robot (sensación de movimiento)
    for i in range(-1, 9):
        x = int((i * 48 - sim.th * 120) % (w + 48)) - 24
        d.line([x, 0, x, horizon], fill=(52, 62, 76), width=3)

    if sim.trailer_seen:
        k = _clamp(1.0 - sim.align_err / 0.35, 0.0, 1.0)
        bw = int(80 + 130 * k)
        bh = int(bw * 0.7)
        cx = int(w / 2 + (sim.align_err / 0.35) * 70)
        bottom = int(h * 0.85)
        x0, x1 = cx - bw // 2, cx + bw // 2
        y0, y1 = bottom - bh, bottom
        d.rectangle([x0, y0, x1, y1], fill=(150, 150, 155), outline=(210, 210, 215), width=2)
        if sim.aruco_id is not None:
            m = max(18, bw // 5)
            mx0, my0 = cx - m // 2, y0 + bh // 3
            d.rectangle([mx0, my0, mx0 + m, my0 + m], fill=(0, 0, 0))
            q = m // 4
            for r in range(3):
                for c in range(3):
                    if (r * 3 + c + sim.aruco_id) % 2 == 0:
                        d.rectangle([mx0 + q * (c + 0.5), my0 + q * (r + 0.5),
                                     mx0 + q * (c + 1.5), my0 + q * (r + 1.5)], fill=(255, 255, 255))
        d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=(60, 220, 90), width=2)
        d.text((x0 - 4, max(0, y0 - 16)), f"trailer {sim.trailer_conf:.2f}", fill=(60, 220, 90))

    d.text((6, 6), f"SIM  {sim.state}", fill=(255, 255, 255))
    d.text((6, h - 16), time.strftime("%H:%M:%S"), fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Publicación periódica (independiente de MQTT para poder probarla)
# ---------------------------------------------------------------------------
class SimRunner:
    """Avanza la simulación y publica según la frecuencia de cada tópico.

    publish(topic, payload: str|bytes, qos, retain) es cualquier función.
    """

    RATES = {  # periodo en segundos
        "pose": 0.1, "vel": 0.1, "perception": 0.2, "camera": 0.2,
        "state": 0.5, "system": 1.0, "path": 1.0,
    }

    def __init__(self, sim, publish):
        self.sim = sim
        self.publish = publish
        self._acc = {k: 0.0 for k in self.RATES}
        self._last_key = None

    def _pub_json(self, topic, obj, qos=0, retain=False):
        self.publish(topic, json.dumps(obj, separators=(",", ":")), qos, retain)

    def tick(self, dt):
        sim = self.sim
        sim.step(dt)

        for k in self._acc:
            self._acc[k] += dt

        # cambios de estado/modo/gripper -> publicar al instante (retenido)
        key = (sim.state, sim.mode, sim.gripper)
        if key != self._last_key:
            self._last_key = key
            self._acc["state"] = self.RATES["state"]
            self._pub_json(T_GRIPPER, sim.gripper_msg(), qos=1, retain=True)

        def due(name):
            if self._acc[name] >= self.RATES[name]:
                self._acc[name] = 0.0
                return True
            return False

        if due("state"):
            self._pub_json(T_STATE, sim.state_msg(), qos=1, retain=True)
        if due("pose"):
            self._pub_json(T_POSE, sim.pose_msg())
        if due("vel"):
            self._pub_json(T_VEL, sim.velocity_msg())
        if due("perception"):
            self._pub_json(T_PERCEPTION, sim.perception_msg())
        if due("system"):
            self._pub_json(T_SYSTEM, sim.system_msg())
        if due("path"):
            self._pub_json(T_PATH, sim.path_msg())
        if due("camera"):
            self.publish(T_CAMERA, render_frame(sim), 0, False)

        for entry in sim.pop_logs():
            self._pub_json(T_LOG, entry, qos=1)


def main():
    import paho.mqtt.client as mqtt

    sim = RobotSim()
    lock = threading.Lock()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"puzzlebot-sim-{os.getpid()}")
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.will_set(T_STATUS, "offline", qos=1, retain=True)

    def on_connect(c, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"[sim] conexión rechazada: {reason_code}")
            return
        print(f"[sim] conectado a {MQTT_HOST}:{MQTT_PORT}")
        c.subscribe(T_CMD, qos=1)
        c.publish(T_STATUS, "online", qos=1, retain=True)

    def on_message(c, userdata, msg):
        try:
            cmd = json.loads(msg.payload)
        except ValueError:
            return
        with lock:
            sim.handle_command(cmd)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=15)
    client.loop_start()

    runner = SimRunner(sim, lambda topic, payload, qos, retain:
                       client.publish(topic, payload, qos=qos, retain=retain))

    print("[sim] simulador corriendo (Ctrl+C para salir)")
    dt = 0.05
    try:
        nxt = time.monotonic()
        while True:
            with lock:
                runner.tick(dt)
            nxt += dt
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        client.publish(T_STATUS, "offline", qos=1, retain=True).wait_for_publish(2)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()