"""Interfaz web en tiempo real del Puzzlebot (NiceGUI + MQTT).

    python app.py          ->  http://localhost:8080

Variables de entorno útiles (ver también config.py):
    UI_HOST (0.0.0.0)   UI_PORT (8080)
    UI_PASSWORD         si se define, la página pide contraseña antes de mostrar el
                        panel (muy recomendado si se publica en la nube: desde aquí
                        se pueden mandar comandos al robot).
    UI_SECRET           secreto para firmar la cookie de sesión (cámbialo en producción).
"""
import base64
import hmac
import json
import math
import os
import threading
import time
from collections import deque

import paho.mqtt.client as mqtt
from nicegui import app, ui

from config import (
    MQTT_HOST, MQTT_PORT, MQTT_USER, MQTT_PASS, PREFIX,
    T_STATUS, T_STATE, T_POSE, T_VEL, T_PERCEPTION, T_GRIPPER,
    T_SYSTEM, T_LOG, T_PATH, T_CAMERA, T_CMD, STATES,
)

UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
UI_PORT = int(os.getenv("UI_PORT", "8080"))
UI_PASSWORD = os.getenv("UI_PASSWORD") or None
UI_SECRET = os.getenv("UI_SECRET", "cambia-este-secreto")

V_MAX = 0.30  # m/s  (debe coincidir con los límites del robot)
W_MAX = 1.5   # rad/s
STALE_AFTER = 3.0  # s sin telemetría => "sin datos"

STATE_COLORS = {
    "IDLE": "grey", "SEARCH_TRAILER": "blue", "ALIGN": "indigo",
    "GRAB_PALLET": "deep-purple", "TRANSPORT": "teal", "RELEASE": "orange",
    "DONE": "positive", "ESTOP": "negative",
}
PIPELINE = [s for s in STATES if s != "ESTOP"]
LEVEL_ICON = {"info": "ℹ", "warn": "⚠", "error": "✖"}


# ---------------------------------------------------------------------------
# Almacén compartido: el hilo de MQTT escribe, las páginas (NiceGUI) leen.
# ---------------------------------------------------------------------------
class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False          # conexión con el broker
        self.robot_status = "offline"   # "online"/"offline" (Last Will del robot)
        self.last_rx = 0.0              # monotonic de la última telemetría recibida
        self.state = {}
        self.pose = {}
        self.vel = {}
        self.perception = {}
        self.gripper = {}
        self.system = {}
        self.path = []
        self.camera = None              # bytes JPEG
        self.camera_seq = 0
        self.camera_rx = 0.0
        self.traj = deque(maxlen=3000)  # (x, y)
        self.vel_hist = deque(maxlen=250)  # (t, sp_r, enc_r, sp_l, enc_l)
        self.logs = deque(maxlen=300)   # (seq, dict)
        self.log_seq = 0

    def add_log(self, level, msg, ts=None):
        with self.lock:
            self.log_seq += 1
            self.logs.append((self.log_seq, {"ts": ts or time.time(), "level": level, "msg": msg}))


store = Store()
_client = None


def _on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        store.connected = False
        store.add_log("error", f"Broker rechazó la conexión: {reason_code}")
        return
    store.connected = True
    for topic, qos in [(T_STATUS, 1), (T_STATE, 1), (T_POSE, 0), (T_VEL, 0),
                       (T_PERCEPTION, 0), (T_GRIPPER, 1), (T_SYSTEM, 0),
                       (T_LOG, 1), (T_PATH, 0), (T_CAMERA, 0)]:
        client.subscribe(topic, qos)
    store.add_log("info", f"Conectado al broker {MQTT_HOST}:{MQTT_PORT} (prefijo '{PREFIX}')")


def _on_disconnect(client, userdata, flags, reason_code, properties):
    store.connected = False
    store.add_log("warn", "Conexión con el broker perdida; reintentando…")


def _on_message(client, userdata, msg):
    topic = msg.topic
    now = time.monotonic()
    try:
        if topic == T_CAMERA:
            with store.lock:
                store.camera = bytes(msg.payload)
                store.camera_seq += 1
                store.camera_rx = now
                store.last_rx = now
            return
        if topic == T_STATUS:
            store.robot_status = msg.payload.decode("utf-8", "replace").strip()
            return

        data = json.loads(msg.payload)
        with store.lock:
            if topic != T_LOG:
                store.last_rx = now
            if topic == T_STATE:
                store.state = data
            elif topic == T_POSE:
                store.pose = data
                store.traj.append((data.get("x", 0.0), data.get("y", 0.0)))
            elif topic == T_VEL:
                store.vel = data
                store.vel_hist.append((now, data.get("enc_r", 0.0), data.get("enc_l", 0.0)))
            elif topic == T_PERCEPTION:
                store.perception = data
            elif topic == T_GRIPPER:
                store.gripper = data
            elif topic == T_SYSTEM:
                store.system = data
            elif topic == T_PATH:
                store.path = data.get("points", [])
        if topic == T_LOG:
            store.add_log(data.get("level", "info"), str(data.get("msg", "")), data.get("ts"))
    except (ValueError, AttributeError, TypeError) as exc:
        store.add_log("warn", f"Mensaje inválido en {topic}: {exc}")


def start_mqtt():
    global _client
    _client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                          client_id=f"puzzlebot-ui-{os.getpid()}")
    if MQTT_USER:
        _client.username_pw_set(MQTT_USER, MQTT_PASS)
    _client.on_connect = _on_connect
    _client.on_disconnect = _on_disconnect
    _client.on_message = _on_message
    _client.reconnect_delay_set(min_delay=1, max_delay=10)
    _client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=15)  # no bloquea si el broker está caído
    _client.loop_start()


def stop_mqtt():
    if _client is not None:
        _client.loop_stop()
        _client.disconnect()


def send_cmd(cmd, qos=1):
    """Publica un comando para la state machine. Devuelve True si se envió."""
    if _client is None or not store.connected:
        return False
    _client.publish(T_CMD, json.dumps(cmd, separators=(",", ":")), qos=qos)
    return True


app.on_startup(start_mqtt)
app.on_shutdown(stop_mqtt)


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def login_view():
    def try_login():
        if hmac.compare_digest(pwd.value.encode(), UI_PASSWORD.encode()):
            app.storage.user["auth"] = True
            ui.navigate.reload()
        else:
            ui.notify("Contraseña incorrecta", color="negative")
            pwd.value = ""

    with ui.card().classes("absolute-center w-80"):
        ui.label("Puzzlebot · Panel de control").classes("text-h6")
        pwd = ui.input("Contraseña", password=True, password_toggle_button=True) \
            .props("autofocus").classes("w-full")
        pwd.on("keydown.enter", try_login)
        ui.button("Entrar", on_click=try_login).classes("w-full")


@ui.page("/")
def index():
    ui.dark_mode(True)
    if UI_PASSWORD and not app.storage.user.get("auth"):
        login_view()
        return

    controls = []  # widgets que se deshabilitan si el robot no está en línea
    view = {"mode": None, "log_seq": 0, "cam_seq": -1}
    joy = {"x": 0.0, "y": 0.0, "on": False}

    def command(cmd, label=None):
        if send_cmd(cmd):
            if label:
                ui.notify(label, timeout=1200)
        else:
            ui.notify("Sin conexión con el broker: comando NO enviado", color="negative")

    # ------------------------------------------------------------ encabezado
    with ui.header().classes("items-center justify-between"):
        ui.label("Puzzlebot · Panel de control").classes("text-h6")
        with ui.row().classes("items-center gap-3"):
            broker_badge = ui.badge("Broker").props("color=grey")
            robot_badge = ui.badge("Robot").props("color=grey")
            age_label = ui.label("").classes("text-caption")
            if UI_PASSWORD:
                ui.button(icon="logout", on_click=lambda: (app.storage.user.clear(),
                                                           ui.navigate.reload())).props("flat round dense")

    with ui.row().classes("w-full items-start gap-3 p-2"):
        # ===================================================== columna: estado
        with ui.column().classes("flex-1 gap-3").style("min-width: 320px"):
            with ui.card().classes("w-full"):
                ui.label("State machine").classes("text-subtitle2 text-grey")
                state_label = ui.label("—").classes("text-h4")
                with ui.row().classes("items-center gap-2"):
                    ui.label("Modo:").classes("text-caption")
                    mode_badge = ui.badge("—").props("color=grey")
                    t_state_label = ui.label("").classes("text-caption")
                ui.label("Progreso de la misión").classes("text-caption")
                progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
                with ui.row().classes("gap-1"):
                    pipeline_badges = {s: ui.badge(s.replace("_", " ").title()).props("color=grey")
                                       for s in PIPELINE}

            with ui.card().classes("w-full"):
                ui.label("Control de misión").classes("text-subtitle2 text-grey")
                with ui.row().classes("gap-2"):
                    for text, icon, color, cmd in [
                        ("Start", "play_arrow", "positive", {"cmd": "start"}),
                        ("Stop", "stop", "warning", {"cmd": "stop"}),
                        ("Reset", "restart_alt", "grey", {"cmd": "reset"}),
                    ]:
                        controls.append(
                            ui.button(text, icon=icon, on_click=lambda c=cmd, t=text: command(c, t))
                            .props(f"color={color}"))
                ui.button("PARO DE EMERGENCIA", icon="warning",
                          on_click=lambda: command({"cmd": "estop"}, "E-STOP enviado")) \
                    .props("color=negative size=lg").classes("w-full mt-2")

                def on_mode(e):
                    if e.value != view["mode"]:
                        command({"cmd": "mode", "value": e.value}, f"Modo {e.value}")

                mode_toggle = ui.toggle(["auto", "manual"], value="auto", on_change=on_mode)
                controls.append(mode_toggle)

                ui.label("Meta (m)").classes("text-caption mt-2")
                with ui.row().classes("items-center gap-2 no-wrap"):
                    goal_x = ui.number("x", value=2.0, step=0.1, format="%.2f").classes("w-24")
                    goal_y = ui.number("y", value=1.0, step=0.1, format="%.2f").classes("w-24")
                    controls.append(ui.button(
                        "Enviar", icon="flag",
                        on_click=lambda: command({"cmd": "goal", "x": goal_x.value or 0.0,
                                                  "y": goal_y.value or 0.0}, "Meta enviada")))

            with ui.card().classes("w-full"):
                ui.label("Teleoperación (modo manual)").classes("text-subtitle2 text-grey")

                def on_move(e):
                    # se limita a [-1, 1] por si la biblioteca entrega otro rango
                    joy.update(x=max(-1.0, min(1.0, float(e.x))),
                               y=max(-1.0, min(1.0, float(e.y))), on=True)

                def on_end(_):
                    joy.update(x=0.0, y=0.0, on=False)
                    send_cmd({"cmd": "teleop", "v": 0.0, "w": 0.0}, qos=0)

                with ui.row().classes("items-center gap-4"):
                    ui.joystick(color="blue", size=90, on_move=on_move, on_end=on_end)
                    teleop_label = ui.label("v = 0.00 m/s\nw = 0.00 rad/s").style("white-space: pre")
                with ui.row().classes("items-center gap-2 mt-2"):
                    ui.label("Gripper:")
                    gripper_label = ui.label("—").classes("text-bold")
                    controls.append(ui.button("Abrir", icon="open_in_full",
                                              on_click=lambda: command({"cmd": "gripper", "value": "open"})).props("dense"))
                    controls.append(ui.button("Cerrar", icon="close_fullscreen",
                                              on_click=lambda: command({"cmd": "gripper", "value": "close"})).props("dense"))

        # ================================================== columna: percepción
        with ui.column().classes("flex-1 gap-3").style("min-width: 340px"):
            with ui.card().classes("w-full"):
                ui.label("Cámara (/cam_img)").classes("text-subtitle2 text-grey")
                camera = ui.image().props("fit=contain ratio=1.3333").classes("w-full bg-black")
                camera_note = ui.label("Sin señal de cámara").classes("text-caption text-grey")

            with ui.card().classes("w-full"):
                ui.label("Percepción").classes("text-subtitle2 text-grey")
                trailer_label = ui.label("Tráiler: —")
                conf_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
                aruco_label = ui.label("ArUco: —")
                align_label = ui.label("Error de alineación: —")

            with ui.card().classes("w-full"):
                ui.label("Sistema").classes("text-subtitle2 text-grey")
                voice_label = ui.label("Último comando de voz: —")
                uptime_label = ui.label("Tiempo activo: —")

        # ================================================= columna: movimiento
        with ui.column().classes("flex-1 gap-3").style("min-width: 380px"):
            with ui.card().classes("w-full"):
                ui.label("Localización y trayectoria").classes("text-subtitle2 text-grey")
                pose_label = ui.label("x = —   y = —   θ = —")
                axis = {"type": "value", "min": -0.5, "max": 3.5,
                        "splitLine": {"lineStyle": {"color": "#333"}}}
                traj_chart = ui.echart({
                    "animation": False, "backgroundColor": "transparent",
                    "textStyle": {"color": "#bbb"},
                    "grid": {"left": 36, "right": 12, "top": 28, "bottom": 28},
                    "legend": {"top": 0, "textStyle": {"color": "#bbb"},
                               "data": ["Planeada", "Recorrido", "Rumbo", "Robot", "Meta"]},
                    "xAxis": {**axis, "name": "x (m)"},
                    "yAxis": {**axis, "name": "y (m)"},
                    "series": [
                        {"name": "Planeada", "type": "line", "data": [], "showSymbol": False,
                         "lineStyle": {"type": "dashed", "color": "#888"}},
                        {"name": "Recorrido", "type": "line", "data": [], "showSymbol": False,
                         "lineStyle": {"color": "#4dabf7"}},
                        {"name": "Rumbo", "type": "line", "data": [], "showSymbol": False,
                         "lineStyle": {"color": "#ffa94d", "width": 3}},
                        {"name": "Robot", "type": "scatter", "data": [], "symbolSize": 12,
                         "itemStyle": {"color": "#ffa94d"}},
                        {"name": "Meta", "type": "scatter", "data": [], "symbol": "diamond",
                         "symbolSize": 14, "itemStyle": {"color": "#51cf66"}},
                    ],
                }).classes("w-full").style("height: 300px")

            with ui.card().classes("w-full"):
                ui.label("Velocidades de rueda (encoders)").classes("text-subtitle2 text-grey")
                vel_label = ui.label("cmd v = —   w = —")
                vel_chart = ui.echart({
                    "animation": False, "backgroundColor": "transparent",
                    "textStyle": {"color": "#bbb"},
                    "grid": {"left": 44, "right": 12, "top": 28, "bottom": 28},
                    "legend": {"top": 0, "textStyle": {"color": "#bbb"},
                               "data": ["Encoder R", "Encoder L"]},
                    "xAxis": {"type": "value", "min": -20, "max": 0, "name": "s",
                              "splitLine": {"lineStyle": {"color": "#333"}}},
                    # escala fija (±1.3 × velocidad máxima): con escala automática, el ruido de
                    # los encoders con el robot parado se ampliaba hasta parecer movimiento
                    "yAxis": {"type": "value", "name": "m/s",
                              "min": -round(V_MAX * 1.3, 2), "max": round(V_MAX * 1.3, 2),
                              "splitLine": {"lineStyle": {"color": "#333"}}},
                    "series": [
                        {"name": "Encoder R", "type": "line", "data": [], "showSymbol": False,
                         "lineStyle": {"color": "#ff6b6b"}},
                        {"name": "Encoder L", "type": "line", "data": [], "showSymbol": False,
                         "lineStyle": {"color": "#4dabf7"}},
                    ],
                }).classes("w-full").style("height: 240px")

    with ui.card().classes("w-full m-2").style("max-width: calc(100% - 16px)"):
        ui.label("Eventos").classes("text-subtitle2 text-grey")
        log_view = ui.log(max_lines=200).classes("w-full").style("height: 180px")

    # ------------------------------------------------------------- refresco
    def refresh():
        now = time.monotonic()
        with store.lock:
            state, pose, vel = dict(store.state), dict(store.pose), dict(store.vel)
            perc, grip, system = dict(store.perception), dict(store.gripper), dict(store.system)
            cam, cam_seq, cam_rx = store.camera, store.camera_seq, store.camera_rx
            new_logs = [(s, e) for s, e in store.logs if s > view["log_seq"]]
            last_rx = store.last_rx
        age = now - last_rx if last_rx else None

        # --- conexión
        if store.connected:
            broker_badge.set_text("Broker: conectado")
            broker_badge.props("color=positive")
        else:
            broker_badge.set_text("Broker: sin conexión")
            broker_badge.props("color=negative")
        if not store.connected:
            robot_badge.set_text("Robot: ?")
            robot_badge.props("color=grey")
            online = False
        elif store.robot_status != "online":
            robot_badge.set_text("Robot: desconectado")
            robot_badge.props("color=negative")
            online = False
        elif age is None or age > STALE_AFTER:
            robot_badge.set_text("Robot: sin datos")
            robot_badge.props("color=warning")
            online = False
        else:
            robot_badge.set_text("Robot: en línea")
            robot_badge.props("color=positive")
            online = True
        age_label.set_text(f"último dato hace {age:.1f} s" if age is not None else "sin datos aún")
        for c in controls:
            c.set_enabled(online)

        # --- state machine
        st = state.get("state", "—")
        mode = state.get("mode", "—")
        state_label.set_text(st.replace("_", " "))
        state_label.classes(replace="text-h4 " + ("text-negative" if st == "ESTOP" else ""))
        mode_badge.set_text(mode)
        mode_badge.props("color=" + ("deep-orange" if mode == "manual" else "primary" if mode == "auto" else "grey"))
        t_state_label.set_text(f"{state['t_state']:.0f} s en este estado" if "t_state" in state else "")
        progress.set_value(float(state.get("progress", 0.0)))
        if mode in ("auto", "manual") and mode != view["mode"]:
            view["mode"] = mode
            mode_toggle.set_value(mode)
        if st in PIPELINE:
            idx = PIPELINE.index(st)
            for i, s in enumerate(PIPELINE):
                pipeline_badges[s].props("color=" + ("primary" if i == idx else "positive" if i < idx else "grey"))
        elif st == "ESTOP":
            for s in PIPELINE:
                pipeline_badges[s].props("color=grey")

        # --- gripper / teleop
        gstate = grip.get("state", "—")
        gripper_label.set_text({"open": "abierto", "closed": "cerrado", "opening": "abriendo…",
                                "closing": "cerrando…"}.get(gstate, gstate))
        teleop_label.set_text(f"v = {joy['y'] * V_MAX:+.2f} m/s\nw = {-joy['x'] * W_MAX:+.2f} rad/s")

        # --- cámara
        if cam is not None and cam_seq != view["cam_seq"]:
            view["cam_seq"] = cam_seq
            camera.set_source("data:image/jpeg;base64," + base64.b64encode(cam).decode())
        cam_age = now - cam_rx if cam_rx else None
        camera_note.set_text("" if cam_age is not None and cam_age < 2.0 else "Sin señal de cámara")

        # --- percepción
        if perc:
            seen = perc.get("trailer_detected")
            conf = float(perc.get("trailer_conf", 0.0))
            trailer_label.set_text(f"Tráiler: {'detectado' if seen else 'no detectado'}"
                                   + (f" ({conf:.0%})" if seen else ""))
            conf_bar.set_value(conf)
            aruco = perc.get("aruco_id")
            aruco_label.set_text(f"ArUco: {'ID ' + str(aruco) if aruco is not None else '—'}")
            align_label.set_text(f"Error de alineación: {perc.get('align_error', 0.0) * 100:.1f} cm")

        # --- sistema
        if system:
            voice_label.set_text(f"Último comando de voz: {system.get('voice') or '—'}")
            up = float(system.get("uptime", 0.0))
            uptime_label.set_text(f"Tiempo activo: {int(up // 60)} min {int(up % 60)} s")

        # --- pose y velocidades (texto)
        if pose:
            pose_label.set_text(f"x = {pose.get('x', 0):.2f} m   y = {pose.get('y', 0):.2f} m   "
                                f"θ = {math.degrees(pose.get('theta', 0)):.0f}°")
        if vel:
            vel_label.set_text(f"cmd v = {vel.get('cmd_v', 0):+.2f} m/s   w = {vel.get('cmd_w', 0):+.2f} rad/s")

        # --- log
        for seq, e in new_logs:
            view["log_seq"] = seq
            log_view.push(f"[{time.strftime('%H:%M:%S', time.localtime(e['ts']))}] "
                          f"{LEVEL_ICON.get(e['level'], '•')} {e['msg']}")

    def refresh_charts():
        now = time.monotonic()
        with store.lock:
            traj = list(store.traj)[-800:]
            path = list(store.path)
            hist = list(store.vel_hist)
            pose, state = dict(store.pose), dict(store.state)
        s = traj_chart.options["series"]
        s[0]["data"] = path
        s[1]["data"] = [[round(x, 3), round(y, 3)] for x, y in traj]
        if pose:
            x, y, th = pose.get("x", 0.0), pose.get("y", 0.0), pose.get("theta", 0.0)
            s[2]["data"] = [[x, y], [x + 0.25 * math.cos(th), y + 0.25 * math.sin(th)]]
            s[3]["data"] = [[x, y]]
        goal = state.get("goal")
        s[4]["data"] = [goal] if goal else []
        traj_chart.update()

        v = vel_chart.options["series"]
        for i in range(2):  # 0 = encoder R, 1 = encoder L
            v[i]["data"] = [[round(h[0] - now, 2), h[i + 1]] for h in hist]
        vel_chart.update()

    def teleop_tick():
        # el joystick solo manda eventos al moverse; se reenvía a 10 Hz mientras esté sostenido
        if joy["on"] and view["mode"] == "manual":
            send_cmd({"cmd": "teleop", "v": round(joy["y"] * V_MAX, 3),
                      "w": round(-joy["x"] * W_MAX, 3)}, qos=0)

    ui.timer(0.2, refresh)
    ui.timer(0.5, refresh_charts)
    ui.timer(0.1, teleop_tick)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host=UI_HOST, port=UI_PORT, title="Puzzlebot", reload=False, show=False,
           storage_secret=UI_SECRET)