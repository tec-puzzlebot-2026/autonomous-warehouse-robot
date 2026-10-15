# Puzzlebot · Interfaz web en tiempo real

```
Robot (ROS2) ⇄ nodo puente (rclpy + paho-mqtt) ⇄ broker MQTT ⇄ NiceGUI (app.py) ⇄ navegador
                 (pendiente)                                      ▲
                                                simulator.py ─────┘  (sustituye al robot mientras desarrollas)
```

| Archivo | Qué es |
|---|---|
| `app.py` | La interfaz web (NiceGUI). Se suscribe por MQTT y publica comandos. |
| `simulator.py` | Robot + state machine simulados: publica telemetría falsa y obedece comandos. |
| `config.py` | Broker, prefijo y **contrato de tópicos** (compártelo con quien haga el nodo puente). |
| `mosquitto.conf` | Configuración mínima de un broker local. |

## Arranque rápido (3 terminales)

```bash
pip install -r requirements.txt

# 1) Broker local (macOS: brew install mosquitto)
mosquitto -c mosquitto.conf
#    Alternativa sin instalar nada: docker run -p 1883:1883 -v $PWD/mosquitto.conf:/mosquitto/config/mosquitto.conf eclipse-mosquitto

# 2) Simulador
python simulator.py

# 3) Interfaz  ->  http://localhost:8080
python app.py
```

Prueba: abre la página, pulsa **Start** y observa la misión
`IDLE → SEARCH_TRAILER → ALIGN → GRAB_PALLET → TRANSPORT → RELEASE → DONE` (~30 s).
Prueba también **PARO DE EMERGENCIA**, **Reset**, el modo **manual** con el joystick y el gripper.

### Broker público (solo para pruebas rápidas)

```bash
export MQTT_HOST=broker.hivemq.com TOPIC_PREFIX=puzzlebot_TUMATRICULA_xyz123
python simulator.py   # y en otra terminal: python app.py
```
Cualquiera en internet puede leer/escribir en un broker público. Usa un prefijo
único y **nunca** lo uses para mover un robot real.

## Variables de entorno

| Variable | Default | Uso |
|---|---|---|
| `MQTT_HOST` / `MQTT_PORT` | `localhost` / `1883` | Broker |
| `MQTT_USER` / `MQTT_PASS` | — | Autenticación del broker |
| `TOPIC_PREFIX` | `puzzlebot` | Prefijo de todos los tópicos |
| `UI_HOST` / `UI_PORT` | `0.0.0.0` / `8080` | Dónde sirve la web |
| `UI_PASSWORD` | — | Si se define, la página pide contraseña |
| `UI_SECRET` | `cambia-este-secreto` | Firma de la cookie de sesión |

## Contrato de tópicos (lo que debe publicar el nodo puente real)

Robot → interfaz (JSON salvo la cámara):

| Tópico | QoS / retain | Contenido | Origen en ROS2 |
|---|---|---|---|
| `…/status` | 1 / sí | `online` / `offline` (poner **Last Will** = `offline`) | — |
| `…/state` | 1 / sí | `{state, mode, progress, t_state, goal:[x,y]}` | State Machine |
| `…/pose` | 0 | `{x, y, theta}` | Localization (Kalman) |
| `…/velocity` | 0 | `{cmd_v, cmd_w, setpoint_r, setpoint_l, enc_r, enc_l}` | `/cmd_vel`, `/VelocityEncR`, `/VelocityEncL` |
| `…/perception` | 0 | `{trailer_detected, trailer_conf, aruco_id, align_error}` | YOLO / ArUco |
| `…/gripper` | 1 / sí | `{state: open\|closed\|opening\|closing}` | `/open_close` |
| `…/system` | 0 | `{battery, voice, uptime}` | NLP, batería |
| `…/path` | 0 | `{points: [[x,y], …]}` | Path Planner (A\*) |
| `…/log` | 1 | `{ts, level: info\|warn\|error, msg}` | eventos |
| `…/camera` | 0 | JPEG binario (≈5 fps, 320×240) | `/cam_img` |

Interfaz → robot, en `…/cmd` (QoS 1), JSON con `cmd`:
`start`, `stop`, `reset`, `estop`, `mode` (`value`: auto/manual), `goal` (`x`, `y`),
`teleop` (`v`, `w`; el robot debe detenerse si no llega otro en 0.5 s), `gripper` (`value`: open/close).

Estados válidos: `IDLE, SEARCH_TRAILER, ALIGN, GRAB_PALLET, TRANSPORT, RELEASE, DONE, ESTOP`.

## Qué muestra la interfaz

Estado de la state machine y progreso · control (start/stop/reset/**E-STOP**, modo, meta) ·
teleoperación con joystick y gripper · cámara con detecciones · percepción (tráiler, ArUco,
error de alineación) · pose, mapa 2D con trayectoria planeada vs. recorrida · velocidades de
rueda (consigna vs. encoder) · batería, voz, tiempo activo · log de eventos ·
estado de conexión con el broker y con el robot (marca "sin datos" si pasan >3 s sin telemetría).

## Antes de llevarlo a la nube / al robot real

- **Seguridad**: broker con usuario/contraseña y TLS (puerto 8883), `UI_PASSWORD` y un
  `UI_SECRET` propio. Desde esta página se pueden mandar comandos al robot.
- **El paro de emergencia no puede depender solo de la nube.** El robot debe frenarse solo
  si pierde la conexión (watchdog local) y debe respetar el E-STOP aunque la UI falle.
- **Video**: la cámara va por MQTT para simplificar el prototipo. Con la cámara real
  conviene bajar resolución/fps o moverla a MJPEG por HTTP o WebRTC.
- La latencia mostrada es "tiempo desde el último dato recibido"; no compara relojes
  del robot y del servidor porque pueden no estar sincronizados.
- El campo `voice` del simulador va vacío: lo llenará el nodo de NLP del robot real.

## Siguiente paso: nodo puente ROS2

Un nodo `rclpy` que (a) se suscriba a los tópicos de ROS2 de la tabla y los republique por
MQTT con este contrato, y (b) se suscriba a `…/cmd` y llame a la state machine / publique
`/cmd_vel` y `/open_close`. Debe fijar el Last Will de `…/status`.
