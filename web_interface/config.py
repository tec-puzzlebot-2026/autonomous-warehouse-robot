"""Configuración compartida entre la interfaz web, el simulador y (después) el nodo puente ROS2.

Todo se puede cambiar con variables de entorno, sin tocar el código:

    MQTT_HOST     (default: localhost)
    MQTT_PORT     (default: 1883)
    MQTT_USER     (opcional)
    MQTT_PASS     (opcional)
    TOPIC_PREFIX  (default: puzzlebot)

IMPORTANTE: si usas un broker público (broker.hivemq.com, test.mosquitto.org),
cualquiera puede leer y escribir en tus tópicos. Cambia TOPIC_PREFIX por algo
único (por ejemplo puzzlebot_a01737354_x7k2) y no lo uses para controlar un
robot real sin autenticación.
"""
import os

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER") or None
MQTT_PASS = os.getenv("MQTT_PASS") or None
PREFIX = os.getenv("TOPIC_PREFIX", "puzzlebot")

# ---------------------------------------------------------------------------
# Contrato de tópicos
# Robot -> UI (telemetría). Todos los payloads son JSON salvo la cámara (JPEG).
# ---------------------------------------------------------------------------
T_STATUS = f"{PREFIX}/status"          # "online" | "offline" (retenido, con Last Will)
T_STATE = f"{PREFIX}/state"            # estado de la state machine (retenido)
T_POSE = f"{PREFIX}/pose"              # pose estimada (Kalman)
T_VEL = f"{PREFIX}/velocity"           # cmd_vel y velocidades de encoders
T_PERCEPTION = f"{PREFIX}/perception"  # YOLO / ArUco / error de alineación
T_GRIPPER = f"{PREFIX}/gripper"        # estado del gripper (retenido)
T_SYSTEM = f"{PREFIX}/system"          # heartbeat, batería, último comando de voz
T_LOG = f"{PREFIX}/log"                # eventos de texto
T_PATH = f"{PREFIX}/path"              # trayectoria planeada (lista de puntos)
T_CAMERA = f"{PREFIX}/camera"          # imagen JPEG binaria

# UI -> Robot (comandos)
T_CMD = f"{PREFIX}/cmd"                # {"cmd": "...", ...}  (QoS 1)

# Comandos válidos en T_CMD:
#   {"cmd": "start"}
#   {"cmd": "stop"}
#   {"cmd": "reset"}
#   {"cmd": "estop"}                       paro de emergencia
#   {"cmd": "mode", "value": "auto"|"manual"}
#   {"cmd": "goal", "x": float, "y": float}
#   {"cmd": "teleop", "v": float, "w": float}   solo en modo manual
#   {"cmd": "gripper", "value": "open"|"close"}

STATES = [
    "IDLE",
    "SEARCH_TRAILER",
    "ALIGN",
    "GRAB_PALLET",
    "TRANSPORT",
    "RELEASE",
    "DONE",
    "ESTOP",
]
