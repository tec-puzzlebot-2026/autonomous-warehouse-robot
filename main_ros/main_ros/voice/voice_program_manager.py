"""
voice_program_manager.py - everything needed to turn the microphone into the
/voice_command ROS 2 topic, in one node.

What it does, in order:
  1. Pre-flight checks: model file exists and loads, microphone is found and
     can record. Clear error messages if anything is missing.
  2. Opens the microphone (at 16 kHz, or at the mic's native rate and
     resamples to 16 kHz if the mic doesn't support 16 kHz).
  3. Calibrates the background noise, then runs the HMM recognizer on every
     utterance in a worker thread.
  4. Publishes:
        /voice_command  (std_msgs/String)  the recognised word, e.g. "stop"
        /voice_status   (std_msgs/String)  "starting" | "calibrating" |
                                            "listening" | "error: ..."
  5. Watchdog: if the microphone stops delivering audio (unplugged, driver
     hiccup), it re-opens it automatically.

Parameters (set them in the launch file):
  model_path    str    path to models.pkl ("" = voice/scripts/models.pkl)
  threshold     str/float  rejection threshold ("" = value chosen in training).
                       Higher = stricter, fewer false triggers.
  stop_bonus    float  makes "stop" easier to accept (safety): its threshold
                       is lowered by this amount. 0 = same as other words.
  device        int    microphone index (-1 = use mic_name or system default)
  mic_name      str    part of the microphone's name, e.g. "USB" (optional)
  cooldown      float  seconds to ignore the SAME word after publishing it
  verbose       bool   also log rejected sounds and their scores
"""
import math
import os
import queue
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import ParameterDescriptor

import sounddevice as sd
from scipy.signal import resample_poly

from .scripts import config as C
from .scripts.recognize import StreamingRecognizer


class VoiceProgramManager(Node):
    def __init__(self):
        super().__init__("voice_program_manager")

        # ---- parameters ------------------------------------------------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("threshold", "",
                               ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("stop_bonus", 0.0)
        self.declare_parameter("device", -1)
        self.declare_parameter("mic_name", "")
        self.declare_parameter("cooldown", 1.0)
        self.declare_parameter("verbose", False)

        p = lambda n: self.get_parameter(n).value
        self.model_path = p("model_path") or C.MODEL_PATH
        thr = p("threshold")
        thr = None if thr in ("", None) else float(thr)
        self.stop_bonus = float(p("stop_bonus"))
        self.cooldown = float(p("cooldown"))
        self.verbose = bool(p("verbose"))

        # ---- publishers ------------------------------------------------------
        self.cmd_pub = self.create_publisher(String, "voice_command", 10)
        self.status_pub = self.create_publisher(String, "voice_status", 10)
        self._status("starting")

        # ---- 1. pre-flight: model --------------------------------------------
        if not os.path.isfile(self.model_path):
            self._fatal(f"Model not found: {self.model_path}\n"
                        "  -> run train.py in voice/scripts/ first, build with "
                        "--symlink-install, or set the model_path parameter.")
        try:
            self.rec = StreamingRecognizer(
                model_path=self.model_path,
                threshold=thr)
        except Exception as e:
            self._fatal(f"Could not load model {self.model_path}: {e}\n"
                        "  -> retrain on this machine (hmmlearn versions must match).")
        self.get_logger().info(
            f"Model loaded: {self.model_path}\n"
            f"  words: {', '.join(self.rec.words)}\n"
            f"  threshold: {self.rec.threshold:.2f}  (stop: "
            f"{self.rec.threshold - self.stop_bonus:.2f})")

        # ---- 1b. pre-flight: microphone --------------------------------------
        self.device = self._find_device(p("device"), p("mic_name"))
        self.native_rate = self._pick_rate(self.device)

        # ---- 2-3. audio + worker thread ----------------------------------------
        self.q = queue.Queue(maxsize=500)
        self.stream = None
        self.last_audio = time.time()
        self.last_pub = {}
        self._running = True
        self._open_stream()

        self.worker = threading.Thread(target=self._process_loop, daemon=True)
        self.worker.start()

        # ---- 5. watchdog -------------------------------------------------------
        self.create_timer(2.0, self._watchdog)

    # ===================================================================== #
    # helpers
    # ===================================================================== #
    def _status(self, text):
        self.status_pub.publish(String(data=text))

    def _fatal(self, msg):
        self.get_logger().fatal(msg)
        self._status("error: " + msg.splitlines()[0])
        raise SystemExit(1)

    def _find_device(self, index, name):
        try:
            devices = sd.query_devices()
        except Exception as e:
            self._fatal(f"Audio system not available: {e}")
        inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
        if not inputs:
            self._fatal("No microphone found. Plug one in and check system sound settings.")

        listing = "\n".join(f"  [{i}] {d['name']}" for i, d in inputs)
        if index >= 0:
            if index not in [i for i, _ in inputs]:
                self._fatal(f"device={index} is not an input device. Available:\n{listing}")
            chosen = index
        elif name:
            matches = [i for i, d in inputs if name.lower() in d["name"].lower()]
            if not matches:
                self._fatal(f"No microphone matching '{name}'. Available:\n{listing}")
            chosen = matches[0]
        else:
            chosen = sd.default.device[0]
            if chosen is None or chosen < 0:
                chosen = inputs[0][0]
        self.get_logger().info(f"Input devices:\n{listing}\n"
                               f"Using [{chosen}] {devices[chosen]['name']}")
        return chosen

    def _pick_rate(self, device):
        """Prefer 16 kHz; otherwise use the mic's own rate and resample."""
        try:
            sd.check_input_settings(device=device, samplerate=C.SR, channels=1)
            return C.SR
        except Exception:
            rate = int(sd.query_devices(device)["default_samplerate"])
            self.get_logger().warn(f"Mic does not support {C.SR} Hz; recording at "
                                   f"{rate} Hz and resampling.")
            return rate

    # ===================================================================== #
    # audio
    # ===================================================================== #
    def _audio_cb(self, indata, frames, time_info, status):
        # Audio thread: copy the data and return immediately.
        if status and self.verbose:
            self.get_logger().warn(f"audio status: {status}")
        self.last_audio = time.time()
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # processing fell behind; drop a block rather than block audio

    def _open_stream(self):
        self._close_stream()
        block = int(round(C.BLOCK_SECONDS * self.native_rate))
        g = math.gcd(C.SR, self.native_rate)
        self.up, self.down = C.SR // g, self.native_rate // g
        try:
            self.stream = sd.InputStream(samplerate=self.native_rate, channels=1,
                                         blocksize=block, dtype="float32",
                                         device=self.device, callback=self._audio_cb)
            self.stream.start()
        except Exception as e:
            self.get_logger().error(f"Could not open microphone: {e}")
            self._status(f"error: could not open microphone: {e}")
            self.stream = None
            return
        self.last_audio = time.time()
        self.rec.floor, self.rec._calib = None, []          # re-calibrate noise
        self._status("calibrating")
        self.get_logger().info("Microphone open. Calibrating noise - stay quiet ~1 s...")

    def _close_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _watchdog(self):
        if self.stream is None or time.time() - self.last_audio > 2.0:
            self.get_logger().warn("No audio from microphone - re-opening it.")
            self._open_stream()

    # ===================================================================== #
    # recognition
    # ===================================================================== #
    def _process_loop(self):
        was_calibrated = False
        while self._running and rclpy.ok():
            try:
                block = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.native_rate != C.SR:
                block = resample_poly(block, self.up, self.down).astype(np.float32)

            r = self.rec.process_block(block)

            if self.rec.calibrated and not was_calibrated:
                self.get_logger().info(f"Noise floor {self.rec.floor:.1f} dB. "
                                       f"Listening -> /voice_command")
                self._status("listening")
            was_calibrated = self.rec.calibrated
            if r is not None:
                self._handle_result(r)

    def _handle_result(self, r):
        word, margin = r["word"], r["margin"]

        # Safety: accept a "stop" a bit more easily than the other words.
        if word is None and r["closest"] == "stop" and self.stop_bonus > 0 \
                and margin > self.rec.threshold - self.stop_bonus:
            word = "stop"

        if word is None:
            if self.verbose:
                self.get_logger().info(f"rejected (closest '{r['closest']}', "
                                       f"margin {margin:.2f})")
            return

        now = time.time()
        if now - self.last_pub.get(word, 0.0) < self.cooldown:
            return
        self.last_pub[word] = now
        self.cmd_pub.publish(String(data=word))
        self.get_logger().info(f"command: {word}  (margin {margin:.2f})")

    def destroy_node(self):
        self._running = False
        self._close_stream()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VoiceProgramManager()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
