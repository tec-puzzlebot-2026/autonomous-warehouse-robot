#!/usr/bin/env python3
"""
near_box.py - find THE near box (10-30 cm in front of the robot) fast enough for a Jetson Nano.

Pipeline per frame (no SAM 2 at runtime - SAM 2 is only the offline labeler):
  1. crop the ROI with numpy slicing (zero cost) and scale it to the network size
  2. YOLO11n-seg (or YOLO11n detect) on the small strip: TensorRT FP16 engine, ONNX Runtime
     or Ultralytics
  3. class-agnostic NMS -> drop boxes smaller than min_size_px (far boxes) -> keep the
     largest (stays on the previous target unless another box is clearly bigger)
  4. mask ONLY for that one box: 32 coefficients x prototype grid (a 32 x 3840 product for
     a 320x192 input), instead of masks for every box in the image

Same file is used by the ROS 2 node (box_target_node.py) and as a stand-alone benchmark:

    python3 near_box.py --model best.engine --source 0                  # CSI/USB camera index
    python3 near_box.py --model best.onnx --source clip.mp4 --show
    python3 near_box.py --model best.engine --source frames_dir/ --bench 300

Settings (ROI, network size, thresholds, class names) come from near_box.yaml, which
train_yolo_seg.py writes next to the exported model. Python 3.6+ compatible.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

DEFAULTS = {
    "classes": ["box_plain", "box_drinks"],
    "frame": [640, 480],
    "roi": [0, 96, 640, 480],
    "net": [320, 192],
    "min_size_px": 100,
    "conf": 0.35,
    "nms_iou": 0.5,
    "max_det": 10,
    "sticky": 0.8,
}


def load_settings(path=None, model_path=None):
    """near_box.yaml given explicitly, or next to the model, or the defaults."""
    s = dict(DEFAULTS)
    if path is None and model_path:
        cand = os.path.join(os.path.dirname(os.path.abspath(model_path)), "near_box.yaml")
        path = cand if os.path.exists(cand) else None
    if path:
        import yaml
        with open(os.path.expanduser(path)) as f:
            s.update(yaml.safe_load(f) or {})
    return s


def roi_for_frame(s, w, h):
    """ROI scaled to the actual frame size (settings are for s['frame'])."""
    fw, fh = s["frame"]
    x0, y0, x1, y1 = s["roi"]
    if (w, h) != (fw, fh):
        x0, x1 = int(round(x0 * w / fw)), int(round(x1 * w / fw))
        y0, y1 = int(round(y0 * h / fh)), int(round(y1 * h / fh))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


# ----------------------------------------------------------------------------
# Pre-processing (same as Ultralytics LetterBox: scale, centre, pad with 114)
# ----------------------------------------------------------------------------
def preprocess(frame, s, fp16=False):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = roi_for_frame(s, W, H)
    crop = frame[y0:y1, x0:x1]                       # a view: no copy
    nw, nh = s["net"]
    ch, cw = crop.shape[:2]
    r = min(nw / cw, nh / ch)
    rw, rh = int(round(cw * r)), int(round(ch * r))
    img = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rw, rh) != (cw, ch) else crop
    px, py = (nw - rw) / 2.0, (nh - rh) / 2.0
    if rw != nw or rh != nh:
        top, left = int(round(py - 0.1)), int(round(px - 0.1))
        img = cv2.copyMakeBorder(img, top, nh - rh - top, left, nw - rw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        px, py = float(left), float(top)
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None]  # BGR->RGB, HWC->NCHW
    blob = np.ascontiguousarray(blob, dtype=np.float16 if fp16 else np.float32) / 255.0
    # meta maps network pixels back to frame pixels: frame = (net - pad) / r + roi offset
    meta = {"r": r, "pad": (px, py), "off": (x0, y0), "roi": (x0, y0, x1, y1), "frame": (W, H)}
    return blob, meta


def net_to_frame(xyxy, meta):
    px, py = meta["pad"]
    ox, oy = meta["off"]
    b = np.asarray(xyxy, np.float32).copy()
    b[..., [0, 2]] = (b[..., [0, 2]] - px) / meta["r"] + ox
    b[..., [1, 3]] = (b[..., [1, 3]] - py) / meta["r"] + oy
    x0, y0, x1, y1 = meta["roi"]
    b[..., [0, 2]] = b[..., [0, 2]].clip(x0, x1)
    b[..., [1, 3]] = b[..., [1, 3]].clip(y0, y1)
    return b


# ----------------------------------------------------------------------------
# Post-processing of the raw YOLO11 outputs
#   out0: (1, 4 + nc + nm, N)  rows = cx, cy, w, h, class scores, mask coefficients
#   out1: (1, nm, H/4, W/4)    mask prototypes (segmentation models only)
# ----------------------------------------------------------------------------
def decode(out0, nc, s, meta):
    """-> list of detections in FRAME pixels, after conf + class-agnostic NMS."""
    p = np.asarray(out0, np.float32)[0].T                     # (N, 4+nc+nm)
    scores = p[:, 4:4 + nc]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(p)), cls]
    keep = conf >= s["conf"]
    p, cls, conf = p[keep], cls[keep], conf[keep]
    if not len(p):
        return []
    xywh = p[:, :4]
    xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    tlwh = np.concatenate([xyxy[:, :2], xywh[:, 2:]], 1)
    idx = cv2.dnn.NMSBoxes(tlwh.tolist(), conf.tolist(), float(s["conf"]), float(s["nms_iou"]))
    idx = np.array(idx, dtype=int).reshape(-1)[: int(s["max_det"])]
    if not len(idx):
        return []
    frame_xyxy = net_to_frame(xyxy[idx], meta)
    dets = []
    for k, i in enumerate(idx):
        dets.append({"cls": int(cls[i]), "conf": float(conf[i]), "box": frame_xyxy[k].tolist(),
                     "net_box": xyxy[i].tolist(), "coef": p[i, 4 + nc:]})
    return dets


def box_size(b):
    return float(np.sqrt(max(b[2] - b[0], 0) * max(b[3] - b[1], 0)))


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def select_target(dets, min_size, prev_box=None, sticky=1.0):
    """Drop far (small) boxes, keep the largest. With prev_box, stay on the box that
    overlaps it unless the largest is more than 1/sticky times bigger."""
    near = [d for d in dets if box_size(d["box"]) >= min_size]
    if not near:
        return None
    area = lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
    best = max(near, key=area)
    if prev_box is not None and sticky < 1.0:
        same = max(near, key=lambda d: box_iou(d["box"], prev_box))
        if box_iou(same["box"], prev_box) > 0.3 and area(same) >= sticky * area(best):
            return same
    return best


def target_mask(det, protos, meta, net_wh):
    """Mask of ONE detection (as Ultralytics process_mask, retina=False) -> polygon in frame px."""
    pr = np.asarray(protos, np.float32)[0]                     # (nm, mh, mw)
    nm, mh, mw = pr.shape
    logit = (det["coef"].astype(np.float32) @ pr.reshape(nm, -1)).reshape(mh, mw)
    nw, nh = net_wh
    x1, y1, x2, y2 = det["net_box"]
    sx, sy = mw / float(nw), mh / float(nh)
    c = np.arange(mw)[None, :]
    r = np.arange(mh)[:, None]
    inside = (c >= x1 * sx) & (c < x2 * sx) & (r >= y1 * sy) & (r < y2 * sy)
    logit = np.where(inside, logit, 0.0).astype(np.float32)
    up = cv2.resize(logit, (nw, nh), interpolation=cv2.INTER_LINEAR) > 0
    # network pixels -> frame pixels: only the box region matters
    px, py = meta["pad"]
    ix1, iy1 = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0)
    ix2, iy2 = min(int(np.ceil(x2)), nw), min(int(np.ceil(y2)), nh)
    sub = up[iy1:iy2, ix1:ix2].astype(np.uint8)
    if sub.size == 0 or not sub.any():
        return None
    # A box is convex: printed stripes, tape or glare can split its mask into pieces, so keep
    # every real piece (>= 5 % of the largest) and return the convex hull of all of them.
    cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cv2.contourArea(c) for c in cnts)
    pts = np.vstack([c.reshape(-1, 2) for c in cnts if cv2.contourArea(c) >= 0.05 * big])
    cnt = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
    cnt[:, 0] = (cnt[:, 0] + ix1 - px) / meta["r"] + meta["off"][0]
    cnt[:, 1] = (cnt[:, 1] + iy1 - py) / meta["r"] + meta["off"][1]
    return cnt


def describe(det, poly, meta, names):
    """Everything the robot needs about the target, in frame pixels."""
    x1, y1, x2, y2 = det["box"]
    out = {"cls": det["cls"], "name": names[det["cls"]] if det["cls"] < len(names) else str(det["cls"]),
           "conf": det["conf"], "box": [x1, y1, x2, y2], "size_px": box_size(det["box"]),
           "center": [(x1 + x2) / 2, (y1 + y2) / 2],
           "touches": {"bottom": y2 >= meta["frame"][1] - 2, "left": x1 <= meta["roi"][0] + 1,
                       "right": x2 >= meta["roi"][2] - 1, "top": y1 <= meta["roi"][1] + 1},
           "polygon": None, "rect": None}
    if poly is not None and len(poly) >= 3:
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(poly)
        m = cv2.moments(poly)
        if m["m00"] > 0:
            out["center"] = [m["m10"] / m["m00"], m["m01"] / m["m00"]]
        out["polygon"] = poly
        out["rect"] = {"center": [cx, cy], "size": [rw, rh], "angle_deg": ang}
    return out


# ----------------------------------------------------------------------------
# Backends: each returns the raw outputs [out0, (out1)] for a preprocessed blob
# ----------------------------------------------------------------------------
class OrtBackend:
    def __init__(self, path):
        import onnxruntime as ort
        prov = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
                if p in ort.get_available_providers()] + ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        self.fp16 = "float16" in self.sess.get_inputs()[0].type
        self.outs = sorted(o.name for o in self.sess.get_outputs())
        shp = self.sess.get_inputs()[0].shape
        self.net = [shp[3], shp[2]] if all(isinstance(v, int) for v in shp[2:]) else None
        self.name = "onnxruntime " + self.sess.get_providers()[0]

    def __call__(self, blob):
        return self.sess.run(self.outs, {self.inp: blob})


class TrtBackend:
    """TensorRT engine (JetPack 4.6 = TensorRT 8.2; also TensorRT 10). GPU buffers via torch.
    Accepts engines from trtexec and from 'yolo export format=engine' (metadata header)."""

    def __init__(self, path):
        import tensorrt as trt
        import torch
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(path, "rb") as f:
            data = f.read()
        try:                                         # Ultralytics header: 4-byte length + JSON
            n = int.from_bytes(data[:4], byteorder="little")
            json.loads(data[4:4 + n].decode("utf-8"))
            data = data[4 + n:]
        except Exception:
            pass
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(data)
        self.ctx = self.engine.create_execution_context()
        self.v10 = hasattr(self.engine, "num_io_tensors")
        tdt = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.inputs, outs = [], []
        if self.v10:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            for nm in names:
                t = torch.empty(tuple(self.engine.get_tensor_shape(nm)),
                                dtype=tdt[self.engine.get_tensor_dtype(nm)], device="cuda")
                self.ctx.set_tensor_address(nm, t.data_ptr())
                is_in = self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT
                (self.inputs if is_in else outs).append((nm, t))
            self.stream = torch.cuda.Stream()
        else:
            self.bindings = []
            for i in range(self.engine.num_bindings):
                nm = self.engine.get_binding_name(i)
                t = torch.empty(tuple(self.engine.get_binding_shape(i)),
                                dtype=tdt[self.engine.get_binding_dtype(i)], device="cuda")
                self.bindings.append(int(t.data_ptr()))
                (self.inputs if self.engine.binding_is_input(i) else outs).append((nm, t))
        self.outs = [t for _, t in sorted(outs)]       # output0, output1
        self.inp = self.inputs[0][1]
        self.fp16 = self.inp.dtype == torch.float16
        self.net = [int(self.inp.shape[3]), int(self.inp.shape[2])]
        self.name = "tensorrt " + trt.__version__ + (" fp16-io" if self.fp16 else "")

    def __call__(self, blob):
        torch = self.torch
        self.inp.copy_(torch.from_numpy(blob))
        if self.v10:
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
        else:
            self.ctx.execute_v2(self.bindings)
        return [o.float().cpu().numpy() for o in self.outs]


def make_backend(path, kind="auto"):
    ext = os.path.splitext(path)[1].lower()
    if kind == "auto":
        kind = {".engine": "tensorrt", ".trt": "tensorrt", ".onnx": "onnxruntime"}.get(ext, "ultralytics")
    if kind == "tensorrt":
        return TrtBackend(path)
    if kind == "onnxruntime":
        return OrtBackend(path)
    if kind == "ultralytics":
        return UltralyticsBackend(path)
    raise ValueError("unknown backend " + kind)


class UltralyticsBackend:
    """Easiest path if ultralytics runs on the Jetson: loads .pt/.onnx/.engine itself.
    Masks are computed for every kept box (max_det small, so still cheap)."""
    raw = False

    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(path)
        self.name = "ultralytics"
        self.net = None
        self.fp16 = False


# ----------------------------------------------------------------------------
class NearBoxDetector:
    """detect(frame BGR) -> target dict or None.  self.timing has the last stage times (ms)."""

    def __init__(self, model, settings=None, backend="auto"):
        self.s = load_settings(settings, model)
        self.be = make_backend(model, backend)
        if getattr(self.be, "net", None) and list(self.be.net) != list(self.s["net"]):
            raise SystemExit("Model input is %s but near_box.yaml says net %s - re-export or fix "
                             "the yaml" % (self.be.net, self.s["net"]))
        self.prev = None
        self.timing = {}
        self.n_det = 0

    def detect(self, frame):
        t0 = time.perf_counter()
        if isinstance(self.be, UltralyticsBackend):
            target = self._detect_ultralytics(frame, t0)
        else:
            blob, meta = preprocess(frame, self.s, self.be.fp16)
            t1 = time.perf_counter()
            outs = self.be(blob)
            t2 = time.perf_counter()
            nc = len(self.s["classes"])
            dets = decode(outs[0], nc, self.s, meta)
            self.n_det = len(dets)
            det = select_target(dets, self.s["min_size_px"], self.prev, self.s["sticky"])
            target = None
            if det is not None:
                poly = target_mask(det, outs[1], meta, self.s["net"]) if len(outs) > 1 else None
                target = describe(det, poly, meta, self.s["classes"])
            t3 = time.perf_counter()
            self.timing = {"pre": 1e3 * (t1 - t0), "infer": 1e3 * (t2 - t1), "post": 1e3 * (t3 - t2)}
        self.prev = target["box"] if target else None
        return target

    def _detect_ultralytics(self, frame, t0):
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = roi_for_frame(self.s, W, H)
        crop = frame[y0:y1, x0:x1]
        nw, nh = self.s["net"]
        r = self.be.model.predict(crop, imgsz=(nh, nw), conf=self.s["conf"], iou=self.s["nms_iou"],
                                  agnostic_nms=True, max_det=int(self.s["max_det"]), verbose=False)[0]
        t2 = time.perf_counter()
        boxes = r.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0]
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
        dets = [{"cls": int(c), "conf": float(cf), "box": b.tolist(), "poly": p}
                for c, cf, b, p in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), boxes, polys)]
        self.n_det = len(dets)
        det = select_target(dets, self.s["min_size_px"], self.prev, self.s["sticky"])
        meta = {"frame": (W, H), "roi": (x0, y0, x1, y1)}
        target = None
        if det is not None:
            poly = None
            if det["poly"] is not None and len(det["poly"]) >= 3:   # same convex outline as the raw path
                poly = cv2.convexHull(det["poly"].astype(np.float32)).reshape(-1, 2) + [x0, y0]
            target = describe(det, None if poly is None else poly.astype(np.float32), meta,
                              self.s["classes"])
        self.timing = {"pre+infer": 1e3 * (t2 - t0), "post": 1e3 * (time.perf_counter() - t2)}
        return target


def draw(frame, target, s, colors=((0, 200, 255), (60, 220, 60))):
    out = frame.copy()
    H, W = out.shape[:2]
    x0, y0, x1, y1 = roi_for_frame(s, W, H)
    dim = np.ones_like(out, dtype=bool)
    dim[y0:y1, x0:x1] = False
    out[dim] = (out[dim] * 0.4).astype(np.uint8)              # outside ROI = darkened
    cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
    if target:
        col = colors[target["cls"] % len(colors)]
        b = [int(v) for v in target["box"]]
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), col, 2)
        if target["polygon"] is not None:
            cv2.polylines(out, [target["polygon"].astype(np.int32)], True, col, 2)
        cx, cy = [int(v) for v in target["center"]]
        cv2.circle(out, (cx, cy), 4, col, -1)
        cv2.putText(out, "%s %.2f  %dpx" % (target["name"], target["conf"], target["size_px"]),
                    (b[0], max(14, b[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return out


# ----------------------------------------------------------------------------
def frames_from(source):
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "*.jpg")) + glob.glob(os.path.join(source, "*.png")))
        for f in files:
            yield cv2.imread(f)
        return
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif source == "csi":                          # Pi Cam v2 on the Nano (same pipeline idea as camera_node)
        cap = cv2.VideoCapture(
            "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1 ! "
            "nvvidconv flip-method=2 ! video/x-raw,width=640,height=480,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=true max-buffers=1", cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".engine (TensorRT), .onnx or .pt")
    ap.add_argument("--settings", help="near_box.yaml (default: next to the model)")
    ap.add_argument("--backend", default="auto", choices=["auto", "tensorrt", "onnxruntime", "ultralytics"])
    ap.add_argument("--source", default="csi", help="csi, camera index, video file or image folder")
    ap.add_argument("--bench", type=int, default=0, help="frames to time (first 20 are warm-up)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", help="write an annotated video here")
    args = ap.parse_args()

    det = NearBoxDetector(args.model, args.settings, args.backend)
    print("backend %s | roi %s -> net %s | min size %d px" %
          (det.be.name, det.s["roi"], det.s["net"], det.s["min_size_px"]))
    times, n, writer, found = [], 0, None, 0
    for frame in frames_from(args.source):
        if frame is None:
            continue
        t = time.perf_counter()
        target = det.detect(frame)
        dt = 1e3 * (time.perf_counter() - t)
        n += 1
        found += target is not None
        if n > 20:
            times.append([dt] + list(det.timing.values()))
        if args.show or args.save:
            vis = draw(frame, target, det.s)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if args.show:
                cv2.imshow("near box", vis)
                if cv2.waitKey(1) == 27:
                    break
        elif not args.bench and n % 30 == 0:
            print("frame %d: %s" % (n, "none" if target is None else
                                    "%s %.2f %dpx" % (target["name"], target["conf"], target["size_px"])))
        if args.bench and n >= args.bench + 20:
            break
    if writer:
        writer.release()
    if times:
        t = np.array(times)
        keys = ["total"] + list(det.timing.keys())
        print("\n%d frames timed (target found in %d of %d)" % (len(t), found, n))
        for k, col in zip(keys, t.T):
            print("  %-10s mean %6.2f ms   p95 %6.2f ms" % (k, col.mean(), np.percentile(col, 95)))
        print("  -> %.1f FPS (detector only, camera and ROS not included)" % (1e3 / t[:, 0].mean()))


if __name__ == "__main__":
    main()