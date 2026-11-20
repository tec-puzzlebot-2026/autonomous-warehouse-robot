"""Global localization at start-up (no ROS): find the robot pose anywhere in the arena.

The EKF needs an initial pose. When the robot is placed at an unknown spot, the pose is
computed from the camera and the known ArUco map:

* Two or more markers (range + bearing each): the ranges put the robot on circles around the
  markers; their intersections are refined with Gauss-Newton on all measurements, and the
  solution that is consistent with the bearings (chi-square test) is kept. This does not use
  the marker orientation, the noisiest part of a PnP.
* One marker, only if it is close and unambiguous: the full pose follows from the marker pose
  in the camera frame and the marker facing direction in the map (landmark_yaw).

Measurements taken at different headings are combined by rotating the robot IN PLACE about the
wheel axle (base_link): the position does not change and each bearing is corrected with the
odometry heading change, alpha_now = alpha_k - (yaw_odo_now - yaw_odo_k).

Classes:
    GlobalLocalizer    averages detections into snapshots and solves for the pose.
    StartupLocalizer   state machine SEARCH -> COLLECT -> TRACK (or FAILED), search rotation
                       command and "lost" detection (re-localization).
"""
import math

import numpy as np

from main_ros.navigation.ekf_core import observe, range_bearing_from_base
from main_ros.vision.aruco import error_model as em
from main_ros.vision.aruco import handeye as he
from main_ros.vision.aruco import transforms as tf
from main_ros.vision.aruco.paths import wrap


def chi2_99(dof):
    """99th percentile of chi-square with `dof` degrees of freedom (Wilson-Hilferty)."""
    k = max(int(dof), 1)
    return k * (1.0 - 2.0 / (9.0 * k) + 2.3263 * math.sqrt(2.0 / (9.0 * k))) ** 3


def marker_rotation(psi):
    """R_map_marker for a marker on a vertical wall whose face points toward angle psi (rad,
    counterclockwise from +x). Marker frame (OpenCV): x right seen head-on, y up, z out of the face.
    """
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[-s, 0.0, c],
                     [c, 0.0, s],
                     [0.0, 1.0, 0.0]])


def pose_from_marker_pose(T_C_M, X, landmark_xy, psi, rel_height):
    """Robot pose (x, y, yaw) in the map from ONE marker pose in the camera frame.

    rel_height: height of the marker center above the base_link plane (m).
    """
    T_W_M = tf.make_T(marker_rotation(psi), [landmark_xy[0], landmark_xy[1], rel_height])
    return tf.T_to_planar(he.robot_from_camera(T_W_M, T_C_M, X))


def circle_intersections(p1, r1, p2, r2):
    """Intersection points of two circles; [] if they do not meet (tolerating small noise)."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    d = math.hypot(dx, dy)
    if d < 1e-3:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < -0.01:                       # circles clearly do not intersect (> 10 cm apart)
        return []
    h = math.sqrt(max(h2, 0.0))
    bx, by = p1[0] + a * dx / d, p1[1] + a * dy / d
    ox, oy = -dy / d, dx / d
    return [(bx + h * ox, by + h * oy), (bx - h * ox, by - h * oy)]


def _refine(s0, meas, landmarks, iters=25):
    """Gauss-Newton on all range-bearing measurements. Returns (cost, s, cov) or None."""
    s = np.array(s0, dtype=float)
    H = np.zeros((3, 3))
    try:
        for _ in range(iters):
            H = np.zeros((3, 3))
            g = np.zeros(3)
            for m in meas:
                zh, G = observe(landmarks[m['id']], s)
                r = np.array([m['rho'] - zh[0], wrap(m['alpha'] - zh[1])])
                W = np.linalg.inv(m['R'])
                H += G.T @ W @ G
                g += G.T @ W @ r
            step = np.linalg.solve(H + 1e-9 * np.eye(3), g)
            s = s + step
            s[2] = wrap(s[2])
            if np.linalg.norm(step[:2]) < 1e-6 and abs(step[2]) < 1e-7:
                break
        cost = 0.0
        H = np.zeros((3, 3))
        for m in meas:
            zh, G = observe(landmarks[m['id']], s)
            r = np.array([m['rho'] - zh[0], wrap(m['alpha'] - zh[1])])
            W = np.linalg.inv(m['R'])
            cost += float(r @ W @ r)
            H += G.T @ W @ G
        return cost, s, np.linalg.inv(H + 1e-12 * np.eye(3))
    except (ValueError, np.linalg.LinAlgError):
        return None


def solve_range_bearing(meas, landmarks, bounds=None, margin=0.3, max_sigma=0.3):
    """Pose from >= 2 range-bearing measurements of known landmarks.

    meas: list of dict(id, rho, alpha, R) with R the 2x2 covariance (already scaled).
    Returns dict(status, pose, cov, cost, dof, n, second_cost) where status is
      'ok'           a unique solution that passes the chi-square test,
      'ambiguous'    a second, different solution also passes the test,
      'inconsistent' no solution passes (wrong ID, bad map, outliers),
      'need_more'    the geometry is too weak (uncertainty above max_sigma).
    """
    n = len(meas)
    dof = 2 * n - 3
    out = {'status': 'inconsistent', 'pose': None, 'cov': None, 'cost': float('inf'),
           'dof': dof, 'n': n, 'second_cost': float('inf')}
    if n < 2:
        out['status'] = 'need_more'
        return out
    starts = []
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = landmarks[meas[i]['id']], landmarks[meas[j]['id']]
            for (x, y) in circle_intersections(pi, meas[i]['rho'], pj, meas[j]['rho']):
                th = [wrap(math.atan2(landmarks[m['id']][1] - y, landmarks[m['id']][0] - x)
                           - m['alpha']) for m in meas]
                theta = math.atan2(np.mean(np.sin(th)), np.mean(np.cos(th)))
                starts.append((x, y, theta))
    sols = []
    for s0 in starts:
        r = _refine(s0, meas, landmarks)
        if r is None:
            continue
        cost, s, cov = r
        if bounds is not None and bounds[1] > bounds[0] and bounds[3] > bounds[2]:
            if not (bounds[0] - margin <= s[0] <= bounds[1] + margin
                    and bounds[2] - margin <= s[1] <= bounds[3] + margin):
                continue
        if any(math.hypot(s[0] - o[1][0], s[1] - o[1][1]) < 0.05
               and abs(wrap(s[2] - o[1][2])) < 0.05 for o in sols):
            continue
        sols.append((cost, s, cov))
    if not sols:
        return out
    sols.sort(key=lambda t: t[0])
    thr = chi2_99(dof)
    cost, s, cov = sols[0]
    out.update({'pose': tuple(float(v) for v in s), 'cov': cov, 'cost': cost})
    if len(sols) > 1:
        out['second_cost'] = sols[1][0]
    if cost > thr:
        out['status'] = 'inconsistent'
    elif len(sols) > 1 and sols[1][0] <= thr:
        out['status'] = 'ambiguous'
    elif math.sqrt(max(cov[0, 0], cov[1, 1])) > max_sigma or math.sqrt(cov[2, 2]) > max_sigma:
        out['status'] = 'need_more'
    else:
        out['status'] = 'ok'
    return out


class GlobalLocalizer:
    """Averages detections into per-marker snapshots and solves for the pose."""

    def __init__(self, landmarks, psi=None, rel_heights=None, X=None, rb_cov_fn=None,
                 pose_cov_fn=None, valid_fn=None, bias=None, bounds=None, n_frames=30,
                 r_scale=2.0, allow_single=True, max_sigma=0.3):
        self.landmarks = dict(landmarks)
        self.psi = dict(psi or {})
        self.rel_heights = dict(rel_heights or {})
        self.X = X
        self.rb_cov_fn = rb_cov_fn          # (distance, view) -> 2x2
        self.pose_cov_fn = pose_cov_fn      # (distance, view, ambiguity) -> 3x3 (W frame) or None
        self.valid_fn = valid_fn or (lambda d, v: True)
        self.bias = dict(bias or {})
        self.bounds = bounds
        self.n_frames = int(n_frames)
        self.r_scale = float(r_scale)
        self.allow_single = bool(allow_single)
        self.max_sigma = float(max_sigma)
        self.reset()

    def reset(self):
        self.buffer = {}
        self.snapshots = {}

    def clear_buffers(self):
        self.buffer = {}

    def accepts(self, det):
        """Known landmark, finite distance/view, inside the validity range of the error model."""
        try:
            i = int(det.get('id', -1))
        except (TypeError, ValueError):
            return False
        d, v = det.get('distance'), det.get('view_angle_deg')
        return (i in self.landmarks and d is not None and v is not None
                and bool(self.valid_fn(d, v)))

    def add_frame(self, det):
        if not self.accepts(det):
            return None
        i = int(det['id'])
        p_base = self.X @ np.append(np.asarray(det['t'], dtype=float), 1.0)
        z = range_bearing_from_base(p_base)
        z[0] -= self.bias.get('range', 0.0)
        z[1] = wrap(z[1] - self.bias.get('bearing', 0.0))
        amb = det.get('ambiguity')
        self.buffer.setdefault(i, []).append({
            'rho': z[0], 'alpha': z[1], 't': list(det['t']), 'q': list(det['q']),
            'dist': float(det['distance']), 'view': float(det['view_angle_deg']),
            'amb': float('inf') if amb is None else float(amb)})
        return i

    def frames(self, i=None):
        if i is not None:
            return len(self.buffer.get(i, []))
        return max([len(v) for v in self.buffer.values()] or [0])

    def commit(self, i, yaw_odo):
        """Average the buffered frames of marker i into a snapshot (needs >= n_frames // 2)."""
        fr = self.buffer.get(i, [])
        if len(fr) < max(3, self.n_frames // 2):
            return False
        t = np.mean([f['t'] for f in fr], axis=0)
        q = tf.average_quaternions([f['q'] for f in fr])
        al = [f['alpha'] for f in fr]
        self.snapshots[i] = {
            'id': i, 'rho': float(np.mean([f['rho'] for f in fr])),
            'alpha': math.atan2(np.mean(np.sin(al)), np.mean(np.cos(al))),
            'dist': float(np.mean([f['dist'] for f in fr])),
            'view': float(np.mean([f['view'] for f in fr])),
            'amb_min': float(min(f['amb'] for f in fr)),
            'T_C_M': tf.make_T(tf.quat_to_matrix(*q), t), 'yaw_odo': float(yaw_odo)}
        self.buffer.pop(i, None)
        return True

    def commit_ready(self, yaw_odo):
        n = 0
        for i in list(self.buffer):
            n += int(self.commit(i, yaw_odo))
        self.buffer = {}                 # partial buffers are stale once the robot turns again
        return n

    def _single(self, snap, yaw_odo_now):
        """Pose from one marker pose, or None if it is not trustworthy."""
        i = snap['id']
        if not (self.allow_single and i in self.psi and self.pose_cov_fn is not None):
            return None
        P = self.pose_cov_fn(snap['dist'], snap['view'], snap['amb_min'])
        if P is None:
            return None
        h = self.rel_heights.get(i, 0.0)
        x, y, yaw = pose_from_marker_pose(snap['T_C_M'], self.X, self.landmarks[i], self.psi[i], h)
        # remove the measured bias (defined in the wall frame W: x toward the wall)
        psi_w = self.psi[i] + math.pi
        c, s = math.cos(psi_w), math.sin(psi_w)
        bx, by = self.bias.get('x', 0.0), self.bias.get('y', 0.0)
        x -= c * bx - s * by
        y -= s * bx + c * by
        yaw = wrap(yaw - self.bias.get('yaw', 0.0))
        yaw = wrap(yaw + (yaw_odo_now - snap['yaw_odo']))      # robot turned in place since then
        cov = em.rotate_to_map(self.r_scale * np.asarray(P, dtype=float), psi_w)
        return (x, y, yaw), cov

    def solve(self, yaw_odo_now):
        """Pose from the snapshots. Returns a dict with 'status' and, when 'ok', the pose."""
        snaps = list(self.snapshots.values())
        res = {'status': 'need_more', 'pose': None, 'cov': None, 'source': '',
               'n': len(snaps), 'cost': float('nan')}
        if len(snaps) >= 2:
            meas = []
            for s in snaps:
                R = self.r_scale * np.asarray(self.rb_cov_fn(s['dist'], s['view']), dtype=float)
                meas.append({'id': s['id'], 'rho': s['rho'],
                             'alpha': wrap(s['alpha'] - (yaw_odo_now - s['yaw_odo'])), 'R': R})
            r = solve_range_bearing(meas, self.landmarks, self.bounds, max_sigma=self.max_sigma)
            res.update({'status': r['status'], 'pose': r['pose'], 'cov': r['cov'],
                        'cost': r['cost'], 'source': 'range_bearing x%d' % len(snaps)})
            if r['status'] in ('ok', 'ambiguous', 'inconsistent'):
                return res
        if len(snaps) >= 1:
            # one marker is enough only with a trustworthy full pose (close, unambiguous)
            best = None
            for s in snaps:
                sp = self._single(s, yaw_odo_now)
                if sp is not None and (best is None or s['dist'] < best[0]):
                    best = (s['dist'], s['id'], sp)
            if best is not None and len(snaps) == 1:
                pose, cov = best[2]
                if self._inside(pose):
                    res.update({'status': 'ok', 'pose': tuple(float(v) for v in pose),
                                'cov': cov, 'source': 'marker_pose %d' % best[1]})
        return res

    def _inside(self, pose, margin=0.3):
        b = self.bounds
        if b is None or not (b[1] > b[0] and b[3] > b[2]):
            return True
        return (b[0] - margin <= pose[0] <= b[1] + margin
                and b[2] - margin <= pose[1] <= b[3] + margin)


class StartupLocalizer:
    """State machine for the start-up: SEARCH -> COLLECT -> TRACK, with FAILED and re-localization.

    * SEARCH:  rotate in place (if `rotate`) until a NEW marker is seen.
    * COLLECT: stop, wait `settle_s` until still, average `n_frames` detections, solve.
               ok -> TRACK. Otherwise go back to SEARCH (snapshots are kept unless 'inconsistent').
    * TRACK:   the EKF runs. `lost_rejects` consecutive rejected corrections -> SEARCH again.
    * FAILED:  search timeout or more than 1.25 turns without a solution (stops rotating).
    """
    SEARCH, COLLECT, TRACK, FAILED = 'SEARCH', 'COLLECT', 'TRACK', 'FAILED'

    def __init__(self, gl, rotate=False, search_speed=0.3, search_timeout_s=60.0, settle_s=0.6,
                 collect_timeout_s=8.0, lost_rejects=8, max_turns=1.25):
        self.gl = gl
        self.rotate = bool(rotate)
        self.search_speed = min(abs(float(search_speed)), 0.5)
        self.search_timeout_s = float(search_timeout_s)
        self.settle_s = float(settle_s)
        self.collect_timeout_s = float(collect_timeout_s)
        self.lost_rejects = int(lost_rejects)
        self.max_turns = float(max_turns)
        self.state = self.FAILED
        self._stop_pending = False
        self.rejects = 0
        self.t0 = self.yaw0 = self.t_collect = self.t_still = 0.0
        self.last_result = None

    def start(self, now, yaw_odo):
        self.gl.reset()
        self.state = self.SEARCH
        self.t0, self.yaw0 = now, yaw_odo
        self.rejects = 0
        self.t_still = None
        self._stop_pending = self.rotate

    def tracking(self):
        return self.state == self.TRACK

    def _to_search(self):
        self.state = self.SEARCH
        self.t_still = None

    def on_detections(self, now, dets, yaw_odo, stationary):
        """Feed one /aruco/detections message. Returns a result dict when a solve is attempted."""
        if self.state == self.SEARCH:
            for d in dets:
                if self.gl.accepts(d) and int(d['id']) not in self.gl.snapshots:
                    self.state, self.t_collect, self.t_still = self.COLLECT, now, None
                    break
        if self.state != self.COLLECT:
            return None
        if not stationary:
            self.t_still = None
            return None
        if self.t_still is None:
            self.t_still = now
        if now - self.t_still < self.settle_s:
            return None
        for d in dets:
            self.gl.add_frame(d)
        if self.gl.frames() < self.gl.n_frames:
            return None
        self.gl.commit_ready(yaw_odo)
        res = self.gl.solve(yaw_odo)
        self.last_result = res
        if res['status'] == 'ok':
            self.state = self.TRACK
            self.rejects = 0
            self._stop_pending = self.rotate
        else:
            if res['status'] == 'inconsistent':
                self.gl.reset()                 # a wrong snapshot would poison later solves
            self._to_search()
        return res

    def tick(self, now, yaw_odo):
        """Timeouts. Call at ~10 Hz."""
        if self.state == self.SEARCH:
            turned = abs(yaw_odo - self.yaw0)
            if now - self.t0 > self.search_timeout_s or turned > self.max_turns * 2 * math.pi:
                self.state = self.FAILED
                self._stop_pending = self.rotate
        elif self.state == self.COLLECT and now - self.t_collect > self.collect_timeout_s:
            self.gl.clear_buffers()
            self._to_search()

    def command(self):
        """Angular velocity to publish on /cmd_vel, or None (do not publish)."""
        if not self.rotate:
            return None
        if self.state == self.SEARCH:
            return self.search_speed
        if self.state in (self.COLLECT, self.FAILED):
            return 0.0
        if self._stop_pending:                  # just left the search: send one stop
            self._stop_pending = False
            return 0.0
        return None

    def on_update(self, now, yaw_odo, accepted):
        """Report each EKF correction. Returns True if the robot is declared lost."""
        if self.state != self.TRACK or self.lost_rejects <= 0:
            return False
        self.rejects = 0 if accepted else self.rejects + 1
        if self.rejects >= self.lost_rejects:
            self.start(now, yaw_odo)
            return True
        return False
