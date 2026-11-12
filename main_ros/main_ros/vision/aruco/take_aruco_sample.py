"""Step C ("Take sample"): records one sample of the grid.

Place the robot at a known point, with the wheel axle over the mark, and run:

    ros2 run main_ros take_aruco_sample --d 0.60 --y -0.20 --yaw 30

        --d    distance from the wall (ArUco plane) to the wheel axle [m]
        --y    lateral offset of the axle [m]  (+ = robot to the left of the ArUco)
        --yaw  robot rotation about its axis [degrees] (+ = counterclockwise, 0 = faces the wall)
        --id   ArUco ID (its own height and size from aruco_localization.yaml are used)

    ros2 run main_ros take_aruco_sample --undo      # deletes the last sample

Averages N detections of the ArUco (frames_per_sample), reads the real camera mount
from the static TF base_link -> camera_optical_frame and appends a row to
<calib_dir>/aruco/samples.csv. Immediately prints the error of that sample.
"""
import argparse
import json
import math
import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from main_ros.vision.aruco import experiment_io as xio
from main_ros.vision.aruco import handeye as he
from main_ros.vision.aruco import transforms as tf


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--d', type=float, help='distance wall -> wheel axle [m]')
    p.add_argument('--y', type=float, default=0.0, help='lateral [m], + left')
    p.add_argument('--yaw', type=float, default=0.0, help='degrees, + counterclockwise')
    p.add_argument('--id', type=int, default=None, help='ArUco ID (default: the one in the config)')
    p.add_argument('--frames', type=int, default=None)
    p.add_argument('--timeout', type=float, default=15.0, help='s to collect the frames')
    p.add_argument('--csv', default=str(xio.default_samples_path()))
    p.add_argument('--config', default=None)
    p.add_argument('--note', default='')
    p.add_argument('--undo', action='store_true', help='deletes the last sample and exits')
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    a = p.parse_args(argv)
    if not a.undo and a.d is None:
        p.error('--d is required')
    if not a.undo and a.d <= 0:
        p.error('--d must be > 0')
    return a


class Sampler(Node):
    def __init__(self, marker_id):
        super().__init__('take_aruco_sample')
        self.marker_id = marker_id
        self.dets = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(String, '/aruco/detections', self.on_det, 10)

    def on_det(self, msg):
        try:
            for d in json.loads(msg.data):
                if d.get('id') == self.marker_id:
                    self.dets.append(d)
        except (ValueError, TypeError) as e:
            self.get_logger().warn('Invalid JSON in /aruco/detections: %s' % e)

    def mount_from_tf(self, timeout=3.0):
        """Camera mount as published by the static TF (or None)."""
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tr = self.tf_buffer.lookup_transform('base_link', 'camera_optical_frame',
                                                     Time(), Duration(seconds=0.1))
            except TransformException:
                continue
            q = tr.transform.rotation
            t = tr.transform.translation
            X = tf.make_T(tf.quat_to_matrix(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])
            return he.X_to_mount(X)
        return None


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    a = parse_args(argv)
    if a.undo:
        print('Last sample deleted' if xio.remove_last_row(a.csv) else 'Nothing to delete')
        return
    cfg = xio.load_config(a.config)
    exp = cfg['experiment']
    marker_id = a.id if a.id is not None else int(exp.get('marker_id', 21))
    n_frames = a.frames or int(exp.get('frames_per_sample', 30))
    h_marker = xio.marker_z(exp, marker_id)   # center of THIS ArUco above base_link

    rclpy.init()
    node = Sampler(marker_id)
    try:
        mount = node.mount_from_tf()
        if mount is None:
            mount = cfg['mount']
            print('WARNING: no TF base_link->camera_optical_frame; using the mount from '
                  'the config %s' % (cfg['path'],))
        X = he.mount_to_X(*mount)
        print('Collecting %d detections of ArUco %d ...' % (n_frames, marker_id))
        end = time.time() + a.timeout
        while len(node.dets) < n_frames and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        dets = node.dets[:n_frames]
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if len(dets) < max(3, n_frames // 3):
        sys.exit('Only %d detections in %.0f s: is ArUco %d visible? Is aruco_detector running?'
                 % (len(dets), a.timeout, marker_id))

    row = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'marker_id': marker_id,
           'gt_d': a.d, 'gt_y': a.y, 'gt_yaw_deg': a.yaw, 'note': a.note}
    row.update(xio.summarize_frames(dets, X, h_marker))
    for k, v in zip(('mount_x', 'mount_y', 'mount_z', 'mount_roll', 'mount_pitch',
                     'mount_yaw'), mount):
        row[k] = v
    try:
        row['sample'] = len(xio.read_rows(a.csv))
    except FileNotFoundError:
        row['sample'] = 0
    xio.append_row(a.csv, row)

    ev = xio.evaluate(row, X, h_marker)
    print('\nSample %d saved in %s' % (row['sample'], a.csv))
    print('              real        camera      error     jitter(1σ)')
    print('  X [m]   %9.3f   %9.3f   %+8.3f   %8.4f' % (ev['x_real'], ev['x_cam'],
                                                       ev['err_x'], row['std_x']))
    print('  Y [m]   %9.3f   %9.3f   %+8.3f   %8.4f' % (ev['y_real'], ev['y_cam'],
                                                       ev['err_y'], row['std_y']))
    print('  θ [°]   %9.2f   %9.2f   %+8.2f   %8.3f' % (
        math.degrees(ev['yaw_real']), math.degrees(ev['yaw_cam']),
        math.degrees(ev['err_yaw']), row['std_yaw_deg']))
    print('  Z [m]   %9.3f   %9.3f   %+8.3f   (depth in the optical frame)' % (
        ev['z_real'], ev['z_cam'], ev['err_z']))
    print('  distance %.2f m, view angle %.1f°, reprojection %.2f px, min ambiguity %.2f'
          % (row['distance'], row['view_angle_deg'], row['reproj_err_px'], row['ambiguity_min']))
    thr = float(exp.get('ambiguity_threshold', 1.5))
    if math.isfinite(row['ambiguity_min']) and row['ambiguity_min'] < thr:
        print('  WARNING: ambiguous pose (ratio < %.1f): the yaw may jump between 2 solutions.' % thr)


if __name__ == '__main__':
    main()
