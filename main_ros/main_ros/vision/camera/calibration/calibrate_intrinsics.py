"""Intrinsic camera calibration with OpenCV (ROS 2 port of the MAS507 script).

    ros2 run main_ros calibrate_intrinsics                       # defaults
    ros2 run main_ros calibrate_intrinsics --cols 6 --rows 4 --square 0.03

Reads <calib_dir>/images/*.png and writes to <calib_dir>/data/:
    intrinsicCalibration.npz   mtx, dist, rvecs, tvecs (MAS507 keys) + image_size, rms
    camera_info.yaml           ROS format -> published by camera_node and used by
                               image_proc and image_processing_node for rectification
    undistort_check.png        raw vs rectified side by side, for a visual check

Differences from the MAS507 script:
  * object points are in metres (square size), so tvecs are metric
  * the cornerSubPix window adapts to the square size in pixels
  * per-image reprojection errors are reported; --max-error drops outliers
  * k3 is fixed by default (like the image_pipeline cameracalibrator); the
    Pi Camera v2 lens is mild, and a free k3 tends to overfit. Use --k3 to free it.
"""
import argparse
import sys

import cv2
import numpy as np

from main_ros.main_ros.vision.camera.calibration import calib_io

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calib-dir', default=calib_io.DEFAULT_CAMERA_CALIB_DIR)
    p.add_argument('--images', default='', help='folder (default <calib-dir>/images)')
    p.add_argument('--out-dir', default='', help='folder (default <calib-dir>/data)')
    p.add_argument('--cols', type=int, default=7,
                   help='inner corners per row (8 squares -> 7)')
    p.add_argument('--rows', type=int, default=5,
                   help='inner corners per column (6 squares -> 5)')
    p.add_argument('--square', type=float, default=0.03, help='square size in metres')
    p.add_argument('--k3', action='store_true', help='estimate k3 too')
    p.add_argument('--alpha', type=float, default=0.0,
                   help='rectified view: 0 = only valid pixels (crop), 1 = keep all pixels')
    p.add_argument('--max-error', type=float, default=0.0,
                   help='drop images with RMS reprojection error above this (px) and recalibrate')
    p.add_argument('--camera-name', default='pi_camera_v2')
    p.add_argument('--show', action='store_true', help='show each detection (needs a display)')
    # remove ROS arguments if launched via ros2 run ... --ros-args
    if argv is None:
        argv = sys.argv[1:]
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    return p.parse_args(argv)


def object_points(cols, rows, square):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return objp


def subpix_window(corners, cols):
    """Half-window ~40 % of the smallest square edge in pixels, in [3, 11]."""
    c = corners.reshape(-1, cols, 2)
    d_row = np.linalg.norm(np.diff(c, axis=1), axis=2).min()
    d_col = np.linalg.norm(np.diff(c, axis=0), axis=2).min() if c.shape[0] > 1 else d_row
    w = int(np.clip(0.4 * min(d_row, d_col), 3, 11))
    return (w, w)


def detect(images, pattern, show=False):
    names, imgpoints, size = [], [], None
    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            print('  %-22s unreadable, skipped' % path.name)
            continue
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif size != (w, h):
            print('  %-22s size %dx%d != %dx%d, skipped' % (path.name, w, h, size[0], size[1]))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, FIND_FLAGS)
        if not found:
            print('  %-22s board not found, trashed' % path.name)
            continue
        if not calib_io.corners_clear_of_border(corners, w, h):
            print('  %-22s board touches the image edge, trashed' % path.name)
            continue
        win = subpix_window(corners, pattern[0])
        corners = cv2.cornerSubPix(gray, corners, win, (-1, -1), SUBPIX_CRITERIA)
        names.append(path.name)
        imgpoints.append(corners)
        if show:
            cv2.drawChessboardCorners(img, pattern, corners, found)
            cv2.imshow('detection', img)
            cv2.waitKey(300)
    if show:
        cv2.destroyAllWindows()
    return names, imgpoints, size


def coverage(imgpoints, size, grid=4):
    """Fraction of grid x grid image cells that contain at least one corner."""
    hit = np.zeros((grid, grid), bool)
    for c in imgpoints:
        p = c.reshape(-1, 2)
        gx = np.clip((p[:, 0] / size[0] * grid).astype(int), 0, grid - 1)
        gy = np.clip((p[:, 1] / size[1] * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return hit


def calibrate(objp, imgpoints, size, flags):
    objpoints = [objp] * len(imgpoints)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None,
                                                  flags=flags)
    errs = []
    for ip, rv, tv in zip(imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rv, tv, K, D)
        diff = proj.reshape(-1, 2) - ip.reshape(-1, 2)
        errs.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return rms, K, D, rvecs, tvecs, np.array(errs)


def main(argv=None):
    a = parse_args(argv)
    image_dir = calib_io.resolve(a.calib_dir, a.images, 'images')
    out_dir = calib_io.resolve(a.calib_dir, a.out_dir, 'data')
    pattern = (a.cols, a.rows)
    images = calib_io.list_images(image_dir)
    print('Found %d images in %s, pattern %dx%d inner corners, square %.4f m'
          % (len(images), image_dir, a.cols, a.rows, a.square))
    if not images:
        sys.exit('No images - capture some first (see image_saver).')

    names, imgpoints, size = detect(images, pattern, a.show)
    print('Board detected in %d/%d images' % (len(names), len(images)))
    if len(names) < 10:
        print('WARNING: fewer than 10 usable images; aim for ~20 varied poses.')
    if len(names) < 3:
        sys.exit('Not enough detections to calibrate. Check --cols/--rows '
                 '(they count INNER corners).')

    flags = 0 if a.k3 else cv2.CALIB_FIX_K3
    objp = object_points(a.cols, a.rows, a.square)
    rms, K, D, rvecs, tvecs, errs = calibrate(objp, imgpoints, size, flags)

    if a.max_error > 0 and np.any(errs > a.max_error):
        keep = errs <= a.max_error
        for n, e in zip(names, errs):
            if e > a.max_error:
                print('  dropping %-22s error %.3f px' % (n, e))
        names = [n for n, k in zip(names, keep) if k]
        imgpoints = [p for p, k in zip(imgpoints, keep) if k]
        if len(names) < 3:
            sys.exit('Too few images left after --max-error filtering.')
        rms, K, D, rvecs, tvecs, errs = calibrate(objp, imgpoints, size, flags)

    print('\nPer-image RMS reprojection error (px):')
    for n, e in sorted(zip(names, errs), key=lambda t: -t[1]):
        print('  %-22s %.3f%s' % (n, e, '   <-- check' if e > max(2 * np.median(errs), 0.5) else ''))
    print('\nOverall RMS: %.3f px  (good: < 0.5, acceptable: < 1.0)' % rms)
    print('Image size: %dx%d' % size)
    print('K =\n%s' % np.array2string(K, precision=2, suppress_small=True))
    print('D (k1 k2 p1 p2 k3) = %s' % np.array2string(D.ravel(), precision=5))
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    vfov = 2 * np.degrees(np.arctan(size[1] / (2 * K[1, 1])))
    print('FOV: %.1f x %.1f deg (Pi Camera v2 spec: 62.2 x 48.8)' % (hfov, vfov))
    cov = coverage(imgpoints, size)
    print('Image coverage by corners: %d/%d cells' % (cov.sum(), cov.size))
    if cov.sum() < cov.size:
        print('  Missing cells (row, col): %s -> add images with the board there '
              '(especially near the edges/corners)' % [tuple(x) for x in np.argwhere(~cov)])

    newK = calib_io.new_camera_matrix(K, D, size, a.alpha)
    P = np.hstack([newK, np.zeros((3, 1))])

    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / 'intrinsicCalibration.npz'
    np.savez(str(npz), mtx=K, dist=D, rvecs=np.array(rvecs), tvecs=np.array(tvecs),
             image_size=np.array(size), rms=rms, newcameramtx=newK)
    yml = calib_io.write_camera_info_yaml(out_dir / 'camera_info.yaml', size[0], size[1],
                                          K, D, np.eye(3), P, camera_name=a.camera_name)

    # Visual check on the worst-error image (distortion most visible at edges)
    worst = names[int(np.argmax(errs))]
    raw = cv2.imread(str(image_dir / worst))
    m1, m2, _ = calib_io.build_rectify_maps(K, D, size, P=P)
    rect = cv2.remap(raw, m1, m2, cv2.INTER_LINEAR)
    check = np.hstack([raw, rect])
    cv2.putText(check, 'raw', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(check, 'rectified', (size[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / 'undistort_check.png'), check)

    print('\nSaved:\n  %s\n  %s\n  %s' % (npz, yml, out_dir / 'undistort_check.png'))
    print('Restart camera.launch.py / vision.launch.py so camera_info is reloaded.')


if __name__ == '__main__':
    main()
