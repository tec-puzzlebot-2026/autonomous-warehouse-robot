"""Colour (flat-field) calibration - ROS 2 port of the MAS507 script.

    ros2 run main_ros color_calibrate
    ros2 run main_ros color_calibrate --blur 15      # smoother gains

Reads <calib_dir>/whiteimages/*.png (>= 20 RAW images of a white wall or paper
in even light, at the SAME resolution the camera publishes) and writes
<calib_dir>/data/colorCalibration.npz with key arr_0 = HxWx3 gains.

Model (same as MAS507): for every pixel (i, j) and channel k, find the gain g
that best maps the observed values v_n over the N images to the mean image
brightness m_n, in the least-squares sense:

    g[i,j,k] = sum_n(v_n * m_n) / sum_n(v_n ** 2)

Applying corrected = raw * g removes lens shading/vignetting and the colour
cast (the IMX219 on the Jetson shows a strong pink/red tint towards the edges
without a tuned ISP file) and balances the channels to grey.

Differences from the MAS507 script: width/height are read from the images, the
triple Python loop is vectorised (seconds instead of many minutes on the Nano),
images are accumulated one at a time (low memory), exposure is sanity-checked,
and a preview of the result is written.
"""
import argparse
import sys

import cv2
import numpy as np

from main_ros.vision.camera.calibration import calib_io


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--calib-dir', default=calib_io.DEFAULT_CAMERA_CALIB_DIR)
    p.add_argument('--images', default='', help='folder (default <calib-dir>/whiteimages)')
    p.add_argument('--out', default='', help='file (default <calib-dir>/data/colorCalibration.npz)')
    p.add_argument('--blur', type=int, default=0,
                   help='Gaussian kernel (odd px) to smooth the gain map; 0 = off (MAS507)')
    if argv is None:
        argv = sys.argv[1:]
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    return p.parse_args(argv)


def compute_gains(images):
    """images: iterable of HxWx3 uint8 arrays. Returns (gains, means, stats)."""
    num = den = None
    means = []
    saturated = []
    for img in images:
        v = img.astype(np.float64)
        m = float(v.mean())
        if num is None:
            num = np.zeros_like(v)
            den = np.zeros_like(v)
        elif v.shape != num.shape:
            raise ValueError('Image size %s differs from %s' % (v.shape, num.shape))
        num += v * m
        den += v * v
        means.append(m)
        saturated.append(float(np.mean(img >= 250)))
    if num is None:
        raise ValueError('No images')
    with np.errstate(divide='ignore', invalid='ignore'):
        gains = np.where(den > 0, num / den, 1.0)
    return gains, np.array(means), np.array(saturated)


def main(argv=None):
    a = parse_args(argv)
    folder = calib_io.resolve(a.calib_dir, a.images, 'whiteimages')
    out = calib_io.resolve(a.calib_dir, a.out, calib_io.COLOR_NPZ_FILE)
    files = calib_io.list_images(folder)
    print('Found %d images in %s' % (len(files), folder))
    if len(files) < 20:
        print('WARNING: MAS507 recommends at least 20 images.')
    if not files:
        sys.exit('No images - capture some first (image_saver mode:=white).')

    def reader():
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print('  %s unreadable, skipped' % f.name)
                continue
            yield img

    try:
        gains, means, sat = compute_gains(reader())
    except ValueError as e:
        sys.exit(str(e))
    h, w = gains.shape[:2]
    print('Resolution: %dx%d  (must match the camera_node width/height)' % (w, h))
    print('Image mean brightness: min %.0f, max %.0f' % (means.min(), means.max()))
    if means.min() < 80:
        print('WARNING: some images are dark - use brighter, even light.')
    if sat.max() > 0.01:
        print('WARNING: up to %.1f%% of pixels are saturated (>=250) - gains are wrong '
              'where the image clips. Reduce exposure or light.' % (100 * sat.max()))

    if a.blur > 0:
        k = a.blur | 1
        gains = cv2.GaussianBlur(gains, (k, k), 0)
        print('Smoothed gains with %dx%d Gaussian' % (k, k))

    for c, name in enumerate('BGR'):
        g = gains[:, :, c]
        print('  gain %s: min %.3f  median %.3f  max %.3f' % (name, g.min(), np.median(g), g.max()))
    if gains.max() > 4:
        print('WARNING: gains above 4 - parts of the white images are very dark.')

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), arr_0=gains)
    # Preview: first white image before/after correction (should look flat grey)
    first = cv2.imread(str(files[0]))
    fixed = calib_io.apply_color_gains(first, gains.astype(np.float32))
    preview = out.parent / 'color_check.png'
    cv2.imwrite(str(preview), np.hstack([first, fixed]))
    print('\nSaved:\n  %s\n  %s (white image before | after)' % (out, preview))


if __name__ == '__main__':
    main()
