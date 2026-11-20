"""Final arena test: are Q_k and R_k correct? (EKF consistency)

    ros2 run main_ros ekf_consistency_analysis                 # latest ekf_*.csv
    ros2 run main_ros ekf_consistency_analysis --log ~/main_ros_calibration/ekf/ekf_X.csv

Two standard statistical tests:

* NIS (Normalized Innovation Squared) = νᵀ Z⁻¹ ν at each correction. If R_k and the
  Σ̂_k coming from the prediction are correct, it follows χ² with 2 DOF: mean 2 and
  95 % of the values below 5.99. Mean >> 2 = the filter is optimistic (R or Q
  too small); mean << 2 = pessimistic (R too large, the camera is under-used).
* NEES at the checkpoints (real pose measured) = eᵀ Σ⁻¹ e, χ² with 3 DOF (mean 3).
  The EKF is compared with dead reckoning: the EKF must have bounded error and dead
  reckoning growing error.
"""
import argparse
import csv
import glob
import math
import sys

import numpy as np

from main_ros.vision.aruco.paths import DEFAULT_CALIB_DIR, expand

CHI2 = {2: (5.991, 0.103), 3: (7.815, 0.352)}       # (95 % upper, 5 % lower)


def read(path):
    with open(str(path), newline='') as f:
        rows = list(csv.DictReader(f))

    def num(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError, KeyError):
            return float('nan')
    return rows, num


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--log', default=None)
    a = p.parse_args(argv)
    path = a.log
    if path is None:
        logs = sorted(glob.glob(str(expand(DEFAULT_CALIB_DIR) / 'ekf' / 'ekf_*.csv')))
        if not logs:
            sys.exit('No EKF logs; launch ekf_localization.launch.py first.')
        path = logs[-1]
    rows, num = read(expand(path))
    out = []

    def rep(text=''):
        print(text)
        out.append(text)

    rep('# EKF consistency in the arena')
    rep('')
    rep('Log: %s' % path)
    inits = [r for r in rows if r['kind'] == 'init']
    if inits:
        rep('')
        rep('## Start-up / re-localization')
        rep('')
        rep('| t (s) | result | source | markers | cost | pose x, y (m); θ (°) | σ x, y (cm); θ (°) |')
        rep('|---|---|---|---|---|---|---|')
        for i, r in enumerate(inits):
            ok = r.get('source') not in ('failed', '', None)
            rep('| %s | %s | %s | %s | %s | %s | %s |' % (
                r['t'], 'ok' if ok else 'FAILED', r.get('source') or '-', r.get('n_markers') or '-',
                r.get('cost') or '-',
                '%.3f, %.3f; %.1f' % (num(r, 'x'), num(r, 'y'), math.degrees(num(r, 'theta')))
                if ok else '-',
                '%.1f, %.1f; %.1f' % (100 * num(r, 'sx'), 100 * num(r, 'sy'),
                                      math.degrees(num(r, 'stheta'))) if ok else '-'))
        lost = [r for r in rows if r['kind'] == 'lost']
        if lost:
            rep('')
            rep('Robot lost %d time(s) (consecutive rejected corrections) at t = %s.' % (
                len(lost), ', '.join(r['t'] for r in lost)))
        # start-up error: first checkpoint after each init (publish it right after the start)
        errs = []
        for r in inits:
            if r.get('source') in ('failed', '', None):
                continue
            t0 = num(r, 't')
            nxt = [c for c in rows if c['kind'] == 'checkpoint' and num(c, 't') >= t0]
            if nxt:
                c = nxt[0]
                errs.append((math.hypot(num(c, 'gt_x') - num(r, 'x'), num(c, 'gt_y') - num(r, 'y')),
                             abs(math.degrees(math.atan2(
                                 math.sin(num(c, 'gt_theta') - num(r, 'theta')),
                                 math.cos(num(c, 'gt_theta') - num(r, 'theta'))))),
                             num(c, 'nees')))
        if errs:
            rep('')
            rep('Start-up error against the first checkpoint after each start (publish /ekf/checkpoint '
                'with the tape-measured pose while the robot is still): ')
            rep('')
            rep('| # | position error (cm) | heading error (°) | NEES |')
            rep('|---|---|---|---|')
            for k, (ep, et, nn) in enumerate(errs):
                rep('| %d | %.1f | %.1f | %.1f |' % (k, 100 * ep, et, nn))
            rep('')
            rep('Target: error below ~5 cm and ~3° and NEES mostly below 7.8. Over many starts, '
                'mean NEES >> 3 = raise `init_r_scale`; consistently small errors with NEES << 3 = '
                'you can lower it.')

    up = [r for r in rows if r['kind'] == 'update']
    if up:
        nis = np.array([num(r, 'nis') for r in up])
        acc = np.array([num(r, 'accepted') for r in up]) > 0.5
        dist = np.array([num(r, 'distance') for r in up])
        hi, lo = CHI2[2]
        rep('')
        rep('## NIS of the corrections (χ² with 2 DOF)')
        rep('')
        rep('Corrections: %d, accepted %d (%.0f %%). Mean NIS (accepted) = %.2f '
            '(consistent ≈ 2); %.0f %% below %.2f (expected 95 %%).' % (
                len(up), acc.sum(), 100 * acc.mean(), float(np.mean(nis[acc])) if acc.any()
                else float('nan'), 100 * float(np.mean(nis[acc] < hi)) if acc.any() else 0, hi))
        rep('')
        rep('| distance | n | mean NIS | % below 5.99 | rejected |')
        rep('|---|---|---|---|---|')
        edges = [0, 0.5, 0.8, 1.2, 2.0, 10]
        for e0, e1 in zip(edges[:-1], edges[1:]):
            k = (dist >= e0) & (dist < e1)
            if k.any():
                ka = k & acc
                rep('| %.1f–%.1f m | %d | %s | %s | %d |' % (
                    e0, min(e1, 9.9), k.sum(),
                    '%.2f' % float(np.mean(nis[ka])) if ka.any() else '-',
                    '%.0f %%' % (100 * float(np.mean(nis[ka] < hi))) if ka.any() else '-',
                    int((k & ~acc).sum())))
        m = float(np.mean(nis[acc])) if acc.any() else float('nan')
        rep('')
        if math.isfinite(m):
            if m > 3.0:
                rep('The filter is optimistic (mean NIS %.1f > 3). Multiply R by ~%.1f or '
                    'check the α of Q; if the high NIS is concentrated far away, the camera '
                    'error model underestimates the distance error.' % (m, m / 2.0))
            elif m < 1.0:
                rep('The filter is pessimistic (mean NIS %.1f < 1): R or Q are larger than '
                    'necessary; you can reduce R by ~%.1f.' % (m, m / 2.0))
            else:
                rep('NIS within the expected range: R_k and Q_k are consistent.')
    else:
        rep('')
        rep('No corrections: does the detector see the ArUcos of the map (ids and positions)?')

    cps = [r for r in rows if r['kind'] == 'checkpoint']
    if cps:
        rep('')
        rep('## Checkpoints with real pose (NEES, χ² with 3 DOF)')
        rep('')
        rep('| # | EKF error (x, y mm; θ °) | EKF NEES | dead reckoning error (x, y mm; θ °) '
            '| NEES DR |')
        rep('|---|---|---|---|---|')
        ne, nd = [], []
        for i, r in enumerate(cps):
            ex = num(r, 'gt_x') - num(r, 'x')
            ey = num(r, 'gt_y') - num(r, 'y')
            et = math.degrees(math.atan2(math.sin(num(r, 'gt_theta') - num(r, 'theta')),
                                         math.cos(num(r, 'gt_theta') - num(r, 'theta'))))
            dx = num(r, 'gt_x') - num(r, 'dr_x')
            dy = num(r, 'gt_y') - num(r, 'dr_y')
            dt = math.degrees(math.atan2(math.sin(num(r, 'gt_theta') - num(r, 'dr_theta')),
                                         math.cos(num(r, 'gt_theta') - num(r, 'dr_theta'))))
            ne.append(num(r, 'nees'))
            nd.append(num(r, 'dr_nees'))
            rep('| %d | %+.0f, %+.0f; %+.1f | %.1f | %+.0f, %+.0f; %+.1f | %.1f |' % (
                i, 1000 * ex, 1000 * ey, et, ne[-1], 1000 * dx, 1000 * dy, dt, nd[-1]))
        rep('')
        rep('Mean NEES: EKF %.1f, dead reckoning %.1f (consistent ≈ 3; > 7.8 at many '
            'checkpoints = the filter underestimates its uncertainty).' % (
                float(np.nanmean(ne)), float(np.nanmean(nd))))
    else:
        rep('')
        rep('No checkpoints. For the NEES, stop the robot on a measured mark and publish:')
        rep('`ros2 topic pub --once /ekf/checkpoint geometry_msgs/msg/Pose2D '
            '"{x: 1.0, y: 0.5, theta: 1.5708}"`')

    rep_path = expand(path).with_suffix('.md')
    rep_path.write_text('\n'.join(out) + '\n')
    print('\nSaved: %s' % rep_path)


if __name__ == '__main__':
    main()
