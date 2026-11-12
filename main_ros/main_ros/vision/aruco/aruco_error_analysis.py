"""Steps 3.1, 3.2, and 4: Validate the calibration and calculate the error model for the Kalman filter.

ros2 run main_ros aruco_error_analysis
ros2 run main_ros aruco_error_analysis --csv ~/main_ros_calibration/aruco/samples.csv \
--k-pos 0.02 --plots

Reads the samples from `take_sample` and writes the following to the CSV directory:
report.md          Everything printed to the console
error_model.yaml   Sigma (distance, angle) per axis -> R matrix for the Kalman filter
error_*.png        Plots (requires `matplotlib`; generated when `--plots` is used)

Report sections:
1. Table comparing ground truth vs. camera data (X, Y, θ, Z) for each sample, including jitter and ambiguity.
2. Scale: Linear fit of `camera_distance = s * ground_truth_distance + b`. A value of `s != 1` indicates an error in the ArUco marker size or in the `fx/fy` parameters; `b != 0` indicates a mounting error (in the X-axis).
3. Error margin (RAS/MoveIt style, AX = XB): includes the measured mounting configuration, the planar solution for AX = XB (with fixed Z), and the absolute estimate for each sample.
4. Error as a function of distance and viewing angle.
5. Sigma (r, angle) model and a table of weights (odometry vs. perception) for the Kalman filter.
"""
import argparse
import math
import sys

import numpy as np
import yaml

from main_ros.vision.aruco import experiment_io as xio
from main_ros.vision.aruco import handeye as he
from main_ros.vision.aruco import error_model as em
from main_ros.vision.aruco.paths import DEFAULT_CALIB_DIR, motion_dir

ANGLE_BINS = [0, 15, 30, 45, 60, 90]
UNITS = {'x': 'm', 'y': 'm', 'yaw': 'rad', 'range': 'm', 'bearing': 'rad'}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default=str(xio.default_samples_path()))
    p.add_argument('--config', default=None)
    p.add_argument('--mount-from-csv', action='store_true',
                   help='use the mounting configuration registered in the TF instead of the one from the config')
    p.add_argument('--k-pos', type=float, default=0.02,
                   help='odometry noise [m/sqrt(m)] if no motion_model.yaml')
    p.add_argument('--motion-model', default=str(motion_dir() /
                                                 'motion_model.yaml'),
                   help='measured motion model (analyze_motion_noise)')
    p.add_argument('--speed', type=float, default=0.15, help='m/s for the Kalman table')
    p.add_argument('--handeye-max-dist', type=float, default=1.0,
                   help='only samples at this distance or less for AX = XB [m]')
    p.add_argument('--mount-id', type=int, default=None,
                   help='ArUco used to validate the camera mount (AX = XB); default: marker_id '
                        'of the config. The error model uses all ArUcos.')
    p.add_argument('--plots', action='store_true')
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    return p.parse_args(argv)


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=''):
        print(text)
        self.lines.append(text)
    

def rms(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float('nan')


def fit_sigma(r, s):
    """Fit s ~ c0 + c2 r^2 (least squares, coefficients >= 0)."""
    r, s = np.asarray(r, float), np.asarray(s, float)
    ok = np.isfinite(r) & np.isfinite(s)
    r, s = r[ok], s[ok]
    if len(s) == 0:
        return 0.0, 0.0
    if len(s) >= 2 and np.ptp(r) > 1e-6:
        A = np.column_stack([np.ones_like(r), r ** 2])
        c0, c2 = np.linalg.lstsq(A, s, rcond=None)[0]
        if c2 < 0:
            return rms(s), 0.0
        if c0 < 0:
            c2 = float(np.sum(s * r ** 2) / max(np.sum(r ** 4), 1e-12))
            return 0.0, c2
        return float(c0), float(c2)
    return rms(s), 0.0


def odometry_variances(a, travels):
    """Fit the position variance (worst axis) after traveling each distance, with motion_model.yaml."""
    path = xio.expand(a.motion_model) if a.motion_model else None
    if path is None or not path.is_file():
        return None
    from main_ros.vision.aruco.paths import MotionModel, dead_reckon
    with open(str(path)) as f:
        raw = yaml.safe_load(f)
    mm = MotionModel.from_dict(raw.get('motion_model', raw))
    rate = float(raw.get('encoder_rate_hz', 50.0))
    wR, wL = mm.twist_to_wheels(a.speed, 0.0)
    out = []
    for s in travels:
        n = max(2, int(round(rate * s / max(a.speed, 1e-3))))
        t = np.arange(n + 1) / rate
        _, P, _ = dead_reckon(mm, t, np.full(n + 1, wR), np.full(n + 1, wL))
        out.append(float(max(P[0, 0], P[1, 1])))
    a.motion_model = str(path)
    return out


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = xio.load_config(a.config)
    exp = cfg['experiment']
    amb_thr = float(exp.get('ambiguity_threshold', 1.5))
    try:
        rows = xio.read_rows(a.csv)
    except FileNotFoundError:
        sys.exit('%s does not exist: take samples using take_sample first.' % a.csv)
    if len(rows) < 3:
        sys.exit('At least 3 samples are needed (there are %d).' % len(rows))

    mount = cfg['mount']
    if a.mount_from_csv and math.isfinite(rows[0]['mount_x']):
        mount = tuple(rows[0][k] for k in ('mount_x', 'mount_y', 'mount_z',
                                           'mount_roll', 'mount_pitch', 'mount_yaw'))
    X = he.mount_to_X(*mount)
    # Each sample is evaluated with the height and frame of ITS OWN ArUco (different heights per ID)
    ids = [int(r['marker_id']) if math.isfinite(r['marker_id']) else int(exp.get('marker_id', 21))
           for r in rows]
    ev = [xio.evaluate(r, X, xio.marker_z(exp, i)) for r, i in zip(rows, ids)]
    ref_id = a.mount_id if a.mount_id is not None else int(exp.get('marker_id', 21))
    if ref_id not in ids:
        ref_id = max(set(ids), key=ids.count)          # the most sampled one
    ref = [i for i, k in enumerate(ids) if k == ref_id]
    h_marker = xio.marker_z(exp, ref_id)
    out_dir = xio.expand(a.csv).parent
    rep = Report()
    deg = math.degrees

    rep('# Validation of camera–ArUco–robot calibration')
    rep('')
    rep('Samples: %d   Dictionary: %s' % (len(rows), cfg['detector'].get('dictionary', '?')))
    rep('')
    rep('| ArUco | samples | side mm | center above floor mm | above base_link mm |')
    rep('|---|---|---|---|---|')
    for k in sorted(set(ids)):
        rep('| %d%s | %d | %.1f | %.0f | %.0f |' % (
            k, ' (mount)' if k == ref_id else '', ids.count(k),
            1000 * xio.marker_size(cfg['detector'], k),
            1000 * xio.marker_height_floor(exp, k), 1000 * xio.marker_z(exp, k)))
    rep('Mount used (base_link -> camera_link): x=%.3f y=%.3f z=%.3f m, '
        'roll=%.2f° pitch=%.2f° yaw=%.2f°' % (mount[0], mount[1], mount[2],
                                             deg(mount[3]), deg(mount[4]), deg(mount[5])))

    # ---------------------------------------------------------------- 1
    rep('')
    rep('## 1. Real vs camera (errors = camera - real)')
    rep('')
    rep('| # | d | y | θ° | dist | vista° | eX mm | eY mm | eθ° | eZ mm | σX mm | σθ° | amb |')
    rep('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r, e in zip(rows, ev):
        amb = r['ambiguity_min']
        rep('| %d | %.2f | %+.2f | %+.0f | %.2f | %.0f | %+.1f | %+.1f | %+.2f | %+.1f | %.1f '
            '| %.2f | %s |' % (
                int(r['sample']), r['gt_d'], r['gt_y'], r['gt_yaw_deg'], r['distance'],
                r['view_angle_deg'], 1000 * e['err_x'], 1000 * e['err_y'], deg(e['err_yaw']),
                1000 * e['err_z'], 1000 * r['std_x'], r['std_yaw_deg'],
                ('%.2f%s' % (amb, ' ⚠' if amb < amb_thr else '')) if math.isfinite(amb)
                else '-'))

    # ---------------------------------------------------------------- 2
    rr = np.array([e['range_real'] for e in ev])
    rc = np.array([e['range_cam'] for e in ev])
    rep('')
    rep('## 2. Scale (intrinsic parameters and ArUco size)')
    rep('')
    for k in sorted(set(ids)):
        m = np.array([i == k for i in ids])
        rep('**ArUco %d** (%d samples)' % (k, m.sum()))
        rep('')
        if m.sum() < 3 or np.ptp(rr[m]) < 1e-6:
            rep('- Too few samples or a single distance: the scale cannot be fitted.')
            rep('')
            continue
        s_fit, b_fit = np.polyfit(rr[m], rc[m], 1)
        size = xio.marker_size(cfg['detector'], k)
        rep('camera_distance = %.4f · real_distance %+.4f m' % (s_fit, b_fit))
        rep('- Scale %+.1f %%: if it comes from the ArUco size, the real side would be ~%.4f m '
            '(configured %.4f m). If the size is measured correctly, check fx/fy.' % (
                100 * (s_fit - 1), size / s_fit if s_fit > 0 else float('nan'), size))
        rep('- Offset %+.1f mm: error in the x position of the camera (mounting) or in the '
            'reference of the tape.' % (1000 * b_fit))
        rep('- Mean bias: range %+.1f mm, bearing %+.2f°' % (
            1000 * float(np.mean(rc[m] - rr[m])),
            deg(float(np.mean([ev[i]['err_bearing'] for i in range(len(ev)) if m[i]])))))
        rep('')
    if len(set(ids)) > 1:
        rep('If the scale or bias of one ArUco differs from the others, check the size and height '
            'of THAT marker (per_marker_size / per_marker_height) before pooling.')

    # ---------------------------------------------------------------- 3
    # AX = XB needs ONE fixed marker: only the samples of the reference ArUco
    E = [ev[i]['T_W_B'] for i in ref]
    O = [ev[i]['T_C_M'] for i in ref]
    rel = [j for j, i in enumerate(ref) for r in [rows[i]]
           if r['distance'] <= a.handeye_max_dist
           and not (math.isfinite(r['ambiguity_min']) and r['ambiguity_min'] < amb_thr)]
    Er, Or = [E[i] for i in rel], [O[i] for i in rel]
    rep('')
    rep('## 3. Error Margin of the Calibration (AX = XB, as RAS / MoveIt)')
    rep('')
    rep('Only samples of ArUco %d (one fixed marker is the condition of AX = XB).' % ref_id)
    rep('')
    rep('Metric of moveit_calibration (getReprojectionError): RMS of the angle and the '
        'translation between AX and XB in pairs of consecutive samples.')
    rep('')
    r_c, t_c = he.reprojection_error(E, O, X, 'consecutive')
    r_r, t_r = he.reprojection_error(Er, Or, X, 'consecutive')
    rep('- Mount measured, all samples (%d): %.2f° / %.1f mm' % (
        len(E), deg(r_c), 1000 * t_c))
    rep('- Mount measured, reliable samples (%d; dist ≤ %.2f m, no ambiguity): '
        '%.2f° / %.1f mm' % (len(Er), a.handeye_max_dist, deg(r_r), 1000 * t_r))
    rep('')
    rep('The difference between both lines is the effect of the noisy orientation of the distant/ambiguous ArUco: '
        'AX = XB depends on it through B.')

    sols = []
    try:
        sols.append(('AX=XB plane (reliable, z fixed)', he.solve_planar(Er, Or, X)))
    except ValueError as exc:
        rep('AX = XB plane not available: %s' % exc)
    try:
        pts = [O_[:3, 3] for O_ in O]
        sols.append(('Position of the ArUco (all)', he.solve_from_points(
            E, pts, he.marker_in_world(h_marker)[:3, 3], X)))
    except ValueError as exc:
        rep('Position-based estimation not available: %s' % exc)
    rep('')
    rep('| mount camera_link | x mm | y mm | z mm | roll° | pitch° | yaw° | margin AX=XB '
        '(confiables) |')
    rep('|---|---|---|---|---|---|---|---|')
    rep('| measured (tape measure) | %.1f | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f° / %.1f mm |' % (
        1000 * mount[0], 1000 * mount[1], 1000 * mount[2], deg(mount[3]), deg(mount[4]),
        deg(mount[5]), deg(r_r), 1000 * t_r))
    for name, sol in sols:
        m = sol['mount']
        rs, ts = he.reprojection_error(Er, Or, sol['X'], 'consecutive')
        rep('| %s%s | %.1f | %.1f | %.1f | %.2f | %.2f | %.2f | %.2f° / %.1f mm |' % (
            name, '' if sol['well_conditioned'] else ' ⚠ poorly conditioned',
            1000 * m[0], 1000 * m[1], 1000 * m[2], deg(m[3]), deg(m[4]), deg(m[5]),
            deg(rs), 1000 * ts))
    rep('')
    rep('In the plane, AX = XB cannot observe z (the measurement is left). The estimation by '
        'positions uses the known height of the ArUco and only the position of the PnP, the part '
        'precise of the measurement: it is the recommended for correcting the config.')
    if sols:
        best = sols[-1][1]['mount']
        dpos = 1000 * math.hypot(best[0] - mount[0], best[1] - mount[1])
        dang = deg(max(abs(best[4] - mount[4]), abs(best[5] - mount[5])))
        if dpos > 3.0 or dang > 0.5:
            rep('')
            rep('Suggestion for camera_mount in aruco_localization.yaml: x: %.4f  y: %.4f  '
                'z: %.4f  roll: %.4f  pitch: %.4f  yaw: %.4f' % best)

    # ---------------------------------------------------------------- 4
    view = np.array([r['view_angle_deg'] for r in rows])
    dist = np.array([r['distance'] for r in rows])
    err = {q: np.array([e['err_' + q] for e in ev]) for q in em.QUANTITIES}
    jit = {'x': np.array([r['std_x'] for r in rows]),
           'y': np.array([r['std_y'] for r in rows]),
           'yaw': np.radians([r['std_yaw_deg'] for r in rows]),
           'range': np.array([r['std_dist'] for r in rows]),
           'bearing': np.zeros(len(rows))}
    rep('')
    rep('## 4. Error against distance and angle')
    rep('')
    rep('| real d | n | RMS eX mm | RMS eY mm | RMS eθ° | RMS eZ mm | jitter X mm |')
    rep('|---|---|---|---|---|---|---|')
    gt_d = np.array([r['gt_d'] for r in rows])
    ez = np.array([e['err_z'] for e in ev])
    for d in sorted(set(np.round(gt_d, 3))):
        k = np.abs(gt_d - d) < 1e-3
        rep('| %.2f | %d | %.1f | %.1f | %.2f | %.1f | %.2f |' % (
            d, k.sum(), 1000 * rms(err['x'][k]), 1000 * rms(err['y'][k]),
            deg(rms(err['yaw'][k])), 1000 * rms(ez[k]), 1000 * rms(jit['x'][k])))
    rep('')
    rep('| view angle | n | RMS eX mm | RMS eθ° | jitter θ° |')
    rep('|---|---|---|---|---|')
    for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:]):
        k = (view >= lo) & (view < hi)
        if k.any():
            rep('| %d–%d° | %d | %.1f | %.2f | %.2f |' % (
                lo, hi, k.sum(), 1000 * rms(err['x'][k]), deg(rms(err['yaw'][k])),
                deg(rms(jit['yaw'][k]))))

    # ---------------------------------------------------------------- 5
    base = view <= 20 if (view <= 20).sum() >= 3 else np.ones(len(rows), bool)
    model = {'dictionary': cfg['detector'].get('dictionary'),
             'marker_size': float(cfg['detector'].get('marker_size', float('nan'))),
             'marker_size_per_id': {int(k): xio.marker_size(cfg['detector'], k) for k in set(ids)},
             'marker_height_per_id': {int(k): xio.marker_height_floor(exp, k) for k in set(ids)},
             'mount': [float(v) for v in mount],
             'bias': {}, 'sigma': {}, 'angle_factor': {}, 'units': UNITS}
    floors = {'x': 0.001, 'y': 0.001, 'yaw': math.radians(0.2), 'range': 0.001,
              'bearing': math.radians(0.1)}
    amb = np.array([r['ambiguity_min'] for r in rows])
    clear = ~(np.isfinite(amb) & (amb < amb_thr))   # pose without IPPE ambiguity
    for q in em.QUANTITIES:
        # x, y, and yaw depend on the ArUco orientation: ambiguous samples are
        # excluded (the filter rejects them). Distance and bearing do not depend on it.
        use = clear if q in ('x', 'y', 'yaw') else np.ones(len(rows), bool)
        fit = base & use
        if fit.sum() < 2:
            fit = use
        bias = float(np.mean(err[q][fit]))
        s = np.sqrt((err[q] - bias) ** 2 + jit[q] ** 2)
        c0, c2 = fit_sigma(dist[fit], s[fit])
        model['bias'][q] = bias
        model['sigma'][q] = {'c0': c0, 'c2': c2, 'floor': floors[q]}
        angles, factors = [0.0], [1.0]
        for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:]):
            k = (view >= lo) & (view < hi) & use
            if k.sum() >= 1 and lo >= 15:
                pred = np.maximum(c0 + c2 * dist[k] ** 2, floors[q])
                angles.append(float(np.mean(view[k])))
                factors.append(max(1.0, rms(s[k] / pred)))
        model['angle_factor'][q] = {'angles_deg': angles, 'factor': factors}
    max_angle = min(float(np.nanmax(view)), 60.0)
    model['valid'] = {'max_distance': float(np.nanmax(dist)),
                      'max_view_angle_deg': max_angle, 'min_ambiguity': amb_thr}

    rep('')
    rep('## 5. Error Model for the Kalman Filter')
    rep('')
    rep('sigma(r, angle) = (c0 + c2·r²)·f(angle), adjusted with samples of angle ≤ 20°; '
        'x, y, θ without ambiguous samples (%d of %d are).' % ((~clear).sum(), len(rows)))
    rep('')
    rep('| axis | bias | c0 | c2 | sigma at 0.5 m | sigma at 1.0 m | sigma at 1.5 m |')
    rep('|---|---|---|---|---|---|---|')
    for q in em.QUANTITIES:
        f = deg if UNITS[q] == 'rad' else (lambda v: 1000 * v)
        unit = '°' if UNITS[q] == 'rad' else 'mm'
        c = model['sigma'][q]
        rep('| %s [%s] | %+.2f | %.2f | %.2f | %.2f | %.2f | %.2f |' % (
            q, unit, f(model['bias'][q]), f(c['c0']), f(c['c2']),
            *[f(em.sigma(model, q, r, 0.0)) for r in (0.5, 1.0, 1.5)]))
    rep('')
    rep('Valid up to %.2f m and %.0f° of view angle; θ of the ArUco ignored if '
        'ambiguity < %.1f.' % (model['valid']['max_distance'], max_angle, amb_thr))
    s_lat = em.sigma(model, 'bearing', 1.0, 0) * 1.0
    if em.sigma(model, 'y', 1.0, 0) > 3 * max(s_lat, em.sigma(model, 'range', 1.0, 0)):
        rep('')
        rep('The robot pose obtained from a single ArUco (x, y, θ) inherits the error of '
            'marker orientation; (distance, bearing) is %.0f times more precise at 1 m.'
            'For the EKF use distance and bearing to the ArUco (known position in the map) as '
            'measurement, and θ of the ArUco only close and unambiguous.' % (
                em.sigma(model, 'y', 1.0, 0) / max(s_lat, em.sigma(model, 'range', 1.0, 0))))
    if any(abs(model['bias'][q]) > 2 * em.sigma(model, q, 0.5, 0) for q in ('range', 'bearing')):
        rep('')
        rep('The bias of distance or bearing exceeds 2σ: correct the mounting or size of the ArUco '
            '(sections 2 and 3) and repeat the analysis; the Kalman filter assumes a zero-mean error.')

    rep('')
    rep('### Odometry or perception? Camera weight K = P/(P+R)')
    rep('')
    travels = [0.25, 0.5, 1.0, 2.0, 4.0]
    odo_var = odometry_variances(a, travels)
    if odo_var is None:
        odo_var = [a.k_pos ** 2 * s for s in travels]
        rep('Odometry: k_pos = %.3f m/√m assumed (no motion_model.yaml yet; run '
            'analyze_motion_noise to use the measured model).' % a.k_pos)
    else:
        rep('Odometry: %s, straight-line travel at %.2f m/s (σ of position = worst axis).' % (
            a.motion_model, a.speed))
    rep('Distance–bearing measurement, ArUco in front; sigma of position = max(sigma_distance, '
        'r·sigma_heading). Columns: meters traveled since the last correction.')
    rep('')
    rep('| dist ArUco | sigma pos | ' + ' | '.join('%.2f m' % s for s in travels) + ' |')
    rep('|---|---|' + '---|' * len(travels))
    for r in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        if r > model['valid']['max_distance'] + 1e-6:
            continue
        sp = em.position_sigma_range_bearing(model, r, 0.0)
        ws = [em.perception_weight([pv], [sp ** 2])[0] for pv in odo_var]
        rep('| %.1f m | %.1f mm | ' % (r, 1000 * sp) + ' | '.join('%.2f' % w for w in ws) + ' |')
    rep('')
    rep('Reading: > 0.5 the correction is more based on the camera; < 0.5, on the odometry. '
        'With large viewing angles the sigma grows by f(angle) and the weight decreases.')

    path = out_dir / 'error_model.yaml'
    with open(str(path), 'w') as f:
        yaml.safe_dump(model, f, sort_keys=False)
    (out_dir / 'report.md').write_text('\n'.join(rep.lines) + '\n')
    print('\nSaved: %s\n          %s' % (path, out_dir / 'report.md'))
    if a.plots:
        make_plots(out_dir, model, dist, view, err, jit)


def make_plots(out_dir, model, dist, view, err, jit):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; no plots')
        return
    rr = np.linspace(0.05, max(float(np.nanmax(dist)), 0.2), 100)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, q, scale, unit in zip(axes, ('x', 'y', 'yaw'), (1000, 1000, 180 / math.pi),
                                  ('mm', 'mm', '°')):
        sc = ax.scatter(dist, np.abs(err[q] - model['bias'][q]) * scale, c=view, cmap='viridis',
                        vmin=0, vmax=75, label='|error - bias|')
        ax.plot(rr, [em.sigma(model, q, r, 0) * scale for r in rr], 'r-', label='σ modelo (0°)')
        ax.set_xlabel('distance [m]')
        ax.set_ylabel('error %s [%s]' % (q, unit))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.colorbar(sc, ax=axes, label='viewing angle [°]')
    fig.savefig(str(out_dir / 'error_vs_distance.png'), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(view, np.degrees(np.abs(err['yaw'] - model['bias']['yaw'])), c=dist,
               cmap='plasma')
    ax.set_xlabel('viewing angle [°]')
    ax.set_ylabel('|error θ| [°]')
    ax.grid(alpha=0.3)
    fig.savefig(str(out_dir / 'error_vs_angle.png'), dpi=110)
    plt.close(fig)
    print('Plots: error_vs_distance.png, error_vs_angle.png')


if __name__ == '__main__':
    main()