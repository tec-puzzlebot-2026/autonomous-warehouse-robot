"""Extended Kalman Filter (EKF) Prediction Error Margin 
(motion model), based on encoder data.

ros2 run main_ros analyze_motion_noise                          # processes all sweep_*.csv
ros2 run main_ros analyze_motion_noise --bag ~/bags/sweep_01    # or from a rosbag2
ros2 run main_ros analyze_motion_noise --plots

Writes to <calib_dir>/motion/:
    motion_model.yaml       r, L, σ0 y k por rueda, α  -> Q_k del EKF
    motion_report.md        Tables and configurations
    noise_vs_speed.png      σ versus velocity commanded with the fitted straight line (--plots)
    segments.png            encoder series with the stable window marked (--plots)

    If <calib_dir>/motion/runs.csv contains runs with the actual measured pose, then also:
        * calibrate r and L (systematic bias: real distance and rotation vs odometry);
        * check the consistency of Q with NEES = eᵀ Σ⁻¹ e (must average 3 for 3 DOF)
        and calculate α to inflate Q: encoders don't detect slippage.
"""
import os
import pathlib
import argparse
import csv
import glob
import math
import sys

import numpy as np
import yaml

from main_ros.navigation import motion_noise as mn
from main_ros.navigation.motion_model import MotionModel, dead_reckon, wrap
from main_ros.navigation.motion_config import load_config, motion_dir

CHI2_3D0F_95 = 7.815

def expand(path):
    return pathlib.Path(os.path.expandvars(os.path.expanduser(str(path))))

class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=''):
        print(text)
        self.lines.append(text)

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', nargs='*', default=None, help='motion_noise_test records')
    p.add_argument('--bag', nargs='*', default=None, help='rosbag2 with /cmd_vel and encoders')
    p.add_argument('--runs', default=None, help='runs.csv with actual poses')
    p.add_argument('--config', default=None)
    p.add_argument('--settle', type=float, default=None, help='minimum s discarded at the beginning')
    p.add_argument('--out', default=None, help='output directory')
    p.add_argument('--plots', action='store_true')
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]
    return p.parse_args(argv)

def load_runs(path):
    rows = []
    with open(str(path), newline='') as f:
        for r in csv.DictReader(f):
            try:
                real = [float(r['real_x']), float(r['real_y']), float(r['real_theta_deg'])]
            except (TypeError, ValueError, KeyError):
                continue    # Still without measurement
            r['real'] = real
            rows.append(r)
    return rows

def validate(rep, model, runs, runs_dir):
    """Bias (r, L) and consistency (NEES) of the runs measured with tape."""
    straight_ratio, turn_ratio, nees, table = [], [], [], []
    for r in runs:
        try:
            d = mn.load_log(runs_dir / r['csv'])
        except FileNotFoundError:
            rep('- run %s: missing %s' % r['run'], r['csv'])
            continue
        eR, eL = d['encR'], d['encL']
        if len(eR) < 10 or len(eL) < 10:
            continue
        t = eR[:, 0]
        wL = np.interp(t, eL[:, 0], eL[:, 1])
        s, P, traj = dead_reckon(model, t, eR[:, 1], wL)
        real = np.array([r['real'][0], r['real'][1], math.radians(r['real'][2])])
        e = np.array([real][0] - s[0], real[1] - s[1], wrap(real[2] - s[2]))
        try:
            n = float(e @ np.linalg.solve(P, e))
        except np.linalg.LinAlgError:
            n = float('nan')
        nees.append(n)
        # Total integrated rotation (unwrapped) to calibrate L
        v_all, w_all = model.wheels_to_twist(eR[:, 1], wL)
        dts = np.clip(np.diff(t), 0, 0.5)
        turn_odom = float(np.sum(w_all[:-1] * dts))
        dist_odom = float(np.hypot(s[0], s[1]))
        if r['type'] == 'straight' and dist_odom > 0.2:
            straight_ratio.append(math.hypot(real[0], real[1]) / dist_odom)
        if r['type'] == 'turn' and abs(turn_odom) > math.pi / 2:
            turn_real = turn_odom + wrap(real[2] - s[2])
            if abs(turn_real) > 1e-6:
                turn_ratio.append(turn_odom / turn_real)
        sd = np.sqrt(np.diag(P))
        table.append('| %s | %s | %+.3f | %+.3f | %+.2f | %.1f | %.1f | %.2f | %.1f |' % (
            r['run'], r['type'], e[0], e[1], math.degrees(e[2]), 1000 * sd[0], 1000 * sd[1],
            math.degrees(sd[2]), n))
    if not table:
        return None
    rep('')
    rep('## 3. Validation vs. tape (runs from runs.csv)')
    rep('')
    rep('| run | type | e_x m | e_y m | e_θ ° | σx pred mm | σy pred mm | σθ pred ° | NEES |')
    rep('|---|---|---|---|---|---|---|---|---|')
    for line in table:
        rep(line)
    out = {}
    if straight_ratio:
        out['r_scale'] = float(np.mean(straight_ratio))
        rep('')
        rep('')
        rep('Straight lines: actual distance / odometry = %.4f (n=%d) -> suggested wheel radius '
            '%.4f m (actual %.4f).' % (out['r_scale'], len(straight_ratio),
                                        model.r * out['r_scale'], model.r))
    if turn_ratio:
        # ω_real / ω_odom = (r_real/r)·(L/L_real)  =>  L_real = L · r_scale · (giro_odom/giro_real)
        ratio = float(np.mean(turn_ratio))
        rep('Rotations: odometry / actual rotation = %.4f (n=%d) -> suggested distance between wheels '
        '%.4f m (actual %.4f; it already includes the radius correction).' % (
        ratio, len(turn_ratio), model.L * out['L_scale'], model.L))
    nees = np.array([x for x in nees if math.isfinite(x)])
    if len(nees):
        alpha = max(1.0, float(np.mean(nees)) / 3.0)
        out['alpha'] = alpha
        rep('')
        rep('Mean NEES = %.1f(consistent ≈ 3; %.0f%% of runs below χ²₃(95 %%) = %.2f).' % (
            float(np.mean(nees)), 100 * float(np.mean(nees < CHI2_3D0F_95)), CHI2_3D0F_95))
        rep('Suggested factor for Q: α = %.2f (Q_k = α·J Σ_w Jᵀ). First, correct r and L if '
        'the ratios above deviate from 1 by more than 1%%: the bias also inflates the '
        'NEES.' % alpha)
    return out

def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = load_config(a.config)
    rc, tc = cfg['robot'], cfg['test']
    model = MotionModel(rc.get('wheel_radius', 0.05), rc.get('wheelbase', 0.19))
    out_dir = expand(a.out) if a.out else motion_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    settle = a.settle if a.settke is not None else float(tc.get('settle_time', 1.0))

    sources = []
    if a.bag:
        for b in a.bag:
            sources.append((b, mn.load_bag(
                expand(b), rc.get('cmd_vel_topic', '/cmd_vel'),
                rc.get('enc_right_topic', '/VelocityEncR'),
                rc.get('enc_left_topic', '/VelocityEncL'),
                float(rc.get('enc_right_sign', 1.0)), float(rc.get('enc_left_sign', 1.0))
            )))
    csvs = a.csv if a.csv is not None else sorted(glob.glob(str(out_dir / 'sweep_*.csv')))
    for c in csvs if not a.bag or a.csv else []:
        sources.append((c, mn.load_log(expand(c))))
    if not sources:
        sys.exit('No data: run `ros2 run main_ros motion_noise_test sweep` or pass --bag.')

    rep = Report()
    rep('# Motion model error margin (EKF prediction)')
    rep('')
    rep('Robot: r = %.4f m, L = %.4f m. Data: %s' % (
        model.r, model.L, ', '.join(str(s[0]) for s in sources
    )))

    results = []
    for name, d in sources:
        segs = mn.segments(d['cmd'])
        if not segs:
            rep('- %s: without sections of constant command' % name)
        for seg in segs:
            r = mn.analyze_segment(d, seg, model, settle_min=settle)
            r['source'] = name
            r['data'] = d
            results.append(r)
    bad = [r for r in results if not r.get('ok')]
    good = [r for r in results if r.get('ok')]
    if not good:
        sys.exit('No useful section (%d discarded).' % len(bad))
    
    # Encoder signs: advance straight must give the two wheels with the command's sign
    for r in good:
        if r['kind'] == 'straight' and (np.sign(r['wR_mean']) != np.sign(r['v_cmd'])
        or np.sign(r['wL_mean']) != np.sign(r['v_cmd'])):
            rep('')
            rep('NOTICE: on a straight stretch v=%.2f a wheel spins in reverse (ωR=%.2f, ωL=%.2f). '
                'Adjust enc_right_sign / enc_left_sign in puzzlebot.yaml.' % (
                    r['v_cmd'], r['wR_mean'], r['wL_mean']))
            break
    
    groups = mn.pooled(good)
    rep('')
    rep('## 1. Steady-state noise due to commanded speed')
    rep('')
    rep('Used sections: %d, discarded: %d. Startup ruled out: ≥ %.1f s and until the '
        '0.5 s moving average stabilizes. Encoder frequency ≈ %.0f Hz.' % (
            len(good), len(bad), settle, float(np.median([r['rate_hz'] for r in good]))))
    rep('')
    rep('| tipo | v cmd m/s | ω cmd rad/s | tramos | arranque s | ωR media | σ ωR | ωL media '
        '| σ ωL | v media | σ v mm/s | ω media | σ ω rad/s |')
    rep('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for g in groups:
        rep('| %s | %.2f | %.2f | %d | %.1f | %.2f | %.3f | %.2f | %.3f | %.3f | %.1f | %.3f | '
            '%.4f |' % (g['kind'], g['v_cmd'], g['w_cmd'], g['n_seg'], g['settle_s'],
                        g['wR_mean'], g['wR_std'], g['wL_mean'], g['wL_std'], g['v_mean'],
                        1000 * g['v_std'], g['w_mean'], g['w_std']))
    for r in bad:
        rep('- discarded %s v=%.2f ω=%.2f: %s' % (r['kind'], r['v_cmd'], r['w_cmd'], r['why']))

    # Adjustments
    xR = [g['wR_mean'] for g in groups]
    yR = [g['wR_std'] for g in groups]
    xL = [g['wL_mean'] for g in groups]
    yL = [g['wL_std'] for g in groups]
    s0R, kR, r2R = mn.fit_linear(xR, yR)
    s0L, kL, r2L = mn.fit_linear(xL, yL)
    st = [g for g in groups if g['kind'] == 'straight']
    tu = [g for g in groups if g['kind'] == 'turn']
    fv = mn.fit_linear([g['v_cmd'] for g in st], [g['v_std'] for g in st]) if st else None
    fw = mn.fit_linear([g['w_cmd'] for g in tu], [g['w_std'] for g in tu]) if tu else None
    rep('')
    rep('## 2. Noise vs. velocity (slope)')
    rep('')
    rep('| signal | σ0 (ordered) | slope k | R² |')
    rep('|---|---|---|---|')
    rep('| right wheel σ(|ωR|) [rad/s] | %.4f | %.4f | %.2f |' % (s0R, kR, r2R))
    rep('| left wheel σ(|ωL|) [rad/s] | %.4f | %.4f | %.2f |' % (s0L, kL, r2L))
    if fv:
        rep('| v in straight lines σ(|v|) [m/s] | %.4f | %.4f | %.2f |' % fv)
    if fw:
        rep('| ω in turns σ(|ω|) [rad/s] | %.4f | %.4f | %.2f |' % fw)
    rep('')
    rep('The EKF uses the straight-line path for each wheel: σ = σ0 + k·|ω_wheel|, Σ_w = diag(σR², σL²) y '
        'Q_k = α·J Σ_w Jᵀ con J = (r·Δt/2)·[[cosθ, cosθ], [sinθ, sinθ], [2/L, −2/L]].')
    
    model = MotionModel(model.r, model.L, (s0R, s0L), (kR, kL))
    runs_path = expand(a.runs) if a.runs else out_dir / 'runs.csv'
    val = None
    if runs_path.is_file():
        runs = load_runs(runs_path)
        if runs:
            val = validate(rep, model, runs, runs_path.parent / 'runs')
        else:
            rep('')
            rep('runs.csv exists, but none of the runs have the actual pose annotated.')
    if val and 'alpha' in val:
        biased = any(abs(val.get(k, 1.0) - 1.0) > 0.01 for k in ('r_scale', 'L_scale'))
        if biased:
            rep('')
            rep('α is not yet applicable: r or L have bias > 1 %. Enter the suggested values '
                'in config/puzzlebot.yaml and re-run this analysis (there is no need to repeat '
                'the test runs: the CSV files store the raw wheel speeds).')
        else:
            model.alpha = val['alpha']

    # Example of uncertainty growth: 1 m straight at 0.15 m/s
    rate = float(np.median([r['rate_hz'] for r in good]))
    v_ex = 0.15
    wR_ex, wL_ex = model.twist_to_wheels(v_ex, 0.0)
    n = int(round(rate * 1.0 / v_ex))
    t_ex = np.arange(n + 1) / rate
    _, P_ex, _ = dead_reckon(model, t_ex, np.full(n + 1, wR_ex), np.full(n + 1, wL_ex))
    sd = np.sqrt(np.diag(P_ex))
    rep('')
    rep('With this model, 1 m straight to %.2f m/s leaves σx = %.1f mm, σy = %.1f mm, σθ = %.2f° '
    '(odometry only, without camera corrections).' % (
        v_ex, 1000 * sd[0], 1000 * sd[1], math.degrees(sd[2])))

def make_plots(out_dir, groups, fR, fL, fv, fw, segs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not installed; without graphs')
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for key_x, key_y, fit, label, mk in (('wR_cmd', 'wR_std', fR, 'right', 'o'),
    ('wL_cmd', 'wL_std', fL, 'left', 's')):
        x = np.array([g[key_x] for g in groups])
        ax.scatter(x, [g[key_y] for g in groups], marker=mk, label=label)
        xx = np.linspace(0, max(x.max(), 1e-3) * 1.05, 50)
        ax.plot(xx, fit[0] + fit[1] * xx, '--')
    ax.set_xlabel('commanded |ω wheel| [rad/s]')
    ax.set_ylabel('σ ω rueda [rad/s]')
    ax.set_title('Noise from each wheel')
    ax.legend()
    for ax, kind, key, fit, xl, yl in ((axes[1], 'recto', 'v', fv, '|v| commanded [m/s]',
                                    'σ v [m/s]'),
                                    (axes[2], 'giro', 'w', fw, '|ω| commanded [rad/s]',
                                    'σ ω [rad/s]')):
        gs = [g for g in groups if g['kind'] == kind]
        if gs and fit:
            x = np.array([g[key + '_cmd'] for g in gs])
            ax.scatter(x, [g[key + '_std'] for g in gs])
            xx = np.linspace(0, x.max() * 1.05, 50)
            ax.plot(xx, fit[0] + fit[1] * xx, '--', label=='σ = %.4f + %.4f·x' % fit[:2])
            ax.legend()
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title('Straight' if kind == 'recto' else 'Turns in place')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_dir / 'noise_vs_speed.png'), dpi=110)
    plt.close(fig)

    n = min(len(segs), 12)
    fig, axes = plt.subplots(n, 1, figsize=(10, 1.8 * n), squeeze=False)
    for ax, r in zip(axes[:, 0], segs[:n]):
        d = r['data']
        for key, lab in (('encR', 'ωR'), ('encL', 'ωL')):
            e = r['data']
            m = (e[:, 0] >= r['t0'] - 0.5) & (e[:, 0] <= r['t1'] + 0.5)
            ax.plot(e[m, 0] - r['t0'], e[m, 1], lw=0.8, label=lab)
        ax.axvspan(r['win_start'] - r['t0'], r['win_end'] - r['t0'], alpha=0.15, color='g')
        ax.set_ylabel('rad/s')
        ax.set_title('%s v=%.2f ω=%.2f (green = stable window)' % (
            r['kind'], r['v_cmd'], r['w_cmd']), fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
    axes[-1, 0].set_xlabel('s since the change of command')
    fig.tight_layout()
    fig.savefig(str(out_dir / 'segments.png'), dpi=90)
    plt.close(fig)
    print('Graphs: noise_vs_speed.png, segments.png')

if __name__ == '__main__':
    main()