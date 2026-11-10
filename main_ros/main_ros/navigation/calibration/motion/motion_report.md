# Motion model error margin (EKF prediction)

Robot: r = 0.0511 m, L = 0.1791 m. Data: /ws/main_ros/main_ros/navigation/calibration/motion/sweep_20261104_210024.csv, /ws/main_ros/main_ros/navigation/calibration/motion/sweep_20261109_163028.csv, /ws/main_ros/main_ros/navigation/calibration/motion/sweep_20261109_235046.csv

## 1. Steady-state noise due to commanded speed

Used sections: 130, discarded: 14. Startup ruled out: ≥ 1.0 s and until the 0.5 s moving average stabilizes. Encoder frequency ≈ 84 Hz.

| type | v cmd m/s | ω cmd rad/s | sections | start s | ωR media | σ ωR | ωL media | σ ωL | v media | σ v mm/s | ω media | σ ω rad/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| straight | 0.05 | 0.00 | 10 | 12.5 | 0.99 | 0.220 | 0.99 | 0.209 | 0.051 | 8.0 | 0.001 | 0.0757 |
| straight | 0.10 | 0.00 | 12 | 10.7 | 1.98 | 0.244 | 1.98 | 0.209 | 0.101 | 8.4 | 0.002 | 0.0816 |
| straight | 0.15 | 0.00 | 12 | 5.9 | 2.97 | 0.282 | 2.97 | 0.222 | 0.152 | 9.4 | 0.001 | 0.0919 |
| straight | 0.20 | 0.00 | 12 | 2.0 | 3.96 | 0.310 | 3.96 | 0.217 | 0.202 | 10.2 | 0.002 | 0.0944 |
| straight | 0.30 | 0.00 | 12 | 3.2 | 5.94 | 0.354 | 5.94 | 0.230 | 0.304 | 11.5 | 0.002 | 0.1046 |
| straight | 0.40 | 0.00 | 12 | 1.0 | 7.92 | 0.352 | 7.92 | 0.249 | 0.405 | 11.9 | 0.003 | 0.1051 |
| turn | 0.00 | 0.30 | 3 | 12.0 | 0.55 | 0.226 | 0.55 | 0.245 | 0.000 | 7.6 | 0.314 | 0.0958 |
| turn | 0.00 | 0.60 | 10 | 14.1 | 1.07 | 0.245 | 1.09 | 0.235 | 0.001 | 7.8 | 0.615 | 0.0996 |
| turn | 0.00 | 1.00 | 11 | 14.1 | 1.80 | 0.254 | 1.81 | 0.211 | 0.000 | 8.2 | 1.030 | 0.0890 |
| turn | 0.00 | 1.40 | 12 | 7.2 | 2.54 | 0.298 | 2.54 | 0.232 | 0.000 | 9.4 | 1.447 | 0.1033 |
| turn | 0.00 | 1.70 | 12 | 5.7 | 3.08 | 0.306 | 3.08 | 0.228 | 0.000 | 9.4 | 1.758 | 0.1052 |
| turn | 0.00 | 2.00 | 12 | 5.8 | 3.62 | 0.309 | 3.63 | 0.222 | 0.000 | 9.4 | 2.068 | 0.1050 |
- discarded turn v=0.00 ω=0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=-0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=-0.30: Didn't stabilize or window < 0.5 s
- discarded straight v=-0.05 ω=0.00: Didn't stabilize or window < 0.5 s
- discarded straight v=0.05 ω=0.00: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=-0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=-0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=-0.30: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=0.60: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=0.60: Didn't stabilize or window < 0.5 s
- discarded turn v=0.00 ω=1.00: Didn't stabilize or window < 0.5 s

## 2. Noise vs. velocity (slope)

| signal | σ0 (ordered) | slope k | R² |
|---|---|---|---|
| right wheel σ(|ωR|) [rad/s] | 0.2234 | 0.0198 | 0.86 |
| left wheel σ(|ωL|) [rad/s] | 0.2186 | 0.0023 | 0.14 |
| v in straight lines σ(|v|) [m/s] | 0.0075 | 0.0120 | 0.96 |
| ω in turns σ(|ω|) [rad/s] | 0.0923 | 0.0063 | 0.42 |

The EKF uses the straight-line path for each wheel: σ = σ0 + k·|ω_wheel|, Σ_w = diag(σR², σL²) y Q_k = α·J Σ_w Jᵀ con J = (r·Δt/2)·[[cosθ, cosθ], [sinθ, sinθ], [2/L, −2/L]].

## 3. Validation vs. tape (runs from runs.csv)

| run | type | e_x m | e_y m | e_θ ° | σx pred mm | σy pred mm | σθ pred ° | NEES |
|---|---|---|---|---|---|---|---|---|
| 0 | straight | -0.002 | -0.001 | -0.17 | 3.1 | 22.1 | 2.02 | 0.3 |
| 1 | straight | -0.002 | -0.027 | -4.96 | 3.2 | 23.2 | 2.01 | 8.6 |
| 2 | straight | -0.007 | -0.002 | -1.32 | 3.4 | 26.5 | 2.18 | 5.4 |
| 3 | straight | +0.005 | -0.028 | -3.67 | 3.0 | 21.8 | 1.96 | 7.2 |
| 4 | straight | +0.003 | -0.013 | -2.34 | 4.8 | 45.9 | 3.01 | 2.3 |
| 5 | turn | -0.003 | -0.012 | +7.19 | 2.5 | 2.0 | 2.03 | 47.4 |
| 6 | turn | +0.005 | +0.002 | -3.84 | 4.6 | 1.8 | 3.18 | 4.1 |
| 7 | turn | +0.001 | +0.003 | -4.70 | 4.3 | 1.8 | 2.99 | 5.4 |
| 8 | turn | +0.004 | -0.002 | -5.22 | 3.8 | 3.6 | 3.31 | 4.6 |
| 9 | turn | +0.005 | +0.003 | +5.97 | 2.4 | 1.9 | 1.96 | 14.3 |


Straight lines: actual distance / odometry = 0.9995 (n=5) -> suggested wheel radius 0.0511 m (actual 0.0511).
Rotations: odometry / actual rotation = 1.0005 (n=5) -> suggested distance between wheels 0.1791 m (actual 0.1791; it already includes the radius correction).

Mean NEES = 10.0(consistent ≈ 3; 70% of runs below χ²₃(95 %) = 7.82).
Suggested factor for Q: α = 3.32 (Q_k = α·J Σ_w Jᵀ). First, correct r and L if the ratios above deviate from 1 by more than 1%: the bias also inflates the NEES.

With this model, 1 m straight to 0.15 m/s leaves σx = 4.7 mm, σy = 30.4 mm, σθ = 3.02° (odometry only, without camera corrections).
