# Validation of camera–ArUco–robot calibration

Samples: 29   Dictionary: DICT_ARUCO_ORIGINAL

| ArUco | samples | side mm | center above floor mm | above base_link mm |
|---|---|---|---|---|
| 5 | 1 | 87.0 | 45 | -2 |
| 7 | 1 | 87.0 | 48 | 2 |
| 13 | 1 | 87.0 | 43 | -4 |
| 21 (mount) | 25 | 87.0 | 46 | -2 |
| 37 | 1 | 87.0 | 43 | -4 |
Mount used (base_link -> camera_link): x=0.100 y=0.000 z=0.178 m, roll=0.00° pitch=14.90° yaw=-0.71°

## 1. Real vs camera (errors = camera - real)

| # | d | y | θ° | dist | vista° | eX mm | eY mm | eθ° | eZ mm | σX mm | σθ° | amb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.40 | +0.00 | +0 | 0.35 | 32 | -2.3 | -1.0 | -0.08 | +0.1 | 0.3 | 0.11 | 11.51 |
| 1 | 0.40 | +0.20 | +0 | 0.41 | 44 | +3.3 | +21.1 | -2.82 | +5.8 | 0.8 | 0.11 | 6.29 |
| 2 | 0.40 | +0.40 | +0 | 0.54 | 56 | -6.5 | +5.7 | -1.73 | +17.1 | 1.6 | 0.17 | 4.05 |
| 3 | 0.40 | -0.20 | +0 | 0.41 | 43 | -4.7 | -18.6 | +1.90 | +10.8 | 0.9 | 0.14 | 5.40 |
| 4 | 0.40 | -0.30 | +0 | 0.48 | 49 | -19.6 | -5.2 | -0.03 | +18.6 | 1.9 | 0.23 | 5.50 |
| 5 | 0.60 | +0.00 | +0 | 0.53 | 19 | -2.6 | -34.4 | +2.87 | +2.8 | 0.6 | 0.26 | 5.89 |
| 6 | 0.60 | +0.20 | +0 | 0.58 | 29 | +0.9 | +34.8 | -3.12 | +11.6 | 1.0 | 0.15 | 5.04 |
| 7 | 0.60 | +0.40 | +0 | 0.68 | 44 | +11.8 | +47.5 | -5.35 | +29.5 | 2.6 | 0.21 | 5.33 |
| 8 | 0.60 | -0.20 | +0 | 0.56 | 32 | +19.8 | -48.4 | +5.62 | +0.6 | 1.4 | 0.18 | 5.05 |
| 9 | 0.60 | -0.40 | +0 | 0.66 | 45 | +30.8 | -42.0 | +5.78 | +10.4 | 2.5 | 0.18 | 5.12 |
| 10 | 0.80 | +0.00 | +0 | 0.72 | 13 | +15.3 | +125.5 | -9.47 | -0.0 | 14.2 | 3.96 | 1.07 ⚠ |
| 11 | 0.80 | +0.20 | +0 | 0.74 | 28 | +44.9 | +101.0 | -7.78 | -10.8 | 2.8 | 0.26 | 2.57 |
| 12 | 0.80 | +0.40 | +0 | 0.81 | 40 | +62.0 | +78.8 | -7.52 | -2.8 | 4.0 | 0.28 | 3.08 |
| 13 | 0.80 | -0.20 | +0 | 0.76 | 25 | +21.0 | -95.7 | +6.74 | +5.0 | 2.3 | 0.25 | 5.13 |
| 14 | 0.80 | -0.40 | +0 | 0.83 | 37 | +35.2 | -82.7 | +7.22 | +15.4 | 4.2 | 0.29 | 2.88 |
| 15 | 1.00 | +0.00 | +0 | 0.94 | 16 | +7.1 | +229.4 | -12.49 | +20.6 | 2.6 | 0.44 | 1.35 ⚠ |
| 16 | 1.00 | +0.20 | +0 | 0.94 | 26 | +46.6 | +147.1 | -9.28 | +1.3 | 4.2 | 0.39 | 2.14 |
| 17 | 1.00 | +0.40 | +0 | 1.02 | 34 | +48.3 | +146.3 | -8.35 | +25.9 | 6.9 | 0.42 | 2.13 |
| 18 | 1.00 | -0.20 | +0 | 0.97 | 20 | +8.4 | -131.5 | +6.63 | +29.8 | 4.0 | 0.40 | 1.64 |
| 19 | 1.00 | -0.40 | +0 | 1.04 | 30 | +10.8 | -114.1 | +5.58 | +40.6 | 4.6 | 0.43 | 1.33 ⚠ |
| 20 | 1.25 | +0.00 | +0 | 1.18 | 22 | +69.6 | +450.3 | -21.18 | +13.3 | 6.8 | 0.66 | 1.32 ⚠ |
| 21 | 1.25 | +0.20 | +0 | 1.19 | 28 | +88.8 | +312.3 | -15.06 | +6.9 | 6.6 | 0.61 | 1.44 ⚠ |
| 22 | 1.25 | +0.40 | +0 | 1.25 | 33 | +100.6 | +260.9 | -11.74 | +15.9 | 4.5 | 0.31 | 1.61 |
| 23 | 1.25 | -0.20 | +0 | 1.21 | 26 | +97.0 | +770.5 | -36.01 | +24.5 | 7.3 | 0.64 | 1.17 ⚠ |
| 24 | 1.25 | -0.40 | +0 | 1.28 | 28 | +71.8 | +1054.0 | -46.51 | +49.0 | 8.1 | 0.52 | 1.12 ⚠ |
| 25 | 0.60 | +0.00 | +0 | 0.55 | 13 | -2.5 | -119.7 | +12.79 | +22.0 | 1.3 | 0.41 | 1.06 ⚠ |
| 26 | 0.60 | +0.00 | +0 | 0.54 | 15 | -23.2 | +42.2 | -4.29 | +13.7 | 0.5 | 0.34 | 3.05 |
| 27 | 0.60 | +0.00 | +0 | 0.54 | 20 | -24.7 | +11.3 | -1.32 | +13.3 | 0.5 | 0.20 | 2.07 |
| 28 | 0.60 | +0.00 | +0 | 0.54 | 16 | -22.3 | +1.4 | +0.30 | +9.9 | 0.6 | 0.35 | 3.39 |

## 2. Scale (intrinsic parameters and ArUco size)

**ArUco 5** (1 samples)

- Too few samples or a single distance: the scale cannot be fitted.

**ArUco 7** (1 samples)

- Too few samples or a single distance: the scale cannot be fitted.

**ArUco 13** (1 samples)

- Too few samples or a single distance: the scale cannot be fitted.

**ArUco 21** (25 samples)

camera_distance = 1.0202 · real_distance -0.0058 m
- Scale +2.0 %: if it comes from the ArUco size, the real side would be ~0.0853 m (configured 0.0870 m). If the size is measured correctly, check fx/fy.
- Offset -5.8 mm: error in the x position of the camera (mounting) or in the reference of the tape.
- Mean bias: range +11.6 mm, bearing +0.10°

**ArUco 37** (1 samples)

- Too few samples or a single distance: the scale cannot be fitted.

If the scale or bias of one ArUco differs from the others, check the size and height of THAT marker (per_marker_size / per_marker_height) before pooling.

## 3. Error Margin of the Calibration (AX = XB, as RAS / MoveIt)

Only samples of ArUco 21 (one fixed marker is the condition of AX = XB).

Metric of moveit_calibration (getReprojectionError): RMS of the angle and the translation between AX and XB in pairs of consecutive samples.

- Mount measured, all samples (25): 11.55° / 204.9 mm
- Mount measured, reliable samples (16; dist ≤ 1.00 m, no ambiguity): 9.30° / 131.2 mm

The difference between both lines is the effect of the noisy orientation of the distant/ambiguous ArUco: AX = XB depends on it through B.

| mount camera_link | x mm | y mm | z mm | roll° | pitch° | yaw° | margin AX=XB (confiables) |
|---|---|---|---|---|---|---|---|
| measured (tape measure) | 100.0 | 0.0 | 178.0 | 0.00 | 14.90 | -0.71 | 9.30° / 131.2 mm |
| AX=XB plane (reliable, z fixed) ⚠ poorly conditioned | 37387294.8 | 24771904.2 | 178.0 | -2.52 | 17.02 | -1.51 | 9.30° / 3561375.6 mm |
| Position of the ArUco (all) | 83.5 | -5.3 | 167.9 | -0.55 | 13.87 | -0.52 | 9.30° / 129.3 mm |

In the plane, AX = XB cannot observe z (the measurement is left). The estimation by positions uses the known height of the ArUco and only the position of the PnP, the part precise of the measurement: it is the recommended for correcting the config.

Suggestion for camera_mount in aruco_localization.yaml: x: 0.0835  y: -0.0053  z: 0.1679  roll: -0.0096  pitch: 0.2421  yaw: -0.0090

## 4. Error against distance and angle

| real d | n | RMS eX mm | RMS eY mm | RMS eθ° | RMS eZ mm | jitter X mm |
|---|---|---|---|---|---|---|
| 0.40 | 5 | 9.6 | 13.0 | 1.71 | 12.6 | 1.25 |
| 0.60 | 9 | 18.7 | 52.7 | 5.73 | 15.2 | 1.46 |
| 0.80 | 5 | 39.4 | 98.2 | 7.80 | 8.8 | 7.06 |
| 1.00 | 5 | 30.8 | 158.7 | 8.80 | 27.0 | 4.69 |
| 1.25 | 5 | 86.5 | 643.9 | 29.23 | 26.4 | 6.77 |

| view angle | n | RMS eX mm | RMS eθ° | jitter θ° |
|---|---|---|---|---|
| 0–15° | 2 | 11.0 | 11.25 | 2.81 |
| 15–30° | 15 | 47.5 | 17.50 | 0.42 |
| 30–45° | 9 | 44.9 | 6.56 | 0.25 |
| 45–60° | 3 | 21.4 | 3.48 | 0.20 |

## 5. Error Model for the Kalman Filter

sigma(r, angle) = (c0 + c2·r²)·f(angle), adjusted with samples of angle ≤ 20°; x, y, θ without ambiguous samples (8 of 29 are).

| axis | bias | c0 | c2 | sigma at 0.5 m | sigma at 1.0 m | sigma at 1.5 m |
|---|---|---|---|---|---|---|
| x [mm] | -18.19 | 9.08 | 0.00 | 9.08 | 9.08 | 9.08 |
| y [mm] | +5.10 | 27.52 | 0.00 | 27.52 | 27.52 | 27.52 |
| yaw [°] | -0.61 | 2.61 | 0.00 | 2.61 | 2.61 | 2.61 |
| range [mm] | +17.65 | 10.68 | 0.00 | 10.68 | 10.68 | 10.68 |
| bearing [°] | -0.15 | 0.65 | 0.00 | 0.65 | 0.65 | 0.65 |

Valid up to 1.28 m and 56° of view angle; θ of the ArUco ignored if ambiguity < 1.5.

### Odometry or perception? Camera weight K = P/(P+R)

Odometry: /home/puzzlebot/te3003b_ws/src/autonomous-warehouse-robot/main_ros/main_ros/navigation/calibration/motion/motion_model.yaml, straight-line travel at 0.15 m/s (σ of position = worst axis).
Distance–bearing measurement, ArUco in front; sigma of position = max(sigma_distance, r·sigma_heading). Columns: meters traveled since the last correction.

| dist ArUco | sigma pos | 0.25 m | 0.50 m | 1.00 m | 2.00 m | 4.00 m |
|---|---|---|---|---|---|---|
| 0.2 m | 10.7 mm | 0.11 | 0.50 | 0.89 | 0.98 | 1.00 |
| 0.4 m | 10.7 mm | 0.11 | 0.50 | 0.89 | 0.98 | 1.00 |
| 0.6 m | 10.7 mm | 0.11 | 0.50 | 0.89 | 0.98 | 1.00 |
| 0.8 m | 10.7 mm | 0.11 | 0.50 | 0.89 | 0.98 | 1.00 |
| 1.0 m | 11.4 mm | 0.10 | 0.47 | 0.88 | 0.98 | 1.00 |

Reading: > 0.5 the correction is more based on the camera; < 0.5, on the odometry. With large viewing angles the sigma grows by f(angle) and the weight decreases.
