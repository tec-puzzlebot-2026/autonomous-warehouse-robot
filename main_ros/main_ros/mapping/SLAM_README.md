# Team Zero Custom 2D LiDAR SLAM — v2 (loop closure)

Particle-filter front-end + pose-graph back-end with loop closure, written from scratch in
Python/NumPy for ROS 2. No SLAM library is used. SciPy is used only for the sparse linear
solve of the pose graph (optional: a dense NumPy fallback handles up to ~1200 keyframes).

## 1. What v2 adds

v1 corrected each scan locally, but small errors still accumulated: after a long loop the robot
came back to a known place slightly off, and the old and new walls were drawn twice or bent.
v2 detects when the robot is back in a known place, measures the exact error with scan
matching, spreads the correction over the whole trajectory, and redraws the map.

```text
FRONT-END (main thread, every scan)                BACK-END (background thread)
 odometry ─► predict particles                       every 2 new keyframes:
 LiDAR    ─► weight against LOCAL map                 1. candidates: old keyframes within 3 m (x, y)
            resample, estimate pose                      and at least 10 m back along the path
            new keyframe? ─────────────────────────►  2. verify: correlative scan match of the new
              • pose-graph node + "seq" edge              scan against the old submap
              • into local map + global map           3. accepted ─► "loop" edge in the pose graph
                                                      4. edge disagrees with the estimate?
 apply correction ◄───────────────────────────────       optimize the graph (Gauss-Newton/LM,
  • keyframes 0..K ← optimized poses                      Huber kernel, outlier rejection)
  • newer keyframes + particles moved rigidly         5. rebuild the global map from every keyframe
  • rebuilt map swapped in                               at its optimized pose
```

## 2. Files

```text
mapping/
├── slam.py                  # ROS 2 node: messages, TF, publishing, services
├── slam_params.yaml         # all tunable parameters
├── SLAM_README.md           # this file
├── slam_core/               # the algorithm, no ROS imports
│   ├── se2.py               # rigid 2D transforms (compose, inverse, relative, vectorized)
│   ├── scan_processing.py   # beam filtering, mounting transform, matching / loop point sets
│   ├── odometry_buffer.py   # odometry history + interpolation at the scan time
│   ├── particle_filter.py   # motion model, weighting, systematic resampling, estimate
│   ├── likelihood_field.py  # distance transform + likelihood-field sensor model
│   ├── occupancy_grid.py    # log-odds grid, ray tracing, reversible updates, auto-growth
│   ├── keyframes.py         # keyframe storage + sliding-window local map       (new)
│   ├── pose_graph.py        # pose graph + Gauss-Newton/LM optimizer             (new)
│   ├── scan_matcher.py      # correlative scan matcher with branch-and-bound     (new)
│   ├── loop_closure.py      # loop detection, verification, optimize, rebuild    (new)
│   └── slam_system.py       # front-end loop + applying back-end corrections
└── tests/
    └── offline_sim_test.py  # ROS-free simulation with ground truth, v1 vs v2
```

## 3. Theory of the new parts

### 3.1 Keyframes become a pose graph
Every keyframe i is a **node** with pose X_i = (x, y, θ). Every pair of consecutive keyframes is
joined by a **seq edge** carrying the relative pose the front-end measured,
z = X_(i-1)⁻¹ ⊕ X_i, and an information matrix Ω = Σ⁻¹ (confidence).

Σ is anisotropic, expressed in the robot frame: along-track, cross-track and heading. Each
standard deviation has a fixed part plus a part that grows with distance and rotation. The values
were **calibrated against ground truth** in the offline simulation. Two lessons came out of it:
- the front-end is about 3× less certain along the direction of travel than sideways, because
  corridors constrain the robot laterally but not along their length;
- calibration must use the drift over **hundreds** of keyframes, not single steps. The front-end
  keeps re-matching the local map, so its heading errors partly cancel: 0.10° per step but only
  0.36° after 500 steps, where a random walk would reach 2.1°. A model built from single-step
  errors was 2–3× too loose in heading, and the optimizer then "fixed" loops by bending corridors.

Using the particle-cloud covariance as the edge covariance was also tested and rejected. It
measures the uncertainty of the *absolute* pose, not of one step, so summed over a long chain it let
the optimizer slide whole corridors by a metre during intermediate corrections.

Each keyframe also stores its scan in the robot frame. Scans never change, so the map can be
re-drawn from scratch whenever the poses change.

### 3.2 Local front-end map
The front-end now matches against a map of only the last ~20 m of keyframes (a sliding window).
If it matched against the whole map, then when the robot came back with 40 cm of drift it would
"snap" onto the old walls in a single step. That hides the error inside one edge, and the rest
of the loop stays bent. With a local map the drift stays visible, and the pose graph spreads it
correctly over the whole loop. Karto and Google Cartographer use the same split.

The window must be at least about **twice the LiDAR range**. A pillar the robot passed 6 m ago is
still in view, and the keyframes that mapped it were taken up to one LiDAR range before reaching it.
If they leave the window, the front-end loses the features that tell it how far it has moved along a
corridor. With an 8 m window and an 8 m LiDAR, testing showed a 1% stretch along corridors. If the
Isaac Sim LiDAR reaches farther than 10 m, set `local_map_window` to about twice that range, or cap
`match_max_range`.

Removing old keyframes from the local map is exact. The log-odds update is a sum, so a keyframe
is removed by applying its own update with the opposite sign (no clamping in the local map).

### 3.3 Loop detection ("have I been here before?")
For every second new keyframe k:
1. **Candidates:** old keyframes j whose estimated (x, y) is within `loop_search_radius` (3 m)
   of k and that are at least `loop_min_path` (10 m) back along the travelled path. The path
   rule skips keyframes the robot has just passed, which are always nearby but are not a loop.
2. **Verification by scan matching.** Proximity alone is not enough: the estimate has drifted,
   and identical rack aisles look alike. Keyframe k's scan is matched against a submap made
   of keyframe j and its path neighbours (±2.5 m), searching ±1.5 m and ±15° around the current
   guess. The match is accepted only if its score ≥ 0.50 and ≥ 60% of the points land on old
   walls.

### 3.4 Correlative scan matching with branch-and-bound
The submap points are rasterized into a lookup table
S(c) = exp(−d(c)² / 2σ²), where d is the distance from cell c to the nearest submap point
(σ = 5 cm). A candidate pose scores the mean of S over the transformed scan points
(Olson, ICRA 2009). The search is exhaustive over the whole window, so it finds the answer even
when the error is too big for a local optimizer. Exhaustive search is slow, so the translations use
**branch-and-bound** (Hess et al., ICRA 2016):
- a coarse table C(c) = max of S over a 4×4 block is an **upper bound** of every fine translation
  in that block;
- coarse blocks are expanded to 5 cm resolution only while their bound can beat the best score.

The result is identical to the exhaustive search (verified in testing) but about 100× cheaper,
roughly 40–60 ms per match. The angular step is the smallest rotation that moves the farthest
point by one cell. The **covariance** of the match comes from the score landscape around the
best pose. In a corridor the score is flat along the corridor, so the loop edge is weak in that
direction (the aperture problem, handled automatically).

### 3.5 Pose-graph optimization
The error of an edge (i, j) is the difference between what it measured and what the current
poses predict (Grisetti et al., "A Tutorial on Graph-Based SLAM", 2010):

    e_ij = [ R_zᵀ (R_iᵀ (t_j − t_i) − t_z) ;  θ_j − θ_i − θ_z ]

The optimizer finds the poses that minimize F = Σ ρ(e_ijᵀ Ω_ij e_ij). Linearizing each error
with its Jacobians A_ij, B_ij gives the sparse normal equations H Δx = −b with
H = Σ Jᵀ Ω J and b = Σ Jᵀ Ω e. They are solved repeatedly (Gauss-Newton), with
Levenberg-Marquardt damping if a step would increase the error. Node 0 is fixed, because it
defines the map frame. A loop edge pulls its two ends together, and every seq edge in between
gives way in proportion to its uncertainty. That is how the error is spread over the loop.

**Robustness to wrong loop closures:**
- loop edges use a **Huber kernel**: an error larger than 2σ grows linearly instead of
  quadratically, so a single bad edge cannot pull the whole graph;
- after convergence, any loop edge with χ² > 16.27 (99.9% bound, 3 degrees of freedom) is
  **rejected as an outlier**, and the optimization is redone without it.

The Jacobians were checked against numerical derivatives (error < 1e-9).

### 3.6 What loop closure can and cannot fix
Loop closure enforces **consistency**: a place seen twice is drawn once, and the error of the loop
is spread over its keyframes. It cannot observe a **uniform scale error**: a rectangle that is 0.5%
too long still closes perfectly. In testing, the error left after loop closure was largely a scale
error of 0.1–0.8% along one axis, inherited from the wheel-odometry scale bias in long corridors
where the LiDAR cannot measure progress. Calibrating the wheel radius and separation (the encoder
calibration task) is what removes it.

### 3.7 When to optimize and rebuild
- A new loop edge that disagrees with the current estimate by > 5 cm or > 1° → optimize now.
- Small closures (the map is already consistent) are only stored, and optimized after 10 of them.
- If the optimized poses moved < 2 cm, the map is not rebuilt (nothing would change).
- The rebuild ray-traces every keyframe into a new grid at its optimized pose.

- While the back-end is busy (a rebuild takes a few seconds), new keyframes are **queued** and all
  checked afterwards, so the robot cannot drive through a revisited area unchecked.

### 3.8 Applying a correction while SLAM keeps running
The back-end works on a snapshot of keyframes 0..K, while the front-end keeps adding K+1..N.
When the result is ready, the front-end applies it at the start of the next scan:
- keyframes 0..K take their optimized poses;
- the newer keyframes, the current pose and every particle move rigidly with keyframe K:
  T = X_K(new) ⊕ X_K(old)⁻¹, X_i ← T ⊕ X_i;
- the rebuilt global map and local map (both built by the back-end) are swapped in, and keyframes
  K+1..N are added to them. The front-end never stops to rebuild anything.

The `map → odom` TF jumps by the correction and the odometry stays smooth, as REP 105 intends.

## 4. ROS 2 interface (changes from v1)

| Interface | Type | Notes |
| --- | --- | --- |
| `/slam/path` | `nav_msgs/Path` | Now the corrected keyframe trajectory, republished after every correction |
| `/slam/graph` | `visualization_msgs/MarkerArray` | Keyframes (blue), seq edges (grey), loop closures (pink) |
| `/slam/optimize` | `std_srvs/Trigger` | Force an optimization + map rebuild |

The log shows each loop closure, each optimization and each correction applied.

## 5. Integration

1. Copy the whole `mapping/` folder (it includes the new files in `slam_core/`).
2. Install SciPy in the Docker container (recommended):
   ```bash
   sudo apt install -y python3-scipy      # or: pip install scipy --break-system-packages
   ```
3. Rebuild:
   ```bash
   cd /ws && source /opt/ros/jazzy/setup.bash
   colcon build --packages-select main_ros --symlink-install
   source /ws/install/setup.bash
   ```
4. RViz: add **MarkerArray** `/slam/graph` and **Path** `/slam/path` (both with Durability
   Transient Local).

`setup.py` needs no change: `main_ros.mapping.slam_core` is already listed.

## 6. Testing

**Offline (no ROS):**
```bash
cd main_ros/main_ros/mapping
python3 tests/offline_sim_test.py                     # 40 x 26 m warehouse, all modes
python3 tests/offline_sim_test.py --world small       # the v1 test world
python3 tests/offline_sim_test.py --modes v1 v2-thread
```

**In Isaac Sim:** drive a loop around a rack block and back through the start. Watch the log for
`loop closure: keyframe ...` and then `Loop-closure correction applied`. In RViz, pink lines appear
between the old and new passes, and doubled walls merge into one.

## 7. Results (offline simulation, ground truth)

40 × 26 m warehouse with four long rack rows and an 8 m LiDAR. The robot drives a 205 m tour that
revisits the east corridor, the start area and two aisles. Wheel odometry has a 3% scale error and
a 4% rotation error (14.5 m of error by the end). Datasets 1–3 use different random noise.

| Dataset | v1 error (ATE) | v2 error (ATE) | v1 doubled walls | v2 doubled walls | v1 ghost walls | v2 ghost walls |
| --- | --- | --- | --- | --- | --- | --- |
| Large 1 | 22.6 cm | 14.6 cm | 23.2% | 4.7% | 31.0% | 18.5% |
| Large 2 | 6.0 cm | 4.0 cm | 0.0% | 0.0% | 0.0% | 0.0% |
| Large 3 | 17.1 cm | 12.8 cm | 11.3% | 0.5% | 27.2% | 12.6% |
| Large 3, back-end in a thread at real-time speed | 17.1 cm | 15.5 cm | 11.3% | 4.4% | 27.2% | 21.4% |
| Small 20 × 12 m (v1 test) | 8.9 cm | 8.0 cm | 0.0% | 0.0% | 3.1% | 1.8% |

*Ghost walls*: occupied cells more than 15 cm from a real wall. *Doubled walls*: the same after
removing the global scale error, so what is left is walls drawn twice or bent. Most of v2's
remaining ghost-wall percentage is a 0.8% stretch along y (see 3.6), not doubled walls.

Timing (laptop CPU, 720-beam scans): front-end 6–10 ms per scan, 95th percentile 18–24 ms with
the back-end running beside it. Loop verification 40–60 ms per match. Optimization of 1300
keyframes 12–30 ms. Map rebuild 2.7–4.8 s in the background.

## 8. Tuning and debugging

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| No loop closures logged | Drift larger than the search, or not far enough along the path | Raise `loop_search_radius` / `loop_match_window`; check `loop_min_path` |
| Closures found but rejected as outliers | Repetitive aisles matched to the wrong one | Raise `loop_min_score` / `loop_min_inliers`; reduce `loop_match_window` |
| Map flickers / rebuilds very often | Every small closure triggers a rebuild | Raise `loop_trigger_xy` and `loop_rebuild_min_correction` |
| Front-end slower while a rebuild runs | Python threads share the CPU (GIL) | Expected; raise `loop_check_every`, or raise `map_beam_step` |
| Front-end drifts more than v1 between loops, or corridors come out stretched | Local map shorter than ~2x the LiDAR range | Raise `local_map_window` |
| A correction bends straight corridors | Odometry edges too loose in heading / sideways | Lower `seq_sigma_theta*` and `seq_sigma_cross*` |
| Loops close but the map is slightly too long or short | Global scale error (loop closure cannot see it) | Calibrate wheel radius and separation |
| Want v1 behaviour | — | `loop_closure: false`, `frontend_local_map: false` |

## 9. References

- S. Thrun, W. Burgard, D. Fox, *Probabilistic Robotics*, MIT Press, 2005.
- G. Grisetti, R. Kümmerle, C. Stachniss, W. Burgard, "A tutorial on graph-based SLAM,"
  *IEEE Intell. Transp. Syst. Mag.*, 2010.
- E. Olson, "Real-time correlative scan matching," *Proc. IEEE ICRA*, 2009.
- W. Hess, D. Kohler, H. Rapp, D. Andor, "Real-time loop closure in 2D LIDAR SLAM,"
  *Proc. IEEE ICRA*, 2016.
- K. Konolige et al., "Efficient sparse pose adjustment for 2D mapping," *Proc. IEEE/RSJ IROS*, 2010.
