"""
Keyframes and the front-end's local map.

A keyframe is a scan that was inserted into the map. Its pose lives in the
pose graph (it changes when loops are closed); everything stored here is
expressed in the robot frame and never changes, so the map can be rebuilt
from scratch at any time with corrected poses.

The front-end matches new scans against a LOCAL map built only from the most
recent keyframes (a sliding window along the travelled path), as in Karto and
Cartographer. Matching against the whole global map would make the front-end
"snap" onto old walls when the robot comes back with accumulated drift, hiding
the error inside one step instead of letting the pose graph distribute it.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .occupancy_grid import OccupancyGrid
from .se2 import transform_points


@dataclass(frozen=True)
class Keyframe:
    id: int
    stamp: float
    map_hits: np.ndarray        # (K, 2) endpoints in base_link, used for mapping
    map_misses: np.ndarray      # (J, 2) free-space ray ends in base_link
    sensor_origin: np.ndarray   # (2,) LiDAR position in base_link
    loop_points: np.ndarray     # (P, 2) points for loop-closure matching
    path_dist: float            # distance travelled from the first keyframe [m]


def keyframe_from_scan(kf_id, stamp, scan, path_dist):
    return Keyframe(
        id=kf_id,
        stamp=float(stamp),
        map_hits=np.asarray(scan.map_hits_base, dtype=np.float32),
        map_misses=np.asarray(scan.map_misses_base, dtype=np.float32),
        sensor_origin=np.asarray(scan.sensor_origin_base, dtype=np.float64),
        loop_points=np.asarray(scan.loop_base, dtype=np.float32),
        path_dist=float(path_dist),
    )


def insert_keyframe(grid, kf, pose, sign=1.0):
    """Ray-trace one keyframe into a grid at the given map pose."""
    sensor = transform_points(pose, kf.sensor_origin[None, :])[0]
    hits = transform_points(pose, kf.map_hits.astype(np.float64))
    misses = transform_points(pose, kf.map_misses.astype(np.float64))
    grid.integrate_scan(sensor, hits, misses, sign)


class LocalMap:
    """Sliding window of the most recent keyframes, with exact removal."""

    def __init__(self, grid_kwargs, window_m=8.0, min_keyframes=10, max_keyframes=80):
        self.grid_kwargs = dict(grid_kwargs)
        self.grid_kwargs["clamp"] = False
        self.window_m = float(window_m)
        self.min_keyframes = int(min_keyframes)
        self.max_keyframes = int(max_keyframes)
        self.grid = OccupancyGrid(**self.grid_kwargs)
        self.members = deque()          # (keyframe, pose used for insertion)

    def add(self, kf, pose):
        pose = np.array(pose, dtype=np.float64)
        insert_keyframe(self.grid, kf, pose, +1.0)
        self.members.append((kf, pose))
        while len(self.members) > self.min_keyframes and (
                len(self.members) > self.max_keyframes
                or kf.path_dist - self.members[0][0].path_dist > self.window_m):
            old_kf, old_pose = self.members.popleft()
            insert_keyframe(self.grid, old_kf, old_pose, -1.0)

    def rebuild(self, keyframes_and_poses):
        """Rebuild from scratch (after a loop closure moved the poses)."""
        self.grid = OccupancyGrid(**self.grid_kwargs)
        self.members.clear()
        for kf, pose in keyframes_and_poses:
            self.add(kf, pose)

    def member_ids(self):
        return [kf.id for kf, _ in self.members]
