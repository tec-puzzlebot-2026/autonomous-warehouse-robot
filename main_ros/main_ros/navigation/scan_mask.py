"""The carried box's footprint in a LiDAR scan (no ROS)."""
# Author: Raymundo Gomez

import numpy as np


def box_mask(ranges, angle_min, angle_inc, mount, x_min, x_max, half_width):
    """Boolean mask of the beams whose endpoints lie inside the box rectangle.

    ranges: (N,) array; mount: (x, y, yaw) of the LiDAR in base_link.
    """
    r = np.asarray(ranges, dtype=np.float64)
    a = angle_min + angle_inc * np.arange(r.size) + mount[2]
    with np.errstate(invalid='ignore'):
        x = mount[0] + r * np.cos(a)
        y = mount[1] + r * np.sin(a)
        inside = (x >= x_min) & (x <= x_max) & (np.abs(y) <= half_width) & np.isfinite(r)
    return inside
