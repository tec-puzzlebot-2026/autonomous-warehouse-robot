"""sensor_msgs/Image <-> numpy conversion without cv_bridge.

On the Jetson Nano, cv_bridge is often linked against a different OpenCV than
the CUDA/GStreamer-enabled one used by Python, which leads to crashes or
"undefined symbol" errors. These helpers avoid that dependency entirely.
"""
import array

import cv2
import numpy as np

# encoding -> (numpy dtype, channels)
_ENCODINGS = {
    'bgr8': (np.uint8, 3),
    'rgb8': (np.uint8, 3),
    'bgra8': (np.uint8, 4),
    'rgba8': (np.uint8, 4),
    'mono8': (np.uint8, 1),
    '8UC1': (np.uint8, 1),
    '8UC3': (np.uint8, 3),
    'mono16': (np.uint16, 1),
    '16UC1': (np.uint16, 1),
    'yuv422_yuy2': (np.uint8, 2),
    'yuyv': (np.uint8, 2),
}


def imgmsg_to_bgr(msg):
    """Convert a sensor_msgs/Image to a contiguous BGR uint8 numpy array."""
    enc = msg.encoding
    if enc not in _ENCODINGS:
        raise ValueError('Unsupported image encoding: %s' % enc)
    dtype, ch = _ENCODINGS[enc]
    itemsize = np.dtype(dtype).itemsize
    row_bytes = msg.width * ch * itemsize

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    raw = raw.reshape(msg.height, msg.step)[:, :row_bytes]
    img = np.ascontiguousarray(raw).view(dtype).reshape(msg.height, msg.width, ch)
    if itemsize > 1 and bool(msg.is_bigendian) != (np.little_endian is False):
        img = img.byteswap()

    if enc in ('bgr8', '8UC3'):
        return img
    if enc == 'rgb8':
        return np.ascontiguousarray(img[:, :, ::-1])
    if enc == 'bgra8':
        return np.ascontiguousarray(img[:, :, :3])
    if enc == 'rgba8':
        return np.ascontiguousarray(img[:, :, 2::-1])
    if enc in ('yuv422_yuy2', 'yuyv'):
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)
    # mono
    gray = img[:, :, 0]
    if gray.dtype != np.uint8:
        gray = (gray >> 8).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def bgr_to_imgmsg(img, header, msg_cls):
    """Build a bgr8 sensor_msgs/Image (msg_cls) from a BGR uint8 array."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError('Expected HxWx3 uint8 BGR image')
    img = np.ascontiguousarray(img)
    msg = msg_cls()
    msg.header = header
    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    # Assigning an array.array skips rclpy's slow per-element validation.
    data = array.array('B')
    data.frombytes(img.tobytes())
    msg.data = data
    return msg