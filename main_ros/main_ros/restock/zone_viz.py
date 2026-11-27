"""
zone_viz.py — the ArUco zones as a Foxglove / RViz MarkerArray
================================================================
One square per zone (filled, translucent), its outline and its label. The drop zone
is green, every other ArUco zone blue. Published latched, so Foxglove shows them as
soon as it subscribes:

    /landmark_mapper/zones   live, while the map is being made and improved
    /restock/zones           the zones of the mission folder, during restocking
"""
# Author: Raymundo Gomez

import math

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from main_ros.restock.zones import DROP_ZONE

GREEN = (0.20, 0.80, 0.30)
BLUE = (0.20, 0.50, 0.95)


def _color(m, rgb, a):
    m.color.r, m.color.g, m.color.b, m.color.a = float(rgb[0]), float(rgb[1]), float(rgb[2]), a


def zone_markers(zones, stamp, frame='map', ns='zones'):
    """MarkerArray for [Zone]: DELETEALL first, so zones that disappeared go away."""
    arr = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    for i, z in enumerate(zones):
        rgb = GREEN if z.label == DROP_ZONE else BLUE
        fill = Marker()
        fill.header.frame_id, fill.header.stamp = frame, stamp
        fill.ns, fill.id, fill.type, fill.action = ns + '/area', i, Marker.CUBE, Marker.ADD
        fill.pose.position.x, fill.pose.position.y, fill.pose.position.z = \
            float(z.center[0]), float(z.center[1]), 0.005
        fill.pose.orientation.z, fill.pose.orientation.w = math.sin(z.yaw / 2), math.cos(z.yaw / 2)
        fill.scale.x = fill.scale.y = float(z.size)
        fill.scale.z = 0.01
        _color(fill, rgb, 0.25)
        arr.markers.append(fill)

        edge = Marker()
        edge.header.frame_id, edge.header.stamp = frame, stamp
        edge.ns, edge.id, edge.type, edge.action = ns + '/edge', i, Marker.LINE_STRIP, Marker.ADD
        edge.pose.orientation.w = 1.0
        edge.scale.x = 0.03
        for x, y in z.corners + [z.corners[0]]:
            edge.points.append(Point(x=float(x), y=float(y), z=0.02))
        _color(edge, rgb, 1.0)
        arr.markers.append(edge)

        text = Marker()
        text.header.frame_id, text.header.stamp = frame, stamp
        text.ns, text.id, text.type, text.action = ns + '/label', i, Marker.TEXT_VIEW_FACING, \
            Marker.ADD
        text.pose.position.x, text.pose.position.y, text.pose.position.z = \
            float(z.center[0]), float(z.center[1]), 0.3
        text.pose.orientation.w = 1.0
        text.scale.z = 0.18
        name = 'DROP ZONE' if z.label == DROP_ZONE else z.label
        text.text = f'{name} (ArUco {z.marker_id})'
        _color(text, (1.0, 1.0, 1.0), 1.0)
        arr.markers.append(text)
    return arr
