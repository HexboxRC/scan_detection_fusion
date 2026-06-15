#!/usr/bin/env python3
"""
ab_fusion/lidar_camera_fuser.py

Pure LiDAR–camera fusion logic.  No rclpy or ROS message imports.
Inputs and outputs are plain Python / NumPy types.
"""

import math
import time


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class LidarCameraFuser:
    """
    Stateful LiDAR–camera fusion.

    Public attribute:
        registry (dict): keyed by object label/key; each value is a dict with
                         label, friendly_label, key, distance, angle_deg,
                         map_x, map_y, n_points, confidence, last_seen.
    """

    def __init__(
        self,
        min_range: float,
        max_range: float,
        lidar_offset: float,
        angle_expand: float,
        ema_alpha: float,
        stale_sec: float,
    ) -> None:
        self.min_range    = min_range
        self.max_range    = max_range
        self.lidar_offset = lidar_offset
        self.angle_expand = angle_expand
        self.ema_alpha    = ema_alpha
        self.stale_sec    = stale_sec
        self.registry: dict = {}

    def lidar_range_for_angle(
        self,
        ranges: list,
        angle_min: float,
        angle_increment: float,
        center_angle: float,
        half_width: float,
    ):
        """
        Find the lower-quartile LiDAR range inside an angular window.

        Args:
            ranges:          scan.ranges (iterable of floats)
            angle_min:       scan.angle_min
            angle_increment: scan.angle_increment
            center_angle:    window centre in radians
            half_width:      half-width of the window in radians

        Returns:
            (distance, n_points) tuple, or None if no valid points found.
        """
        lo = normalize_angle(center_angle - half_width)
        hi = normalize_angle(center_angle + half_width)

        valid = []
        for i, r in enumerate(ranges):
            if not math.isfinite(r):
                continue
            if r < self.min_range or r > self.max_range:
                continue
            raw   = angle_min + i * angle_increment + self.lidar_offset
            angle = normalize_angle(raw)
            if lo <= hi:
                in_range = lo <= angle <= hi
            else:
                in_range = angle >= lo or angle <= hi
            if in_range:
                valid.append(r)

        if not valid:
            return None
        valid.sort()
        distance = valid[max(0, len(valid) // 4)]
        n_points = len(valid)
        return distance, n_points

    def fuse(
        self,
        detections: list,
        ranges: list,
        angle_min: float,
        angle_increment: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        has_pose: bool,
    ) -> None:
        """
        Fuse one batch of camera detections with the current LiDAR scan.

        Updates self.registry in-place with EMA-smoothed map positions.
        Stale entries (older than stale_sec) are removed at the end.

        Args:
            detections:      list of detection dicts (from detector JSON)
            ranges:          scan.ranges
            angle_min:       scan.angle_min
            angle_increment: scan.angle_increment
            robot_x/y/yaw:  current robot pose in the map frame
            has_pose:        False until first valid pose received
        """
        now         = time.time()
        seen_labels: dict = {}

        for det in detections:
            label        = det.get('label', 'unknown')
            friendly     = det.get('friendly_label', label)
            confidence   = det.get('confidence', 0.0)
            center_angle = float(det.get('center_angle_rad', 0.0))
            left_angle   = float(det.get('left_angle_rad', center_angle))
            right_angle  = float(det.get('right_angle_rad', center_angle))

            bbox_half  = abs(left_angle - right_angle) / 2.0
            half_width = bbox_half + self.angle_expand

            result = self.lidar_range_for_angle(
                ranges, angle_min, angle_increment, center_angle, half_width
            )
            if result is None:
                continue
            distance, n_points = result

            # Robot-frame position
            obj_rx = distance * math.cos(center_angle)
            obj_ry = distance * math.sin(center_angle)

            # Map-frame position
            if has_pose:
                cy    = math.cos(robot_yaw)
                sy    = math.sin(robot_yaw)
                map_x = robot_x + cy * obj_rx - sy * obj_ry
                map_y = robot_y + sy * obj_rx + cy * obj_ry
            else:
                # No pose yet — use robot-frame as fallback (objects at map origin)
                map_x = obj_rx
                map_y = obj_ry

            # Unique key: chair, chair_1, chair_2 …
            count = seen_labels.get(label, 0)
            key   = label if count == 0 else f'{label}_{count}'
            seen_labels[label] = count + 1

            # EMA smooth
            if key in self.registry:
                old      = self.registry[key]
                a        = self.ema_alpha
                map_x    = a * map_x    + (1 - a) * old['map_x']
                map_y    = a * map_y    + (1 - a) * old['map_y']
                distance = a * distance + (1 - a) * old['distance']

            self.registry[key] = {
                'label':          label,
                'friendly_label': friendly,
                'key':            key,
                'distance':       round(distance, 2),
                'angle_deg':      round(math.degrees(center_angle), 1),
                'map_x':          round(map_x, 3),
                'map_y':          round(map_y, 3),
                'n_points':       n_points,
                'confidence':     round(confidence, 3),
                'last_seen':      now,
            }

        # Expire stale entries
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]

    def expire_stale(self, now: float | None = None) -> None:
        """Remove registry entries older than stale_sec."""
        if now is None:
            now = time.time()
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]
