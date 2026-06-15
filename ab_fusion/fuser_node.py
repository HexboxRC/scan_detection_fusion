#!/usr/bin/env python3
"""
ab_fusion/fuser_node.py

Fuses 2D LiDAR scan with camera detections to produce named object
positions in both robot-frame and map-frame coordinates.

Algorithm (per detection cycle):
  1. For each camera detection (label + horizontal angular range):
       - Find all LiDAR scan points whose angle falls within the
         detection's angular range (plus a small expansion margin)
       - Take the lower-quartile distance of valid points
         (picks the nearest cluster, which is typically the object)
       - Compute object position in robot frame: (d·cosθ, d·sinθ)
       - Transform to map frame using current AMCL pose
  2. Maintain object_registry dict keyed by label (with _N suffix for
     multiple instances). Apply EMA smoothing on repeated detections.
  3. Expire objects not seen for stale_sec seconds.
  4. Publish registry as JSON + RViz MarkerArray (cylinders + labels)

Topics subscribed:
  /scan                              (sensor_msgs/LaserScan)
  /camera/detections                 (std_msgs/String)  JSON from detector_node
  /amcl_pose                         (geometry_msgs/PoseWithCovarianceStamped)

Topics published:
  /detected_objects                  (std_msgs/String)  JSON object registry
  /object_markers                    (visualization_msgs/MarkerArray)
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


# ── Utility ───────────────────────────────────────────────────────────────────

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw angle (radians) from a quaternion."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


# ── Node ──────────────────────────────────────────────────────────────────────

class FuserNode(Node):
    def __init__(self):
        super().__init__('fuser_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('stale_sec',               5.0)
        self.declare_parameter('publish_hz',              2.0)
        self.declare_parameter('lidar_angle_offset_deg',  0.0)   # tune if lidar 0° ≠ forward
        self.declare_parameter('min_detection_range',     0.20)  # m — ignore closer points
        self.declare_parameter('max_detection_range',     6.0)   # m — ignore farther points
        self.declare_parameter('angle_expand_deg',        4.0)   # expand bbox angle each side
        self.declare_parameter('ema_alpha',               0.35)  # position smoothing (0=frozen)
        self.declare_parameter('marker_lifetime_sec',     4.0)

        self.stale_sec      = float(self.get_parameter('stale_sec').value)
        self.pub_hz         = float(self.get_parameter('publish_hz').value)
        self.lidar_offset   = math.radians(float(self.get_parameter('lidar_angle_offset_deg').value))
        self.min_range      = float(self.get_parameter('min_detection_range').value)
        self.max_range      = float(self.get_parameter('max_detection_range').value)
        self.angle_expand   = math.radians(float(self.get_parameter('angle_expand_deg').value))
        self.ema_alpha      = float(self.get_parameter('ema_alpha').value)
        self.marker_life    = float(self.get_parameter('marker_lifetime_sec').value)

        # ── State ────────────────────────────────────────────────────────────
        self.latest_scan:   LaserScan = None
        self.robot_x:       float = 0.0
        self.robot_y:       float = 0.0
        self.robot_yaw:     float = 0.0
        self.has_pose:      bool  = False

        # key → { label, key, distance, angle_deg, map_x, map_y,
        #          confidence, last_seen, friendly_label }
        self.registry: dict = {}

        # ── QoS ─────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan', self._cb_scan, sensor_qos)
        self.create_subscription(String, '/camera/detections', self._cb_detections, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._cb_pose, 10
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self.obj_pub    = self.create_publisher(String,      '/detected_objects', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/object_markers',   10)

        # ── Timer ────────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / self.pub_hz, self._publish_cb)

        self.get_logger().info(
            f'fuser_node ready  stale={self.stale_sec}s  '
            f'range=[{self.min_range},{self.max_range}]m  '
            f'expand=±{math.degrees(self.angle_expand):.1f}°  hz={self.pub_hz}'
        )

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        self.latest_scan = msg

    def _cb_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        self.robot_x   = p.position.x
        self.robot_y   = p.position.y
        self.robot_yaw = quat_to_yaw(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w
        )
        self.has_pose = True

    def _cb_detections(self, msg: String):
        """Called every time detector_node publishes. Run fusion immediately."""
        if self.latest_scan is None:
            return
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Detection JSON parse error: {e}')
            return
        self._fuse(data)

    # ── Fusion core ───────────────────────────────────────────────────────────

    def _lidar_range_for_angle(self, center_angle: float, half_width: float):
        """
        Return the lower-quartile LiDAR distance for all scan points
        whose angle falls in [center_angle - half_width, center_angle + half_width].
        Returns None if no valid points found.
        """
        scan = self.latest_scan
        lo = normalize_angle(center_angle - half_width)
        hi = normalize_angle(center_angle + half_width)

        valid = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r):
                continue
            if r < self.min_range or r > self.max_range:
                continue
            raw_angle = scan.angle_min + i * scan.angle_increment + self.lidar_offset
            angle = normalize_angle(raw_angle)

            # Handle wrap-around when range crosses ±π
            if lo <= hi:
                in_range = lo <= angle <= hi
            else:
                in_range = angle >= lo or angle <= hi

            if in_range:
                valid.append(r)

        if not valid:
            return None

        valid.sort()
        # Use lower-quartile: tends to pick the nearest real surface
        return valid[max(0, len(valid) // 4)]

    def _fuse(self, data: dict):
        """Run one fusion cycle for a set of camera detections."""
        detections = data.get('detections', [])
        now = time.time()
        seen_labels: dict = {}   # track count per label for unique keys

        for det in detections:
            label        = det.get('label', 'unknown')
            friendly     = det.get('friendly_label', label)
            confidence   = det.get('confidence', 0.0)
            center_angle = float(det.get('center_angle_rad', 0.0))
            left_angle   = float(det.get('left_angle_rad',   center_angle))
            right_angle  = float(det.get('right_angle_rad',  center_angle))

            # Angular half-width of the bounding box + expansion margin
            bbox_half = abs(left_angle - right_angle) / 2.0
            half_width = bbox_half + self.angle_expand

            distance = self._lidar_range_for_angle(center_angle, half_width)
            if distance is None:
                self.get_logger().debug(
                    f'No LiDAR match for {label} at {math.degrees(center_angle):.1f}°',
                    throttle_duration_sec=2.0
                )
                continue

            # Robot-frame Cartesian position of the object
            obj_rx = distance * math.cos(center_angle)
            obj_ry = distance * math.sin(center_angle)

            # Map-frame position (rotate by robot yaw then translate)
            if self.has_pose:
                cy = math.cos(self.robot_yaw)
                sy = math.sin(self.robot_yaw)
                map_x = self.robot_x + cy * obj_rx - sy * obj_ry
                map_y = self.robot_y + sy * obj_rx + cy * obj_ry
            else:
                # Fallback: use robot-frame coords directly
                map_x = obj_rx
                map_y = obj_ry

            # Build unique key (chair, chair_1, chair_2, …)
            count = seen_labels.get(label, 0)
            key   = label if count == 0 else f'{label}_{count}'
            seen_labels[label] = count + 1

            # EMA smooth if key already exists
            if key in self.registry:
                old = self.registry[key]
                a   = self.ema_alpha
                map_x    = a * map_x    + (1 - a) * old['map_x']
                map_y    = a * map_y    + (1 - a) * old['map_y']
                distance = a * distance + (1 - a) * old['distance']

            self.registry[key] = {
                'label':         label,
                'friendly_label': friendly,
                'key':           key,
                'distance':      round(distance, 2),
                'angle_deg':     round(math.degrees(center_angle), 1),
                'map_x':         round(map_x, 3),
                'map_y':         round(map_y, 3),
                'confidence':    round(confidence, 3),
                'last_seen':     now,
            }

        # Remove stale entries
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]

    # ── Publish cycle ─────────────────────────────────────────────────────────

    def _publish_cb(self):
        # Expire before publishing
        now = time.time()
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]

        # JSON
        payload = String()
        payload.data = json.dumps({
            'timestamp': now,
            'robot_pose': {
                'x':       round(self.robot_x, 3),
                'y':       round(self.robot_y, 3),
                'yaw_deg': round(math.degrees(self.robot_yaw), 1),
            },
            'objects': list(self.registry.values()),
        })
        self.obj_pub.publish(payload)

        # RViz markers
        self._publish_markers()

    def _publish_markers(self):
        arr   = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        lt    = Duration()
        lt.sec = int(self.marker_life)

        for idx, obj in enumerate(self.registry.values()):
            # ── Cylinder at object position ──────────────────────────────────
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = 'map'
            m.ns              = 'fused_objects'
            m.id              = idx
            m.type            = Marker.CYLINDER
            m.action          = Marker.ADD
            m.pose.position.x = obj['map_x']
            m.pose.position.y = obj['map_y']
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 0.25
            m.scale.y = 0.25
            m.scale.z = 1.0
            m.color.a = 0.65
            m.color.r = 0.15
            m.color.g = 0.85
            m.color.b = 0.30
            m.lifetime = lt
            arr.markers.append(m)

            # ── Text label above cylinder ─────────────────────────────────────
            t = Marker()
            t.header.stamp    = stamp
            t.header.frame_id = 'map'
            t.ns              = 'fused_labels'
            t.id              = idx + 1000
            t.type            = Marker.TEXT_VIEW_FACING
            t.action          = Marker.ADD
            t.pose.position.x = obj['map_x']
            t.pose.position.y = obj['map_y']
            t.pose.position.z = 1.25
            t.pose.orientation.w = 1.0
            t.scale.z = 0.18
            t.color.a = 1.0
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.text    = f"{obj['friendly_label']}\n{obj['distance']:.1f} m"
            t.lifetime = lt
            arr.markers.append(t)

        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    try:
        node = FuserNode()
        rclpy.spin(node)
    except Exception as e:
        print(f'[fuser_node] Fatal: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
