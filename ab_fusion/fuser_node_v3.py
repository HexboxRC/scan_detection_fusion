#!/usr/bin/env python3
"""
ab_fusion/fuser_node_v3.py

LiDAR + Camera fusion node — v3 (Research paper build)

Improvements over v2:
  1. Parameterized distance estimator (Q1 / median / mean / trimmed / adaptive)
     — addresses Q1's silent degradation when N < 4 LiDAR points.
  2. Bootstrapped parallax correction for physical sensor offset (dx, dy)
     — addresses close-range angular-window mismatch between camera and LiDAR.
  3. Timestamp-synchronized fusion via ApproximateTimeSynchronizer
     — addresses moving-robot temporal desync between detection and TF lookup.
  4. Spatial-bin composite EMA keys (class, grid_cell)
     — addresses EMA identity collision when multiple same-class objects coexist.
  5. Class-aware geometric footprint reconstruction with LiDAR width refinement
     — addresses sparse LiDAR cross-section misrepresentation in Nav2 costmap.

Every improvement is parameter-toggleable for ablation experiments.

Topics subscribed:
  /scan                 sensor_msgs/LaserScan
  /camera/detections    std_msgs/String   (JSON from detector_node)
  /amcl_pose            geometry_msgs/PoseWithCovarianceStamped  (optional fallback)

Topics published:
  /detected_objects     std_msgs/String          (JSON registry)
  /object_markers       visualization_msgs/MarkerArray
  /object_footprints    geometry_msgs/PolygonStamped  (NEW — for Nav2 obstacle layer)
"""

import json
import math
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Header
from geometry_msgs.msg import PoseWithCovarianceStamped, PolygonStamped, Point32
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from message_filters import ApproximateTimeSynchronizer, Subscriber


# ── Utilities ─────────────────────────────────────────────────────────────────

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ── Class-aware footprint table (W x D in meters) ─────────────────────────────

FOOTPRINT_TABLE = {
    'chair':         {'W': 0.55, 'D': 0.55},
    'dining table':  {'W': 0.90, 'D': 0.90},
    'diningtable':   {'W': 0.90, 'D': 0.90},   # MobileNet-SSD label variant
    'couch':         {'W': 1.80, 'D': 0.85},
    'sofa':          {'W': 1.80, 'D': 0.85},   # legacy label
    'person':        {'W': 0.50, 'D': 0.50},
    'potted plant':  {'W': 0.40, 'D': 0.40},
    'pottedplant':   {'W': 0.40, 'D': 0.40},
    'bed':           {'W': 1.40, 'D': 2.00},
    'toilet':        {'W': 0.45, 'D': 0.70},
    'tv':            {'W': 1.00, 'D': 0.10},
    'tvmonitor':     {'W': 1.00, 'D': 0.10},
    'refrigerator':  {'W': 0.70, 'D': 0.70},
    'oven':          {'W': 0.60, 'D': 0.60},
    'sink':          {'W': 0.60, 'D': 0.50},
    'default':       {'W': 0.50, 'D': 0.50},
}


# ── Node ──────────────────────────────────────────────────────────────────────

class FuserNodeV3(Node):
    def __init__(self):
        super().__init__('fuser_node')

        # ── Parameters: base fusion (inherited from v2) ──────────────────────
        self.declare_parameter('stale_sec',               5.0)
        self.declare_parameter('publish_hz',              2.0)
        self.declare_parameter('lidar_angle_offset_deg',  0.0)
        self.declare_parameter('min_detection_range',     0.20)
        self.declare_parameter('max_detection_range',     6.0)
        self.declare_parameter('angle_expand_deg',        4.0)
        self.declare_parameter('ema_alpha',               0.35)
        self.declare_parameter('marker_lifetime_sec',     4.0)
        self.declare_parameter('map_frame',               'map')
        self.declare_parameter('base_frame',              'base_footprint')

        # ── Parameters: B1 estimator selection ───────────────────────────────
        self.declare_parameter('estimator',               'q1')
        # Options: 'q1', 'median', 'mean', 'trimmed_mean', 'adaptive'

        # ── Parameters: B4 parallax correction ───────────────────────────────
        self.declare_parameter('use_parallax_correction', False)
        self.declare_parameter('parallax_dx',             0.0)   # meters
        self.declare_parameter('parallax_dy',             0.0)   # meters

        # ── Parameters: B5 timestamp synchronization ─────────────────────────
        self.declare_parameter('use_time_sync',           False)
        self.declare_parameter('sync_slop_sec',           0.05)

        # ── Parameters: spatial-bin EMA keys ─────────────────────────────────
        self.declare_parameter('use_spatial_keys',        True)
        self.declare_parameter('spatial_bin_size',        0.75)  # meters

        # ── Parameters: D footprint reconstruction ───────────────────────────
        self.declare_parameter('publish_footprints',      True)
        self.declare_parameter('footprint_width_refine',  True)
        self.declare_parameter('footprint_refine_tol',    0.20)  # 20% tolerance

        # ── Resolve parameters ───────────────────────────────────────────────
        self.stale_sec    = float(self.get_parameter('stale_sec').value)
        self.pub_hz       = float(self.get_parameter('publish_hz').value)
        self.lidar_offset = math.radians(float(self.get_parameter('lidar_angle_offset_deg').value))
        self.min_range    = float(self.get_parameter('min_detection_range').value)
        self.max_range    = float(self.get_parameter('max_detection_range').value)
        self.angle_expand = math.radians(float(self.get_parameter('angle_expand_deg').value))
        self.ema_alpha    = float(self.get_parameter('ema_alpha').value)
        self.marker_life  = float(self.get_parameter('marker_lifetime_sec').value)
        self.map_frame    = str(self.get_parameter('map_frame').value)
        self.base_frame   = str(self.get_parameter('base_frame').value)

        self.estimator    = str(self.get_parameter('estimator').value)

        self.use_parallax = bool(self.get_parameter('use_parallax_correction').value)
        self.parallax_dx  = float(self.get_parameter('parallax_dx').value)
        self.parallax_dy  = float(self.get_parameter('parallax_dy').value)

        self.use_time_sync = bool(self.get_parameter('use_time_sync').value)
        self.sync_slop     = float(self.get_parameter('sync_slop_sec').value)

        self.use_spatial_keys = bool(self.get_parameter('use_spatial_keys').value)
        self.spatial_bin_size = float(self.get_parameter('spatial_bin_size').value)

        self.publish_footprints     = bool(self.get_parameter('publish_footprints').value)
        self.footprint_width_refine = bool(self.get_parameter('footprint_width_refine').value)
        self.footprint_refine_tol   = float(self.get_parameter('footprint_refine_tol').value)

        # ── State ────────────────────────────────────────────────────────────
        self.latest_scan: LaserScan = None
        self.robot_x:    float = 0.0
        self.robot_y:    float = 0.0
        self.robot_yaw:  float = 0.0
        self.has_pose:   bool  = False
        self.registry:   dict  = {}

        # ── TF2 listener ─────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── QoS ──────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        if self.use_time_sync:
            # B5: synchronized scan + detections, then TF lookup at detection stamp
            self.scan_sub = Subscriber(self, LaserScan, '/scan', qos_profile=sensor_qos)
            self.det_sub  = Subscriber(self, String,    '/camera/detections')
            self.sync = ApproximateTimeSynchronizer(
                [self.scan_sub, self.det_sub],
                queue_size=10,
                slop=self.sync_slop,
                allow_headerless=True   # String has no native header; use JSON timestamp
            )
            self.sync.registerCallback(self._cb_synced)
            self.get_logger().info(f'Time-sync ENABLED, slop={self.sync_slop:.3f}s')
        else:
            # v2 behavior: independent callbacks, TF lookup at publish time
            self.create_subscription(LaserScan, '/scan', self._cb_scan, sensor_qos)
            self.create_subscription(String, '/camera/detections', self._cb_detections, 10)

        # /amcl_pose is always available as secondary pose source
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._cb_amcl_pose, 10
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self.obj_pub       = self.create_publisher(String,         '/detected_objects', 10)
        self.marker_pub    = self.create_publisher(MarkerArray,    '/object_markers',   10)
        self.footprint_pub = self.create_publisher(PolygonStamped, '/object_footprints', 10)

        # ── Timer ────────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / self.pub_hz, self._publish_cb)

        self.get_logger().info(
            f'fuser_node_v3 ready\n'
            f'  estimator={self.estimator}  spatial_keys={self.use_spatial_keys}  '
            f'(bin={self.spatial_bin_size}m)\n'
            f'  parallax={self.use_parallax} dx={self.parallax_dx} dy={self.parallax_dy}\n'
            f'  time_sync={self.use_time_sync}  '
            f'footprints={self.publish_footprints}\n'
            f'  stale={self.stale_sec}s  range=[{self.min_range},{self.max_range}]m  '
            f'expand=±{math.degrees(self.angle_expand):.1f}°  hz={self.pub_hz}'
        )

    # ── Pose handling ────────────────────────────────────────────────────────

    def _cb_amcl_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        self.robot_x   = p.position.x
        self.robot_y   = p.position.y
        self.robot_yaw = quat_to_yaw(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w
        )
        self.has_pose = True

    def _update_pose_from_tf(self, stamp=None):
        """
        Look up map → base_frame transform.
        If stamp is provided (B5 mode), look up at that specific time.
        Otherwise look up the latest available transform.
        """
        try:
            target_stamp = stamp if stamp is not None else rclpy.time.Time()
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, target_stamp,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            self.robot_x   = t.x
            self.robot_y   = t.y
            self.robot_yaw = quat_to_yaw(r.x, r.y, r.z, r.w)
            self.has_pose  = True
            return True
        except (LookupException, ConnectivityException, ExtrapolationException):
            return False

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        """v2 mode: store latest scan asynchronously."""
        self.latest_scan = msg

    def _cb_detections(self, msg: String):
        """v2 mode: process detections using latest scan and latest TF."""
        if self.latest_scan is None:
            return
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Detection JSON parse error: {e}')
            return
        self._fuse(data)

    def _cb_synced(self, scan_msg: LaserScan, det_msg: String):
        """
        B5 mode: synchronized scan + detection.
        Look up TF at the camera capture timestamp embedded in the JSON,
        then fuse using both timestamp-aligned scan and pose.
        """
        self.latest_scan = scan_msg
        try:
            data = json.loads(det_msg.data)
        except Exception as e:
            self.get_logger().warn(f'Detection JSON parse error: {e}')
            return

        # Use camera capture timestamp from JSON for TF lookup
        cap_t = data.get('timestamp', None)
        if cap_t is not None:
            stamp = rclpy.time.Time(
                seconds=int(cap_t),
                nanoseconds=int((cap_t - int(cap_t)) * 1e9)
            )
            self._update_pose_from_tf(stamp=stamp.to_msg())
        else:
            self._update_pose_from_tf()

        self._fuse(data)

    # ── Parallax correction (B4) ─────────────────────────────────────────────

    def _correct_bearing(self, theta_cam: float, r_estimate: float) -> float:
        """
        Convert a camera-frame bearing to a LiDAR-frame bearing,
        accounting for the physical offset between the two sensor origins.
        Requires an estimated range for the bootstrap.
        """
        if not self.use_parallax or r_estimate <= 0:
            return theta_cam
        obj_x_cam = r_estimate * math.sin(theta_cam)
        obj_y_cam = r_estimate * math.cos(theta_cam)
        obj_x_lid = obj_x_cam - self.parallax_dx
        obj_y_lid = obj_y_cam - self.parallax_dy
        return math.atan2(obj_x_lid, obj_y_lid)

    # ── Distance estimators (B1) ─────────────────────────────────────────────

    def _estimate_distance(self, valid_ranges):
        """Apply the configured robust estimator to a list of in-window ranges."""
        n = len(valid_ranges)
        if n == 0:
            return None, 0

        arr = np.array(sorted(valid_ranges))

        if self.estimator == 'mean':
            return float(arr.mean()), n
        elif self.estimator == 'median':
            return float(np.median(arr)), n
        elif self.estimator == 'q1':
            return float(arr[max(0, n // 4)]), n
        elif self.estimator == 'trimmed_mean':
            lo, hi = np.percentile(arr, [5, 70])
            trimmed = arr[(arr >= lo) & (arr <= hi)]
            return (float(trimmed.mean()) if len(trimmed) > 0
                    else float(arr.min())), n
        elif self.estimator == 'adaptive':
            if n < 4:
                return float(np.median(arr)), n
            return float(arr[max(0, n // 4)]), n
        else:
            return float(arr[max(0, n // 4)]), n   # fallback Q1

    def _lidar_range_for_angle(self, center_angle: float, half_width: float):
        """
        Collect all LiDAR rays within [center-half, center+half], apply estimator.
        Returns (distance, n_points). n_points is 0 when no valid returns.
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
            raw   = scan.angle_min + i * scan.angle_increment + self.lidar_offset
            angle = normalize_angle(raw)
            if lo <= hi:
                in_range = lo <= angle <= hi
            else:
                in_range = angle >= lo or angle <= hi
            if in_range:
                valid.append(r)

        return self._estimate_distance(valid)

    # ── Spatial-bin EMA key generation ───────────────────────────────────────

    def _make_key(self, label: str, map_x: float, map_y: float, seen_count: int) -> str:
        """
        Generate the EMA registry key.
        - If spatial keys enabled: key = label_gridX_gridY
        - Else (v2 behavior): key = label or label_N
        """
        if self.use_spatial_keys:
            gx = int(round(map_x / self.spatial_bin_size))
            gy = int(round(map_y / self.spatial_bin_size))
            return f'{label}_g{gx}_{gy}'
        else:
            return label if seen_count == 0 else f'{label}_{seen_count}'

    # ── Fusion core ──────────────────────────────────────────────────────────

    def _fuse(self, data: dict):
        if self.latest_scan is None:
            return

        detections  = data.get('detections', [])
        now         = time.time()
        seen_labels: dict = {}

        for det in detections:
            label        = det.get('label', 'unknown')
            friendly     = det.get('friendly_label', label)
            confidence   = det.get('confidence', 0.0)
            center_angle = float(det.get('center_angle_rad', 0.0))
            left_angle   = float(det.get('left_angle_rad',  center_angle))
            right_angle  = float(det.get('right_angle_rad', center_angle))

            bbox_half  = abs(left_angle - right_angle) / 2.0
            half_width = bbox_half + self.angle_expand

            # ── B4: Bootstrapped parallax correction ─────────────────────────
            if self.use_parallax:
                # First pass: uncorrected estimate to bootstrap range
                r_initial, _ = self._lidar_range_for_angle(center_angle, half_width)
                if r_initial is None:
                    continue
                # Correct all three bearings using bootstrap range
                center_angle = self._correct_bearing(center_angle, r_initial)
                left_angle   = self._correct_bearing(left_angle,   r_initial)
                right_angle  = self._correct_bearing(right_angle,  r_initial)
                bbox_half    = abs(left_angle - right_angle) / 2.0
                half_width   = bbox_half + self.angle_expand

            # Final distance estimate
            distance, n_points = self._lidar_range_for_angle(center_angle, half_width)
            if distance is None:
                self.get_logger().debug(
                    f'No LiDAR match for {label} at {math.degrees(center_angle):.1f}°',
                    throttle_duration_sec=2.0
                )
                continue

            # Robot-frame position
            obj_rx = distance * math.cos(center_angle)
            obj_ry = distance * math.sin(center_angle)

            # Map-frame position
            if self.has_pose:
                cy    = math.cos(self.robot_yaw)
                sy    = math.sin(self.robot_yaw)
                map_x = self.robot_x + cy * obj_rx - sy * obj_ry
                map_y = self.robot_y + sy * obj_rx + cy * obj_ry
            else:
                map_x = obj_rx
                map_y = obj_ry

            # Registry key — spatial-bin or class-only
            count = seen_labels.get(label, 0)
            key   = self._make_key(label, map_x, map_y, count)
            seen_labels[label] = count + 1

            # EMA smoothing
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
                'confidence':     round(confidence, 3),
                'n_points':       n_points,
                'angle_span_rad': 2.0 * bbox_half,
                'last_seen':      now,
            }

        # Expire stale
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]

    # ── Footprint reconstruction (D) ─────────────────────────────────────────

    def _compute_footprint(self, label: str, distance: float, angle_span: float,
                            map_x: float, map_y: float):
        """
        Build a class-aware footprint polygon centered at (map_x, map_y),
        oriented to face the robot. Optionally refine width using LiDAR angular span.
        Returns: list of (x, y) corners in map frame.
        """
        dims = FOOTPRINT_TABLE.get(label, FOOTPRINT_TABLE['default'])
        W = dims['W']
        D = dims['D']

        # Width refinement using LiDAR evidence
        if self.footprint_width_refine and distance > 0 and angle_span > 0:
            W_obs = 2.0 * distance * math.tan(angle_span / 2.0)
            if W_obs > W * (1.0 + self.footprint_refine_tol):
                W = W_obs

        # Bearing from robot to object — object's "approach axis"
        beta = math.atan2(map_y - self.robot_y, map_x - self.robot_x)

        hw = W / 2.0
        hd = D / 2.0
        local_corners = [
            (-hw, -hd),
            ( hw, -hd),
            ( hw,  hd),
            (-hw,  hd),
        ]

        c, s = math.cos(beta), math.sin(beta)
        map_corners = []
        for (lx, ly) in local_corners:
            mx = c * lx - s * ly + map_x
            my = s * lx + c * ly + map_y
            map_corners.append((mx, my))
        return map_corners

    def _publish_footprints(self, stamp):
        """Publish one PolygonStamped per tracked object on /object_footprints."""
        if not self.publish_footprints:
            return
        for obj in self.registry.values():
            corners = self._compute_footprint(
                obj['label'],
                obj['distance'],
                obj.get('angle_span_rad', 0.0),
                obj['map_x'],
                obj['map_y'],
            )
            poly = PolygonStamped()
            poly.header.stamp    = stamp
            poly.header.frame_id = self.map_frame
            for (cx, cy) in corners:
                p = Point32()
                p.x = float(cx)
                p.y = float(cy)
                p.z = 0.0
                poly.polygon.points.append(p)
            self.footprint_pub.publish(poly)

    # ── Publish cycle ────────────────────────────────────────────────────────

    def _publish_cb(self):
        # In v2-style mode (no time sync), refresh pose from latest TF
        if not self.use_time_sync:
            self._update_pose_from_tf()

        # Expire stale objects
        now   = time.time()
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]

        # Publish JSON registry
        payload = String()
        payload.data = json.dumps({
            'timestamp':   now,
            'pose_source': 'tf' if self.has_pose else 'none',
            'robot_pose': {
                'x':       round(self.robot_x, 3),
                'y':       round(self.robot_y, 3),
                'yaw_deg': round(math.degrees(self.robot_yaw), 1),
            },
            'objects': list(self.registry.values()),
        })
        self.obj_pub.publish(payload)

        # Publish markers and footprints
        stamp = self.get_clock().now().to_msg()
        self._publish_markers(stamp)
        self._publish_footprints(stamp)

    def _publish_markers(self, stamp):
        arr   = MarkerArray()
        lt    = Duration()
        lt.sec = int(self.marker_life)

        for idx, obj in enumerate(self.registry.values()):
            m = Marker()
            m.header.stamp    = stamp
            m.header.frame_id = self.map_frame
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

            t = Marker()
            t.header.stamp    = stamp
            t.header.frame_id = self.map_frame
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
        node = FuserNodeV3()
        rclpy.spin(node)
    except Exception as e:
        print(f'[fuser_node_v3] Fatal: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()