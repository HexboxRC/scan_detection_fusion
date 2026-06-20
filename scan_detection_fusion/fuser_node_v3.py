#!/usr/bin/env python3
"""
scan_detection_fusion/fuser_node_v3.py  —  LiDAR + Camera fusion node — v3 with optional time-synchronized scan + detection (B5).
================================================================================

Build & run
-----------
  # From the workspace root (one level above scan_detection_fusion/):
  colcon build --packages-select scan_detection_fusion
  source install/setup.bash
  ros2 run scan_detection_fusion fuser_node_v3

Upstream nodes that must already be running
-------------------------------------------
  • LiDAR driver          — publishes /scan (sensor_msgs/LaserScan)
  • Camera detector node  — publishes /camera/detections (std_msgs/String, JSON)
  • SLAM Toolbox or AMCL  — broadcasts TF map → base_footprint
                            (AMCL also publishes /amcl_pose as a secondary source)

Topics subscribed
-----------------
  /scan                   sensor_msgs/LaserScan
  /camera/detections      std_msgs/String         (JSON from detector_node)
  /amcl_pose              geometry_msgs/PoseWithCovarianceStamped  (optional)

Topics published
----------------
  /detected_objects       std_msgs/String          (JSON registry, 2 Hz)
  /object_markers         visualization_msgs/MarkerArray
  /object_footprints      geometry_msgs/PolygonStamped  (one per object, for Nav2)

Key parameters and defaults
---------------------------
  stale_sec               5.0     seconds before an unseen object is dropped
  publish_hz              2.0     publish timer frequency
  lidar_angle_offset_deg  0.0     LiDAR mounting angle correction
  min_detection_range     0.20    ignore LiDAR returns closer than this (m)
  max_detection_range     6.0     ignore LiDAR returns farther than this (m)
  angle_expand_deg        4.0     angular padding added to each bbox edge (°)
  ema_alpha               0.35    EMA weight on newest measurement (0–1)
  marker_lifetime_sec     4.0     RViz marker lifetime
  map_frame               'map'
  base_frame              'base_footprint'
  estimator               'q1'    distance estimator: q1 | median | mean |
                                  trimmed_mean | adaptive
  use_parallax_correction False   enable sensor-offset bearing correction
  parallax_dx             0.0     camera–LiDAR lateral offset (m)
  parallax_dy             0.0     camera–LiDAR forward offset (m)
  use_time_sync           False   enable ApproximateTimeSynchronizer
  sync_slop_sec           0.05    time-sync tolerance window (s)
  use_spatial_keys        True    grid-cell EMA keys (prevents ID collisions)
  spatial_bin_size        0.75    grid cell size for spatial keys (m)
  publish_footprints      True    publish PolygonStamped on /object_footprints
  footprint_width_refine  True    widen footprint when LiDAR arc implies it
  footprint_refine_tol    0.20    tolerance before width override kicks in
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, PolygonStamped, Point32
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from tf2_ros import (Buffer, TransformListener,
                     LookupException, ConnectivityException, ExtrapolationException)
from message_filters import ApproximateTimeSynchronizer, Subscriber

from scan_detection_fusion.lidar_camera_fuser import LidarCameraFuser, quat_to_yaw


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

        # ── Parameters: B4 parallax correction ───────────────────────────────
        self.declare_parameter('use_parallax_correction', False)
        self.declare_parameter('parallax_dx',             0.0)
        self.declare_parameter('parallax_dy',             0.0)

        # ── Parameters: B5 timestamp synchronization ─────────────────────────
        self.declare_parameter('use_time_sync',           False)
        self.declare_parameter('sync_slop_sec',           0.05)

        # ── Parameters: spatial-bin EMA keys ─────────────────────────────────
        self.declare_parameter('use_spatial_keys',        True)
        self.declare_parameter('spatial_bin_size',        0.75)

        # ── Parameters: footprint reconstruction ─────────────────────────────
        self.declare_parameter('publish_footprints',      True)
        self.declare_parameter('footprint_width_refine',  True)
        self.declare_parameter('footprint_refine_tol',    0.20)

        # ── Resolve parameters ───────────────────────────────────────────────
        stale_sec    = float(self.get_parameter('stale_sec').value)
        pub_hz       = float(self.get_parameter('publish_hz').value)
        lidar_offset = math.radians(float(self.get_parameter('lidar_angle_offset_deg').value))
        min_range    = float(self.get_parameter('min_detection_range').value)
        max_range    = float(self.get_parameter('max_detection_range').value)
        angle_expand = math.radians(float(self.get_parameter('angle_expand_deg').value))
        ema_alpha    = float(self.get_parameter('ema_alpha').value)

        self.marker_life = float(self.get_parameter('marker_lifetime_sec').value)
        self.map_frame   = str(self.get_parameter('map_frame').value)
        self.base_frame  = str(self.get_parameter('base_frame').value)
        self.pub_hz      = pub_hz

        estimator    = str(self.get_parameter('estimator').value)

        use_parallax = bool(self.get_parameter('use_parallax_correction').value)
        parallax_dx  = float(self.get_parameter('parallax_dx').value)
        parallax_dy  = float(self.get_parameter('parallax_dy').value)

        self.use_time_sync = bool(self.get_parameter('use_time_sync').value)
        self.sync_slop     = float(self.get_parameter('sync_slop_sec').value)

        use_spatial_keys = bool(self.get_parameter('use_spatial_keys').value)
        spatial_bin_size = float(self.get_parameter('spatial_bin_size').value)

        self.publish_footprints     = bool(self.get_parameter('publish_footprints').value)
        footprint_width_refine      = bool(self.get_parameter('footprint_width_refine').value)
        footprint_refine_tol        = float(self.get_parameter('footprint_refine_tol').value)

        # ── Fusion class (all math lives here) ──────────────────────────────
        self.fuser = LidarCameraFuser(
            min_range    = min_range,
            max_range    = max_range,
            lidar_offset = lidar_offset,
            angle_expand = angle_expand,
            ema_alpha    = ema_alpha,
            stale_sec    = stale_sec,
            estimator    = estimator,
            use_parallax = use_parallax,
            parallax_dx  = parallax_dx,
            parallax_dy  = parallax_dy,
            use_spatial_keys       = use_spatial_keys,
            spatial_bin_size       = spatial_bin_size,
            footprint_width_refine = footprint_width_refine,
            footprint_refine_tol   = footprint_refine_tol,
        )

        # ── Pose state (updated from TF / amcl_pose) ─────────────────────────
        self.latest_scan: LaserScan = None
        self.robot_x:    float = 0.0
        self.robot_y:    float = 0.0
        self.robot_yaw:  float = 0.0
        self.has_pose:   bool  = False

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
            # B5: synchronized scan + detections; TF lookup at detection stamp
            self.scan_sub = Subscriber(self, LaserScan, '/scan', qos_profile=sensor_qos)
            self.det_sub  = Subscriber(self, String,    '/camera/detections')
            self.sync = ApproximateTimeSynchronizer(
                [self.scan_sub, self.det_sub],
                queue_size=10,
                slop=self.sync_slop,
                allow_headerless=True   # String has no native header; JSON timestamp used
            )
            self.sync.registerCallback(self._cb_synced)
            self.get_logger().info(f'Time-sync ENABLED, slop={self.sync_slop:.3f}s')
        else:
            # v2 behaviour: independent callbacks, TF lookup at publish time
            self.create_subscription(LaserScan, '/scan', self._cb_scan, sensor_qos)
            self.create_subscription(String, '/camera/detections', self._cb_detections, 10)

        # /amcl_pose always available as secondary pose source
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._cb_amcl_pose, 10
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self.obj_pub       = self.create_publisher(String,         '/detected_objects',  10)
        self.marker_pub    = self.create_publisher(MarkerArray,    '/object_markers',    10)
        self.footprint_pub = self.create_publisher(PolygonStamped, '/object_footprints', 10)

        # ── Timer ────────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / self.pub_hz, self._publish_cb)

        self.get_logger().info(
            f'fuser_node_v3 ready\n'
            f'  estimator={self.fuser.estimator}  '
            f'spatial_keys={self.fuser.use_spatial_keys}  '
            f'(bin={self.fuser.spatial_bin_size}m)\n'
            f'  parallax={self.fuser.use_parallax} '
            f'dx={self.fuser.parallax_dx} dy={self.fuser.parallax_dy}\n'
            f'  time_sync={self.use_time_sync}  '
            f'footprints={self.publish_footprints}\n'
            f'  stale={self.fuser.stale_sec}s  '
            f'range=[{self.fuser.min_range},{self.fuser.max_range}]m  '
            f'expand=±{math.degrees(self.fuser.angle_expand):.1f}°  hz={self.pub_hz}'
        )

    # ── Pose handling ────────────────────────────────────────────────────────

    def _cb_amcl_pose(self, msg: PoseWithCovarianceStamped):
        """Secondary pose update — used when AMCL is running (nav2.launch.py mode)."""
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
        Primary pose update — lookup map → base_frame via TF2.
        stamp: if provided (B5 mode), look up at that specific time;
               otherwise look up the latest available transform.
        Returns True on success, False if the transform is not yet available.
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

    # ── Scan / detection callbacks ────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        """v2 mode: store latest scan asynchronously."""
        self.latest_scan = msg

    def _cb_detections(self, msg: String):
        """v2 mode: process detections using the latest stored scan and latest TF."""
        if self.latest_scan is None:
            return
        try:
            data = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Detection JSON parse error: {e}')
            return

        scan = self.latest_scan
        self.fuser.fuse(
            detections      = data.get('detections', []),
            ranges          = scan.ranges,
            angle_min       = scan.angle_min,
            angle_increment = scan.angle_increment,
            robot_x         = self.robot_x,
            robot_y         = self.robot_y,
            robot_yaw       = self.robot_yaw,
            has_pose        = self.has_pose,
        )

    def _cb_synced(self, scan_msg: LaserScan, det_msg: String):
        """
        B5 mode: scan and detection arrive time-aligned.
        TF is looked up at the camera capture timestamp embedded in the JSON,
        then fuse() is called with the timestamp-aligned scan and pose.
        """
        self.latest_scan = scan_msg
        try:
            data = json.loads(det_msg.data)
        except Exception as e:
            self.get_logger().warn(f'Detection JSON parse error: {e}')
            return

        # Use the camera capture timestamp from the JSON payload for TF lookup
        cap_t = data.get('timestamp', None)
        if cap_t is not None:
            stamp = rclpy.time.Time(
                seconds=int(cap_t),
                nanoseconds=int((cap_t - int(cap_t)) * 1e9)
            )
            self._update_pose_from_tf(stamp=stamp.to_msg())
        else:
            self._update_pose_from_tf()

        self.fuser.fuse(
            detections      = data.get('detections', []),
            ranges          = scan_msg.ranges,
            angle_min       = scan_msg.angle_min,
            angle_increment = scan_msg.angle_increment,
            robot_x         = self.robot_x,
            robot_y         = self.robot_y,
            robot_yaw       = self.robot_yaw,
            has_pose        = self.has_pose,
        )

    # ── Publish cycle ────────────────────────────────────────────────────────

    def _publish_cb(self):
        # In v2-style mode (no time sync), refresh pose from latest TF
        if not self.use_time_sync:
            self._update_pose_from_tf()

        # Expire stale objects
        now = time.time()
        self.fuser.expire_stale(now)

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
            'objects': list(self.fuser.registry.values()),
        })
        self.obj_pub.publish(payload)

        # Publish markers and footprints
        stamp = self.get_clock().now().to_msg()
        self._publish_markers(stamp)
        self._publish_footprints(stamp)

    def _publish_markers(self, stamp):
        arr = MarkerArray()
        lt  = Duration()
        lt.sec = int(self.marker_life)

        for idx, obj in enumerate(self.fuser.registry.values()):
            # Cylinder marker
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

            # Text label
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

    def _publish_footprints(self, stamp):
        """Publish one PolygonStamped per tracked object on /object_footprints."""
        if not self.publish_footprints:
            return
        for obj in self.fuser.registry.values():
            corners = self.fuser.compute_footprint(
                obj['label'],
                obj['distance'],
                obj.get('angle_span_rad', 0.0),
                obj['map_x'],
                obj['map_y'],
                self.robot_x,
                self.robot_y,
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
