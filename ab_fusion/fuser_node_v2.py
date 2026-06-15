#!/usr/bin/env python3
"""
ab_fusion/fuser_node_v2.py

LiDAR + Camera fusion node — v2 (TXSEF demo build)

Thin ROS wrapper around LidarCameraFuser.  All fusion/geometry math lives in
lidar_camera_fuser.py; this file handles only subscriptions, publishers,
TF lookups, parameter reads, and message building.

Changes from v1:
  - Pose source: TF lookup (map → base_footprint) FIRST, /amcl_pose as fallback.
    v1 only subscribed to /amcl_pose, which AMCL publishes.  SLAM Toolbox does
    NOT publish /amcl_pose — it only broadcasts the map→odom TF.  Without this
    fix, has_pose stays False forever during SLAM mode and every object is placed
    at robot-frame coords (0,0) in the map instead of the real map position.
  - TF lookup runs in the publish timer (2 Hz) — cheap and sufficient.
  - /amcl_pose subscription kept as a secondary update so the node also works
    in the normal AMCL+map workflow (nav2.launch.py) without changes.

Algorithm (unchanged from v1):
  For each camera detection (label + horizontal angular span from detector_node):
    1. Find all LiDAR scan points inside the detection's angular window.
    2. Take the lower-quartile range → closest real surface = the object.
    3. Convert (range, angle) → robot-frame XY → map-frame XY via current pose.
    4. EMA-smooth repeated detections, expire stale ones.
  Publish registry as JSON on /detected_objects and RViz MarkerArray on /object_markers.

Topics subscribed:
  /scan                 sensor_msgs/LaserScan
  /camera/detections    std_msgs/String   (JSON from detector_node)
  /amcl_pose            geometry_msgs/PoseWithCovarianceStamped  (optional / AMCL mode)

Topics published:
  /detected_objects     std_msgs/String   (JSON registry)
  /object_markers       visualization_msgs/MarkerArray
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

from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException

from ab_fusion.lidar_camera_fuser import LidarCameraFuser, quat_to_yaw


# ── Node ──────────────────────────────────────────────────────────────────────

class FuserNodeV2(Node):
    def __init__(self):
        super().__init__('fuser_node')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('stale_sec',               5.0)
        self.declare_parameter('publish_hz',              2.0)
        self.declare_parameter('lidar_angle_offset_deg',  0.0)
        self.declare_parameter('min_detection_range',     0.20)
        self.declare_parameter('max_detection_range',     6.0)
        self.declare_parameter('angle_expand_deg',        4.0)
        self.declare_parameter('ema_alpha',               0.35)
        self.declare_parameter('marker_lifetime_sec',     4.0)
        # Frames used for TF pose lookup
        self.declare_parameter('map_frame',               'map')
        self.declare_parameter('base_frame',              'base_footprint')

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

        # ── Fusion class (all math lives here) ──────────────────────────────
        self.fuser = LidarCameraFuser(
            min_range    = min_range,
            max_range    = max_range,
            lidar_offset = lidar_offset,
            angle_expand = angle_expand,
            ema_alpha    = ema_alpha,
            stale_sec    = stale_sec,
        )

        # ── Pose state (updated from TF / amcl_pose) ─────────────────────
        self.latest_scan: LaserScan = None
        self.robot_x:    float = 0.0
        self.robot_y:    float = 0.0
        self.robot_yaw:  float = 0.0
        self.has_pose:   bool  = False

        # ── TF2 listener (primary pose source — works with SLAM + AMCL) ─────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── QoS ─────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(LaserScan, '/scan', self._cb_scan, sensor_qos)
        self.create_subscription(String, '/camera/detections', self._cb_detections, 10)
        # /amcl_pose kept as secondary — if AMCL is running it updates pose too
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._cb_amcl_pose, 10
        )

        # ── Publishers ───────────────────────────────────────────────────────
        self.obj_pub    = self.create_publisher(String,      '/detected_objects', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/object_markers',   10)

        # ── Timer ────────────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / self.pub_hz, self._publish_cb)

        self.get_logger().info(
            f'fuser_node_v2 ready  '
            f'pose=TF({self.map_frame}→{self.base_frame})+amcl_fallback  '
            f'stale={self.fuser.stale_sec}s  '
            f'range=[{self.fuser.min_range},{self.fuser.max_range}]m  '
            f'expand=±{math.degrees(self.fuser.angle_expand):.1f}°  hz={self.pub_hz}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_scan(self, msg: LaserScan):
        self.latest_scan = msg

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

    def _update_pose_from_tf(self):
        """
        Primary pose update — lookup map → base_footprint via TF2.
        This works with SLAM Toolbox (which only broadcasts TF, no /amcl_pose).
        Called in the publish timer so it runs at 2 Hz — plenty for object tracking.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),          # latest available
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            t = tf.transform.translation
            r = tf.transform.rotation
            self.robot_x   = t.x
            self.robot_y   = t.y
            self.robot_yaw = quat_to_yaw(r.x, r.y, r.z, r.w)
            self.has_pose  = True
        except (LookupException, ConnectivityException, ExtrapolationException):
            # TF not yet available — has_pose may still be True from /amcl_pose
            pass

    def _cb_detections(self, msg: String):
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

    # ── Publish cycle ─────────────────────────────────────────────────────────

    def _publish_cb(self):
        # 1. Update pose from TF (primary source — works with SLAM + AMCL)
        self._update_pose_from_tf()

        # 2. Expire stale objects
        now = time.time()
        self.fuser.expire_stale(now)

        # 3. Publish JSON
        payload = String()
        payload.data = json.dumps({
            'timestamp':  now,
            'pose_source': 'tf' if self.has_pose else 'none',
            'robot_pose': {
                'x':       round(self.robot_x, 3),
                'y':       round(self.robot_y, 3),
                'yaw_deg': round(math.degrees(self.robot_yaw), 1),
            },
            'objects': list(self.fuser.registry.values()),
        })
        self.obj_pub.publish(payload)

        # 4. RViz markers
        self._publish_markers()

    def _publish_markers(self):
        arr   = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        lt    = Duration()
        lt.sec = int(self.marker_life)

        for idx, obj in enumerate(self.fuser.registry.values()):
            # Cylinder
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

            # Text label
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
        node = FuserNodeV2()
        rclpy.spin(node)
    except Exception as e:
        print(f'[fuser_node_v2] Fatal: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
