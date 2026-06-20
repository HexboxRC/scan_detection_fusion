#!/usr/bin/env python3
"""
scan_detection_fusion/lidar_camera_fuser.py

Pure LiDAR–camera fusion logic — rebuilt from fuser_node_v3.
No rclpy or ROS message imports.  All inputs and outputs are plain Python / NumPy types.
"""

import math
import time

import numpy as np


# ── Angle utilities ────────────────────────────────────────────────────────────

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert a quaternion to a yaw angle (radians). Only the Z-axis component matters."""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def normalize_angle(a: float) -> float:
    """Wrap an angle to [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))


# ── Class-aware footprint table (W × D in metres) ─────────────────────────────

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


# ── Main class ────────────────────────────────────────────────────────────────

class LidarCameraFuser:
    """
    Stateful LiDAR–camera object fusion.  No ROS dependency.

    Construct with tunable parameters, call fuse() on each detection batch,
    then read self.registry for the current object map.

    Construction example::

        import math
        from scan_detection_fusion.lidar_camera_fuser import LidarCameraFuser

        fuser = LidarCameraFuser(
            min_range        = 0.20,
            max_range        = 6.0,
            lidar_offset     = 0.0,
            angle_expand     = math.radians(4.0),
            ema_alpha        = 0.35,
            stale_sec        = 5.0,
            estimator        = 'adaptive',
            use_spatial_keys = True,
            spatial_bin_size = 0.75,
        )

        fuser.fuse(
            detections      = [{'label': 'chair', 'friendly_label': 'Chair',
                                 'confidence': 0.9, 'center_angle_rad': 0.1,
                                 'left_angle_rad': 0.05, 'right_angle_rad': 0.15}],
            ranges          = scan_ranges,      # list of floats from LaserScan
            angle_min       = -math.pi,
            angle_increment = 0.00581,
            robot_x=1.0, robot_y=2.0, robot_yaw=0.0, has_pose=True,
        )
        print(fuser.registry)  # {'chair_g1_g2': {'map_x': ..., ...}}

        # Footprint corners for Nav2 obstacle layer:
        for obj in fuser.registry.values():
            corners = fuser.compute_footprint(
                obj['label'], obj['distance'], obj['angle_span_rad'],
                obj['map_x'], obj['map_y'], robot_x=1.0, robot_y=2.0,
            )  # → [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]

    Public attribute:
        registry (dict): live object map, keyed by EMA key string.
            Each entry contains:
                label (str), friendly_label (str), key (str),
                distance (float, m), angle_deg (float),
                map_x (float, m), map_y (float, m),
                confidence (float), n_points (int),
                angle_span_rad (float), last_seen (float, unix time).

    Public methods:
        estimate_distance(valid_ranges)                         → (dist|None, n)
        lidar_range_for_angle(ranges, angle_min, angle_inc,    → (dist|None, n)
                              center_angle, half_width)
        correct_bearing(theta_cam, r_estimate)                  → float
        make_key(label, map_x, map_y, seen_count)              → str
        fuse(detections, ranges, angle_min, angle_inc,         → None
             robot_x, robot_y, robot_yaw, has_pose)
        compute_footprint(label, distance, angle_span,         → list[(x,y)]
                          map_x, map_y, robot_x, robot_y)
        expire_stale(now=None)                                  → None
    """

    def __init__(
        self,
        # ── base (v2 parameters) ──────────────────────────────────────────
        min_range:    float,
        max_range:    float,
        lidar_offset: float,
        angle_expand: float,
        ema_alpha:    float,
        stale_sec:    float,
        # ── B1: estimator selection ───────────────────────────────────────
        estimator:    str   = 'q1',    # 'q1'|'median'|'mean'|'trimmed_mean'|'adaptive'
        # ── B4: parallax correction ───────────────────────────────────────
        use_parallax: bool  = False,
        parallax_dx:  float = 0.0,     # metres, camera–LiDAR lateral offset
        parallax_dy:  float = 0.0,     # metres, camera–LiDAR forward offset
        # ── spatial-bin EMA keys ──────────────────────────────────────────
        use_spatial_keys: bool  = True,
        spatial_bin_size: float = 0.75,  # metres per grid cell
        # ── footprint reconstruction ──────────────────────────────────────
        footprint_width_refine: bool  = True,
        footprint_refine_tol:   float = 0.20,  # 20 % tolerance
    ) -> None:
        self.min_range    = min_range
        self.max_range    = max_range
        self.lidar_offset = lidar_offset
        self.angle_expand = angle_expand
        self.ema_alpha    = ema_alpha
        self.stale_sec    = stale_sec

        self.estimator    = estimator

        self.use_parallax = use_parallax
        self.parallax_dx  = parallax_dx
        self.parallax_dy  = parallax_dy

        self.use_spatial_keys = use_spatial_keys
        self.spatial_bin_size = spatial_bin_size

        self.footprint_width_refine = footprint_width_refine
        self.footprint_refine_tol   = footprint_refine_tol

        self.registry: dict = {}

    # ── B1: distance estimators ───────────────────────────────────────────────

    def estimate_distance(self, valid_ranges: list) -> tuple:
        """
        Apply the configured robust estimator to a pre-filtered list of ranges.

        Args:
            valid_ranges: floats already confirmed within [min_range, max_range].

        Returns:
            (distance, n_points): distance is None when valid_ranges is empty.
        """
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
            # Fall back to median when point count is too low for Q1 to be meaningful
            if n < 4:
                return float(np.median(arr)), n
            return float(arr[max(0, n // 4)]), n
        else:
            return float(arr[max(0, n // 4)]), n   # unknown → Q1 fallback

    # ── LiDAR window search ───────────────────────────────────────────────────

    def lidar_range_for_angle(
        self,
        ranges: list,
        angle_min: float,
        angle_increment: float,
        center_angle: float,
        half_width: float,
    ) -> tuple:
        """
        Collect all valid LiDAR rays within an angular window, then estimate distance.

        Args:
            ranges:          scan.ranges (iterable of floats, may contain inf/nan)
            angle_min:       scan.angle_min (radians)
            angle_increment: scan.angle_increment (radians per index step)
            center_angle:    window centre (radians)
            half_width:      half angular width of the search window (radians)

        Returns:
            (distance, n_points): always a 2-tuple; distance is None when no
            valid rays were found in the window.
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

        return self.estimate_distance(valid)

    # ── B4: parallax correction ───────────────────────────────────────────────

    def correct_bearing(self, theta_cam: float, r_estimate: float) -> float:
        """
        Correct a camera-frame bearing to a LiDAR-frame bearing, accounting
        for the physical offset between the two sensor origins.

        Args:
            theta_cam:   camera-frame bearing to the target (radians)
            r_estimate:  bootstrap range estimate to the target (metres)

        Returns:
            Corrected LiDAR-frame bearing (radians).
            Returns theta_cam unchanged when use_parallax is False or r <= 0.
        """
        if not self.use_parallax or r_estimate <= 0:
            return theta_cam
        obj_x_cam = r_estimate * math.sin(theta_cam)
        obj_y_cam = r_estimate * math.cos(theta_cam)
        obj_x_lid = obj_x_cam - self.parallax_dx
        obj_y_lid = obj_y_cam - self.parallax_dy
        return math.atan2(obj_x_lid, obj_y_lid)

    # ── Spatial-bin EMA key ───────────────────────────────────────────────────

    def make_key(
        self,
        label: str,
        map_x: float,
        map_y: float,
        seen_count: int,
    ) -> str:
        """
        Generate the EMA registry key for one detection.

        use_spatial_keys=True  →  'label_gX_gY'  (grid-cell address)
        use_spatial_keys=False →  'label' or 'label_N'  (v2 behaviour)

        Args:
            label:       COCO / detector class string
            map_x/y:     estimated map-frame position (metres)
            seen_count:  number of detections of this label already processed
                         in the current batch (used only in v2 fallback mode)

        Returns:
            String key into self.registry.
        """
        if self.use_spatial_keys:
            gx = int(round(map_x / self.spatial_bin_size))
            gy = int(round(map_y / self.spatial_bin_size))
            return f'{label}_g{gx}_{gy}'
        else:
            return label if seen_count == 0 else f'{label}_{seen_count}'

    # ── Fusion core ───────────────────────────────────────────────────────────

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
        Fuse one batch of camera detections with a LiDAR scan.

        Per detection:
          1. Optionally apply bootstrapped parallax correction (use_parallax).
          2. Find in-window LiDAR ranges; apply configured estimator.
          3. Convert (range, bearing) → robot-frame XY → map-frame XY.
          4. Generate EMA registry key via make_key().
          5. EMA-smooth position and distance against prior reading.
          6. Write to self.registry (also stores angle_span_rad for footprints).
        Stale entries (older than stale_sec) are pruned at the end.

        Args:
            detections:      list of detection dicts from the detector JSON.
                             Required keys per dict: 'label', 'center_angle_rad',
                             'left_angle_rad', 'right_angle_rad', 'confidence',
                             'friendly_label'.
            ranges:          scan.ranges (list/tuple of floats)
            angle_min:       scan.angle_min (radians)
            angle_increment: scan.angle_increment (radians)
            robot_x/y:       robot position in map frame (metres)
            robot_yaw:       robot heading in map frame (radians)
            has_pose:        pass False until the first valid pose is received;
                             objects are placed in robot-frame coords as fallback
        """
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

            # ── B4: bootstrapped parallax correction ─────────────────────────
            if self.use_parallax:
                # First pass: uncorrected estimate to bootstrap the range
                r_initial, _ = self.lidar_range_for_angle(
                    ranges, angle_min, angle_increment, center_angle, half_width
                )
                if r_initial is None:
                    continue
                # Correct all three bearings using the bootstrap range
                center_angle = self.correct_bearing(center_angle, r_initial)
                left_angle   = self.correct_bearing(left_angle,   r_initial)
                right_angle  = self.correct_bearing(right_angle,  r_initial)
                bbox_half    = abs(left_angle - right_angle) / 2.0
                half_width   = bbox_half + self.angle_expand

            # Final distance estimate with (possibly corrected) bearings
            distance, n_points = self.lidar_range_for_angle(
                ranges, angle_min, angle_increment, center_angle, half_width
            )
            if distance is None:
                continue

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
                # No pose yet — use robot-frame coords as fallback
                map_x = obj_rx
                map_y = obj_ry

            # Registry key (spatial-bin or v2 fallback)
            count = seen_labels.get(label, 0)
            key   = self.make_key(label, map_x, map_y, count)
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

        # Expire stale entries
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]

    # ── Footprint reconstruction ──────────────────────────────────────────────

    def compute_footprint(
        self,
        label: str,
        distance: float,
        angle_span: float,
        map_x: float,
        map_y: float,
        robot_x: float,
        robot_y: float,
    ) -> list:
        """
        Build a class-aware footprint polygon centred at (map_x, map_y),
        oriented so the rectangle's long axis faces the robot.

        Looks up nominal (W, D) in FOOTPRINT_TABLE[label].  If
        footprint_width_refine is True and the LiDAR arc implies a wider
        object (W_obs = 2·d·tan(span/2)), W is expanded to W_obs when it
        exceeds the table value by more than footprint_refine_tol.

        Args:
            label:      object class (key into FOOTPRINT_TABLE; falls back to
                        'default' if not found)
            distance:   range to object centre (metres)
            angle_span: arc subtended by the detection (radians); stored in
                        self.registry as 'angle_span_rad'
            map_x/y:    object centre in map frame (metres)
            robot_x/y:  robot position in map frame (metres); used to compute
                        the approach-axis bearing

        Returns:
            List of 4 (x, y) corner tuples in map frame (counter-clockwise).
        """
        dims = FOOTPRINT_TABLE.get(label, FOOTPRINT_TABLE['default'])
        W = dims['W']
        D = dims['D']

        # Width refinement: use LiDAR arc to sharpen width when evidence supports it
        if self.footprint_width_refine and distance > 0 and angle_span > 0:
            W_obs = 2.0 * distance * math.tan(angle_span / 2.0)
            if W_obs > W * (1.0 + self.footprint_refine_tol):
                W = W_obs

        # Bearing from robot to object — the "approach axis"
        beta = math.atan2(map_y - robot_y, map_x - robot_x)

        hw = W / 2.0
        hd = D / 2.0
        local_corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]

        c, s = math.cos(beta), math.sin(beta)
        return [
            (c * lx - s * ly + map_x, s * lx + c * ly + map_y)
            for lx, ly in local_corners
        ]

    # ── Stale expiry ──────────────────────────────────────────────────────────

    def expire_stale(self, now: float | None = None) -> None:
        """Remove registry entries whose last_seen is older than stale_sec."""
        if now is None:
            now = time.time()
        stale = [k for k, v in self.registry.items()
                 if now - v['last_seen'] > self.stale_sec]
        for k in stale:
            del self.registry[k]
