# scan_detection_fusion

[![CI](https://github.com/HexboxRC/scan_detection_fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/HexboxRC/scan_detection_fusion/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![ROS 2 Jazzy](https://img.shields.io/badge/ROS_2-Jazzy-22314E.svg)](https://docs.ros.org/en/jazzy/)

A reusable ROS 2 library that fuses 2D camera object detections with a planar
LiDAR scan to produce **range-resolved, semantically labeled obstacles** and
**class-aware footprint polygons** for navigation.

A monocular detector knows *what* an object is and roughly *which direction* it
lies in, but not how far away it is. A 2D LiDAR knows precise distance at every
bearing, but nothing about *what* it is seeing. This package marries the two:
it matches each detection's bearing to the LiDAR returns at that bearing,
estimates a robust range, projects the result into the map frame, and publishes
a labeled object together with a reconstructed ground footprint.

The fusion algorithm is implemented as a plain, ROS-free Python class
(`LidarCameraFuser`) wrapped by a thin ROS 2 node, so the math can be reused and
unit-tested independently of the ROS graph.

## Status

This is an open-source ROS 2 package targeting the **Jazzy** distribution. It is
standards-based: detections arrive as `vision_msgs/Detection2DArray`, camera
geometry is read from `sensor_msgs/CameraInfo`, and sensor placement is taken
from the TF tree, so it is not tied to any particular detector or robot.

## What it produces

For each detection the node publishes a fused object carrying its class label,
estimated range, bearing, map-frame position, and the angular span it subtended.
On top of the baseline fusion, two contributions sharpen the output:

**Semantic zone routing.** LiDAR sampling is directed to the vertical region of
the bounding box where a given class actually intersects the scan plane, rather
than sampling the whole box, so the range estimate reflects the physical object
and not the background behind it.

**Class-aware footprint reconstruction.** Each object is given a full *W × D*
ground polygon (published as `geometry_msgs/PolygonStamped` for a Nav2 obstacle
layer) from a per-class size table. When the observed LiDAR arc implies a wider
object than the table assumes, the width is refined from the measured angular
span using

> W_obs = 2 · d̂ · tan( (θ_right − θ_left) / 2 )

where d̂ is the estimated range and (θ_right − θ_left) is the detection's angular
span. The polygon is oriented so its width spans perpendicular to the line of
sight and its depth along it.

## Installation

### From the ROS index (after release)

Once released into the Jazzy distribution this package will be installable with:

```bash
sudo apt install ros-jazzy-scan-detection-fusion
```

### From source (current)

```bash
cd ~/ros2_ws/src
git clone https://github.com/HexboxRC/scan_detection_fusion.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select scan_detection_fusion
source install/setup.bash
```

## Running the node

```bash
ros2 run scan_detection_fusion fuser_node
```

The following must already be publishing on the graph:

| Source | Provides |
| --- | --- |
| LiDAR driver | `/scan` (`sensor_msgs/LaserScan`) |
| Object detector | `/detections` (`vision_msgs/Detection2DArray`) |
| Camera driver | `/camera_info` (`sensor_msgs/CameraInfo`) — optional |
| SLAM Toolbox or AMCL | TF `map` → `base_footprint` |

Every topic name is a parameter, so the node remaps onto an existing graph
without code changes:

```bash
ros2 run scan_detection_fusion fuser_node --ros-args \
  -p topic_scan:=/lidar/scan -p topic_detections:=/yolo/detections
```

## Interface

### Subscribed

| Topic | Type | Notes |
| --- | --- | --- |
| `/scan` | `sensor_msgs/LaserScan` | planar scan |
| `/detections` | `vision_msgs/Detection2DArray` | pixel bounding boxes + class IDs |
| `/camera_info` | `sensor_msgs/CameraInfo` | optional; supplies focal length for pixel→angle |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | optional secondary pose |

### Published

| Topic | Type | Notes |
| --- | --- | --- |
| `/detected_objects` | `std_msgs/String` | JSON object registry |
| `/object_markers` | `visualization_msgs/MarkerArray` | RViz cylinders + labels |
| `/object_footprints` | `geometry_msgs/PolygonStamped` | one polygon per object |

### Pixel-to-angle conversion

The node converts a detection's pixel bounding box to a bearing using the camera
horizontal field of view. If `camera_info` is available it derives the focal
length from the intrinsics; otherwise it falls back to the `hfov_deg` and
`image_width` parameters. The active source is logged once at startup.

### Detection labels

Semantic zone routing and the footprint size table are keyed by **string class
labels** (e.g. `chair`, `person`, `couch`). A feeding detector must emit those
strings as the `class_id` in `Detection2DArray.results[]`. Unknown labels fall
back to a default footprint size.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `stale_sec` | `5.0` | seconds before an unseen object is dropped |
| `publish_hz` | `2.0` | publish timer frequency |
| `lidar_angle_offset_deg` | `0.0` | LiDAR mounting-angle correction |
| `min_detection_range` | `0.20` | ignore LiDAR returns closer than this (m) |
| `max_detection_range` | `6.0` | ignore LiDAR returns farther than this (m) |
| `angle_expand_deg` | `4.0` | angular padding added to each bbox edge |
| `ema_alpha` | `0.35` | EMA weight on the newest measurement (0–1) |
| `estimator` | `q1` | range estimator: `q1` \| `median` \| `mean` \| `trimmed_mean` \| `adaptive` |
| `use_parallax_correction` | `false` | enable camera–LiDAR bearing correction |
| `camera_frame` | `camera_link` | TF frame used to derive parallax offset |
| `use_spatial_keys` | `true` | grid-cell identity keys (reduce ID collisions) |
| `spatial_bin_size` | `0.75` | grid cell size for spatial keys (m) |
| `publish_footprints` | `true` | publish footprint polygons |
| `footprint_width_refine` | `true` | refine width from the observed LiDAR arc |
| `hfov_deg` | `60.0` | horizontal FOV fallback when no `camera_info` |
| `image_width` | `640` | image width fallback when no `camera_info` |
| `map_frame` | `map` | map frame |
| `base_frame` | `base_footprint` | robot base frame |

Topic-name parameters (`topic_scan`, `topic_detections`, `topic_camera_info`,
`topic_amcl_pose`, `topic_detected_objects`, `topic_object_markers`,
`topic_object_footprints`) default to the names above.

## Using the algorithm without ROS

`LidarCameraFuser` has no ROS dependency and can be used directly:

```python
import math
from scan_detection_fusion.lidar_camera_fuser import LidarCameraFuser

fuser = LidarCameraFuser(
    min_range=0.20, max_range=6.0, lidar_offset=0.0,
    angle_expand=math.radians(4.0), ema_alpha=0.35, stale_sec=5.0,
    estimator='adaptive',
)

fuser.fuse(
    detections=[{'label': 'chair', 'friendly_label': 'Chair', 'confidence': 0.9,
                 'center_angle_rad': 0.10, 'left_angle_rad': 0.15,
                 'right_angle_rad': 0.05}],
    ranges=scan_ranges, angle_min=-math.pi, angle_increment=0.00581,
    robot_x=1.0, robot_y=2.0, robot_yaw=0.0, has_pose=True,
)
print(fuser.registry)
```

## Tests

```bash
colcon test --packages-select scan_detection_fusion
colcon test-result --verbose
```

The suite covers the distance estimators, the angular-window search (including
wrap-around at ±π), parallax bearing correction, spatial-bin keying, the
map-frame transform, EMA smoothing, and footprint geometry — all without ROS or
hardware.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

A paper describing the fusion pipeline (semantic zone routing and class-aware
footprint reconstruction) is in preparation. Citation details will be added on
publication.
