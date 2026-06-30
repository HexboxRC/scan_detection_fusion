#!/usr/bin/env python3
"""
Unit tests for scan_detection_fusion.lidar_camera_fuser.LidarCameraFuser

Targets the pure, deterministic geometry/math: distance estimators, the
LiDAR angular-window search (including wrap-around), parallax bearing
correction, spatial-bin keying, the map-frame transform inside fuse(),
EMA smoothing, and footprint reconstruction (W_obs).

No ROS, no hardware — runs anywhere with numpy + pytest.
"""

import math
import pytest

try:  # installed package (colcon test)
    from scan_detection_fusion.lidar_camera_fuser import (
        LidarCameraFuser, quat_to_yaw, normalize_angle, FOOTPRINT_TABLE,
    )
except ImportError:  # standalone run next to the module
    from lidar_camera_fuser import (
        LidarCameraFuser, quat_to_yaw, normalize_angle, FOOTPRINT_TABLE,
    )

INF = float('inf')


def make_fuser(**kw):
    defaults = dict(
        min_range=0.20, max_range=6.0, lidar_offset=0.0,
        angle_expand=math.radians(4.0), ema_alpha=0.35, stale_sec=5.0,
    )
    defaults.update(kw)
    return LidarCameraFuser(**defaults)


def make_scan(n=720, angle_min=-math.pi, hits=None):
    """Scan of all-inf ranges; `hits` = {index: range} sets specific returns."""
    inc = (2 * math.pi) / n
    ranges = [INF] * n
    if hits:
        for i, r in hits.items():
            ranges[i] = r
    return ranges, angle_min, inc


def index_for_angle(angle, n=720, angle_min=-math.pi):
    inc = (2 * math.pi) / n
    return int(round((angle - angle_min) / inc))


# ── angle utilities ────────────────────────────────────────────────────────

def test_quat_to_yaw_identity():
    assert quat_to_yaw(0, 0, 0, 1) == pytest.approx(0.0)

def test_quat_to_yaw_90deg():
    # rotation of +90° about z
    assert quat_to_yaw(0, 0, math.sin(math.pi/4), math.cos(math.pi/4)) == pytest.approx(math.pi/2)

def test_normalize_angle_wraps():
    assert normalize_angle(3 * math.pi) == pytest.approx(math.pi)
    assert abs(normalize_angle(-3 * math.pi)) == pytest.approx(math.pi)  # wraps to ±pi
    assert normalize_angle(0.5) == pytest.approx(0.5)


# ── estimate_distance ──────────────────────────────────────────────────────

def test_estimate_empty_returns_none():
    f = make_fuser()
    assert f.estimate_distance([]) == (None, 0)

def test_estimate_q1_picks_lower_quartile():
    f = make_fuser(estimator='q1')
    # n=8 -> index 8//4 = 2 -> 3rd smallest = 3.0
    dist, n = f.estimate_distance([8, 7, 6, 5, 4, 3, 2, 1])
    assert n == 8
    assert dist == pytest.approx(3.0)

def test_estimate_mean():
    f = make_fuser(estimator='mean')
    dist, n = f.estimate_distance([1, 2, 3, 4])
    assert dist == pytest.approx(2.5)

def test_estimate_median():
    f = make_fuser(estimator='median')
    dist, _ = f.estimate_distance([1, 2, 3, 4, 100])
    assert dist == pytest.approx(3.0)

def test_estimate_adaptive_falls_back_to_median_when_sparse():
    f = make_fuser(estimator='adaptive')
    # n=3 (<4) -> median
    dist, n = f.estimate_distance([5.0, 1.0, 3.0])
    assert n == 3
    assert dist == pytest.approx(3.0)

def test_estimate_adaptive_uses_q1_when_dense():
    f = make_fuser(estimator='adaptive')
    dist, n = f.estimate_distance([1, 2, 3, 4, 5, 6, 7, 8])
    assert dist == pytest.approx(3.0)  # same as q1

def test_estimate_unknown_estimator_falls_back_to_q1():
    f = make_fuser(estimator='nonsense')
    dist, _ = f.estimate_distance([1, 2, 3, 4, 5, 6, 7, 8])
    assert dist == pytest.approx(3.0)

def test_estimate_trimmed_mean_sane_bounds():
    f = make_fuser(estimator='trimmed_mean')
    vals = [1, 2, 3, 4, 5, 6, 7, 8]
    dist, _ = f.estimate_distance(vals)
    assert min(vals) <= dist <= max(vals)


# ── lidar_range_for_angle ──────────────────────────────────────────────────

def test_window_finds_return_straight_ahead():
    f = make_fuser(estimator='mean')
    idx = index_for_angle(0.0)
    ranges, amin, inc = make_scan(hits={idx - 1: 2.0, idx: 2.0, idx + 1: 2.0})
    dist, n = f.lidar_range_for_angle(ranges, amin, inc, 0.0, math.radians(5.0))
    assert n == 3
    assert dist == pytest.approx(2.0)

def test_window_excludes_out_of_band_returns():
    f = make_fuser(estimator='mean')
    idx0 = index_for_angle(0.0)
    idx_far = index_for_angle(math.radians(45.0))
    ranges, amin, inc = make_scan(hits={idx0: 2.0, idx_far: 1.0})
    dist, n = f.lidar_range_for_angle(ranges, amin, inc, 0.0, math.radians(5.0))
    assert n == 1
    assert dist == pytest.approx(2.0)

def test_window_filters_min_max_range_and_nonfinite():
    f = make_fuser(estimator='mean')  # min 0.2, max 6.0
    idx = index_for_angle(0.0)
    ranges, amin, inc = make_scan(hits={
        idx - 1: 0.05,   # below min -> excluded
        idx:     3.0,    # valid
        idx + 1: 7.0,    # above max -> excluded
        idx + 2: INF,    # nonfinite -> excluded
    })
    dist, n = f.lidar_range_for_angle(ranges, amin, inc, 0.0, math.radians(5.0))
    assert n == 1
    assert dist == pytest.approx(3.0)

def test_window_handles_wraparound_at_pi():
    f = make_fuser(estimator='mean')
    # returns just inside +pi and just inside -pi; window centered at pi
    idx_hi = index_for_angle(math.radians(179.0))
    idx_lo = index_for_angle(math.radians(-179.0))
    ranges, amin, inc = make_scan(hits={idx_hi: 2.5, idx_lo: 2.5})
    dist, n = f.lidar_range_for_angle(ranges, amin, inc, math.pi, math.radians(3.0))
    assert n == 2
    assert dist == pytest.approx(2.5)


# ── correct_bearing (parallax) ─────────────────────────────────────────────

def test_correct_bearing_noop_when_disabled():
    f = make_fuser(use_parallax=False)
    assert f.correct_bearing(0.3, 2.0) == 0.3

def test_correct_bearing_roundtrip_zero_offset():
    f = make_fuser(use_parallax=True, parallax_dx=0.0, parallax_dy=0.0)
    for th in (-0.5, 0.0, 0.2, 0.8):
        assert f.correct_bearing(th, 2.0) == pytest.approx(th)

def test_correct_bearing_lateral_offset_shifts_sign():
    # object straight ahead (theta=0), positive lateral camera offset
    f = make_fuser(use_parallax=True, parallax_dx=0.1, parallax_dy=0.0)
    out = f.correct_bearing(0.0, 1.0)
    assert out == pytest.approx(math.atan2(-0.1, 1.0))
    assert out < 0.0

def test_correct_bearing_nonpositive_range_noop():
    f = make_fuser(use_parallax=True, parallax_dx=0.1)
    assert f.correct_bearing(0.4, 0.0) == 0.4


# ── make_key ───────────────────────────────────────────────────────────────

def test_spatial_key_bins_by_cell():
    f = make_fuser(use_spatial_keys=True, spatial_bin_size=0.75)
    # 1.5 / 0.75 = 2.0 -> gx=2 ; 2.25/0.75 = 3 -> gy=3
    assert f.make_key('chair', 1.5, 2.25, 0) == 'chair_g2_3'

def test_spatial_key_same_cell_collides_diff_cell_separates():
    f = make_fuser(use_spatial_keys=True, spatial_bin_size=0.75)
    k_a = f.make_key('chair', 0.1, 0.1, 0)
    k_b = f.make_key('chair', 0.2, 0.2, 1)   # same cell
    k_c = f.make_key('chair', 3.0, 3.0, 2)   # different cell
    assert k_a == k_b
    assert k_a != k_c

def test_v2_key_fallback_uses_count_suffix():
    f = make_fuser(use_spatial_keys=False)
    assert f.make_key('chair', 9, 9, 0) == 'chair'
    assert f.make_key('chair', 9, 9, 1) == 'chair_1'


# ── fuse: map-frame placement ──────────────────────────────────────────────

def _one_detection(label='chair', center=0.0, span=0.0):
    return [{
        'label': label, 'friendly_label': label, 'confidence': 0.9,
        'center_angle_rad': center,
        'left_angle_rad': center + span / 2.0,
        'right_angle_rad': center - span / 2.0,
    }]

def test_fuse_no_pose_places_in_robot_frame():
    f = make_fuser(estimator='mean', use_spatial_keys=False)
    idx = index_for_angle(0.0)
    ranges, amin, inc = make_scan(hits={idx: 2.0})
    f.fuse(_one_detection(center=0.0), ranges, amin, inc,
           robot_x=0, robot_y=0, robot_yaw=0, has_pose=False)
    obj = f.registry['chair']
    assert obj['map_x'] == pytest.approx(2.0, abs=1e-2)
    assert obj['map_y'] == pytest.approx(0.0, abs=1e-2)

def test_fuse_with_pose_translates():
    f = make_fuser(estimator='mean', use_spatial_keys=False)
    idx = index_for_angle(0.0)
    ranges, amin, inc = make_scan(hits={idx: 2.0})
    f.fuse(_one_detection(center=0.0), ranges, amin, inc,
           robot_x=1.0, robot_y=2.0, robot_yaw=0.0, has_pose=True)
    obj = f.registry['chair']
    assert obj['map_x'] == pytest.approx(3.0, abs=1e-2)  # 1.0 + 2.0
    assert obj['map_y'] == pytest.approx(2.0, abs=1e-2)

def test_fuse_with_yaw_rotates():
    f = make_fuser(estimator='mean', use_spatial_keys=False)
    idx = index_for_angle(0.0)
    ranges, amin, inc = make_scan(hits={idx: 2.0})
    # robot at origin, yaw +90°: object straight ahead -> +Y in map
    f.fuse(_one_detection(center=0.0), ranges, amin, inc,
           robot_x=0.0, robot_y=0.0, robot_yaw=math.pi / 2, has_pose=True)
    obj = next(iter(f.registry.values()))
    assert obj['map_x'] == pytest.approx(0.0, abs=1e-2)
    assert obj['map_y'] == pytest.approx(2.0, abs=1e-2)

def test_fuse_ema_smooths_between_cycles():
    f = make_fuser(estimator='mean', use_spatial_keys=False, ema_alpha=0.5)
    idx = index_for_angle(0.0)
    r1, amin, inc = make_scan(hits={idx: 2.0})
    r2, _, _ = make_scan(hits={idx: 4.0})
    f.fuse(_one_detection(), r1, amin, inc, 0, 0, 0, False)
    assert f.registry['chair']['distance'] == pytest.approx(2.0, abs=1e-2)
    f.fuse(_one_detection(), r2, amin, inc, 0, 0, 0, False)
    # alpha 0.5: 0.5*4 + 0.5*2 = 3.0
    assert f.registry['chair']['distance'] == pytest.approx(3.0, abs=1e-2)


# ── compute_footprint ──────────────────────────────────────────────────────

def _extents(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (max(xs) - min(xs)), (max(ys) - min(ys))

def test_footprint_square_object_centered():
    f = make_fuser(footprint_width_refine=False)
    corners = f.compute_footprint('chair', 2.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    assert len(corners) == 4
    cx = sum(c[0] for c in corners) / 4
    cy = sum(c[1] for c in corners) / 4
    assert cx == pytest.approx(2.0)
    assert cy == pytest.approx(0.0)

def test_footprint_table_dims_used_when_no_refine():
    f = make_fuser(footprint_width_refine=False)
    # couch 1.80 x 0.85, object straight ahead (beta=0 -> no rotation)
    corners = f.compute_footprint('couch', 2.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    ex, ey = _extents(corners)  # ex=along-sight, ey=lateral
    # couch straight ahead: W(1.80) spans lateral (y), D(0.85) spans depth (x)
    assert ey == pytest.approx(1.80)
    assert ex == pytest.approx(0.85)

def test_footprint_unknown_label_uses_default():
    f = make_fuser(footprint_width_refine=False)
    corners = f.compute_footprint('unicorn', 2.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    ex, ey = _extents(corners)
    assert ex == pytest.approx(0.50)
    assert ey == pytest.approx(0.50)

def test_W_obs_formula_expands_width_when_arc_implies():
    f = make_fuser(footprint_width_refine=True, footprint_refine_tol=0.20)
    distance = 2.0
    span = math.radians(40.0)
    W_obs_expected = 2.0 * distance * math.tan(span / 2.0)  # ~1.456 m
    assert W_obs_expected > 0.55 * 1.20  # exceeds chair table+tol -> should apply
    corners = f.compute_footprint('chair', distance, span, distance, 0.0, 0.0, 0.0)
    ex, ey = _extents(corners)
    # the W_obs dimension should appear as one of the extents
    bigger = max(ex, ey)
    assert bigger == pytest.approx(W_obs_expected, abs=1e-3)

def test_W_obs_not_applied_when_within_tolerance():
    f = make_fuser(footprint_width_refine=True, footprint_refine_tol=0.20)
    distance = 2.0
    span = math.radians(5.0)  # tiny arc -> small W_obs, below chair+tol
    corners = f.compute_footprint('chair', distance, span, distance, 0.0, 0.0, 0.0)
    ex, ey = _extents(corners)
    assert max(ex, ey) == pytest.approx(0.55, abs=1e-3)  # table value retained


# ── footprint orientation (locks the fix) ──────────────────────────────────

def test_wobs_lands_on_lateral_axis():
    """
    W_obs is derived from the lateral angular span, so it represents extent
    PERPENDICULAR to the robot->object line of sight. For an object straight
    ahead (+x), the lateral axis is y. W_obs must appear on the y-extent;
    the table depth must remain on the x (along-sight) extent.
    """
    f = make_fuser(footprint_width_refine=True, footprint_refine_tol=0.20)
    distance = 2.0
    span = math.radians(40.0)
    W_obs = 2.0 * distance * math.tan(span / 2.0)
    corners = f.compute_footprint('chair', distance, span, distance, 0.0, 0.0, 0.0)
    ex, ey = _extents(corners)  # ex=along-sight, ey=lateral
    assert ey == pytest.approx(W_obs, abs=1e-3)   # lateral carries W_obs
    assert ex == pytest.approx(0.55, abs=1e-3)    # chair depth along sight