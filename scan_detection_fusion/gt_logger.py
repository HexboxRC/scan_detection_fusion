#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json, csv, time, sys

class GTLogger(Node):
    def __init__(self, session):
        super().__init__('gt_logger')
        self.sub = self.create_subscription(
            String, '/detected_objects', self.cb, 10)

        fname = f'exp_{session}_{int(time.time())}.csv'
        self.f = open(fname, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            'wall_time', 'label', 'confidence',
            'fused_dist_m', 'fused_x', 'fused_y',
            'angle_deg', 'n_lidar_pts',
            'pose_source', 'robot_x', 'robot_y',
            'gt_dist_m', 'notes'
        ])
        self.row_count = 0
        self.get_logger().info(f'Logging to {fname}')

    def cb(self, msg):
        try:
            data = json.loads(msg.data)
            objs = data.get('objects', [])           # ← fuser uses 'objects' key
            pose_source = data.get('pose_source', 'unknown')
            robot_pose = data.get('robot_pose', {})
            robot_x = robot_pose.get('x', 0.0)
            robot_y = robot_pose.get('y', 0.0)

            for o in objs:
                self.writer.writerow([
                    time.time(),
                    o.get('label', ''),
                    o.get('confidence', -1),
                    o.get('distance', -1),            # ← fuser uses 'distance'
                    o.get('map_x', -1),               # ← fuser uses 'map_x'
                    o.get('map_y', -1),               # ← fuser uses 'map_y'
                    o.get('angle_deg', -1),
                    o.get('n_points', -1),
                    pose_source,
                    robot_x,
                    robot_y,
                    '',   # fill in gt_dist_m manually after run
                    ''
                ])
                self.row_count += 1
            self.f.flush()

            # Heartbeat: print every 10 rows so you can see progress
            if self.row_count > 0 and self.row_count % 10 == 0:
                self.get_logger().info(f'Logged {self.row_count} rows so far')

        except Exception as e:
            self.get_logger().warn(f'Parse error: {e}')

def main():
    rclpy.init()
    session = sys.argv[1] if len(sys.argv) > 1 else 'default'
    node = GTLogger(session)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Final row count: {node.row_count}')
        node.f.close()

if __name__ == '__main__':
    main()