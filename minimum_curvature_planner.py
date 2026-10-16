import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from centerline_extraction import CenterlineExtractor
import time

class MinimumCurvaturePlanner(Node):
    def __init__(self):        
        super().__init__('minimum_curvature_planner')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos
        )

        self.centerline_extractor = CenterlineExtractor(max_track_width=2.0)

        self.last_req = time.time()

    def scan_callback(self, msg: LaserScan):
        if time.time() - self.last_req < 1:  # limit to 1 Hz
            return
        self.last_req = time.time()
        self.centerline_extractor.extract_centerline(msg)

def main():
    rclpy.init()
    node = MinimumCurvaturePlanner()
    try:
        rclpy.spin(node)       # stays alive and receives /scan callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        import matplotlib.pyplot as plt
        plt.close('all')       # close the GUI window


if __name__ == '__main__':
    main()