import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.utilities import remove_ros_args
from centerline_extraction import CenterlineExtractor, load_saved_scans
import os
import sys
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

def replay(path):
    """Run the extractor on scans saved via the plot button; left/right arrows switch scans."""
    import matplotlib.pyplot as plt
    scans = load_saved_scans(path)
    if not scans:
        print(f'[replay] No scans in {path}', flush=True)
        return
    # matplotlib binds left/right to view back/forward by default
    for keymap in ('keymap.back', 'keymap.forward'):
        plt.rcParams[keymap] = [k for k in plt.rcParams[keymap] if k not in ('left', 'right')]

    extractor = CenterlineExtractor(max_track_width=2.0)
    index = 0

    def show():
        extractor.extract_centerline(scans[index])
        extractor.ax.set_title(f'{os.path.basename(path)}: scan {index + 1}/{len(scans)}  (←/→ to switch)')
        extractor.fig.canvas.draw_idle()

    def on_key(event):
        nonlocal index
        step = {'right': 1, 'left': -1}.get(event.key)
        if step is None or len(scans) < 2:
            return
        index = (index + step) % len(scans)
        show()

    extractor.fig.canvas.mpl_connect('key_press_event', on_key)
    show()
    plt.show(block=True)


def main():
    # remove_ros_args so --ros-args (added by ros2 run/launch) isn't mistaken for a file
    args = remove_ros_args(sys.argv)[1:]
    if args:
        replay(args[0])
        return

    rclpy.init()
    node = MinimumCurvaturePlanner()
    try:
        rclpy.spin(node)       # stays alive and receives /scan callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.centerline_extractor.write_saved_scans()  # flush scans saved via the plot button
        node.destroy_node()
        rclpy.shutdown()
        import matplotlib.pyplot as plt
        plt.close('all')       # close the GUI window


if __name__ == '__main__':
    main()