from sensor_msgs.msg import LaserScan

import atexit
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, CheckButtons, Slider
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN

SCAN_SCALAR_FIELDS = ('angle_min', 'angle_max', 'angle_increment', 'time_increment',
                      'scan_time', 'range_min', 'range_max')


def load_saved_scans(path):
    """Rebuild LaserScan messages from a file written by CenterlineExtractor.write_saved_scans."""
    data = np.load(path)
    scans = []
    for i in range(len(data['ranges'])):
        msg = LaserScan()
        msg.header.frame_id = str(data['frame_id'][i])
        msg.header.stamp.sec = int(data['stamp_sec'][i])
        msg.header.stamp.nanosec = int(data['stamp_nanosec'][i])
        for field in SCAN_SCALAR_FIELDS:
            setattr(msg, field, float(data[field][i]))
        msg.ranges = data['ranges'][i].tolist()
        scans.append(msg)
    return scans


class CenterlineExtractor:
    def __init__(self, max_track_width, scan_save_path=None):
        self.max_track_width = max_track_width
        self.max_range = 15.0 # beams farther than this are dropped (far wall points are too sparse to be useful)
        self.min_point_distance = 0.05 # raw points closer than this to the last kept one are dropped before resampling
        self.discretization_distance = 0.1 # distance between points in the resampled wall
        self.dbscan_eps = 0.4 # max distance between neighbouring points of the same wall
        self.dbscan_min_samples = 3 # points within eps needed to be a core point; fewer -> noise
        self.dbscan_noise = np.empty((0, 2))

        # scans saved via the "Save scan" button, written to disk on program stop/error
        self.current_scan = None
        self.last_saved_scan = None
        self.saved_scans = []
        self.scan_save_path = scan_save_path or os.path.join(
            'saved_scans', datetime.now().strftime('scans_%Y%m%d_%H%M%S.npz'))
        atexit.register(self.write_saved_scans)  # backup in case the caller doesn't flush

        # debug only
        self.has_initial_limits = False
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(right=0.74)
        self.layers = {
            'Scan': self.ax.scatter([], [], c='0.55', s=4, label='Scan'),
            'Left wall': self.ax.scatter([], [], c='tab:orange', s=12, label='Left wall'),
            'Right wall': self.ax.scatter([], [], c='tab:blue', s=12, label='Right wall'),
            'Pairs': LineCollection([], colors='tab:green', linewidths=1.5, label='Pairs'),
            'Extended pairs': LineCollection([], colors='tab:purple', linewidths=1.5, linestyles='--', label='Extended pairs'),
            'Reference line (smoothing spline)': self.ax.plot([], [], color='tab:olive', linewidth=2.0, linestyle=':', label='Reference line (smoothing spline)')[0],
            'Left resampled': self.ax.scatter([], [], c='tab:red', s=22, marker='x', label='Left resampled'),
            'Right resampled': self.ax.scatter([], [], c='tab:cyan', s=22, marker='x', label='Right resampled'),
            'Left wall (smoothed)': self.ax.plot([], [], color='tab:pink', linewidth=1.5, label='Left wall (smoothed)')[0],
            'Right wall (smoothed)': self.ax.plot([], [], color='tab:gray', linewidth=1.5, label='Right wall (smoothed)')[0],
            'DBSCAN noise': self.ax.scatter([], [], c='black', s=30, marker='x', label='DBSCAN noise'),
        }
        checkbox_ax = self.fig.add_axes([0.77, 0.43, 0.21, 0.22])
        self.checkboxes = CheckButtons(checkbox_ax, list(self.layers), [True] * len(self.layers))
        self.checkboxes.on_clicked(self.toggle_layer)
        checkbox_ax.set_title('Layers')
        button_ax = self.fig.add_axes([0.77, 0.33, 0.21, 0.06])
        self.save_button = Button(button_ax, 'Save scan (0)')
        self.save_button.on_clicked(self.save_current_scan)
        self.eps_slider = Slider(self.fig.add_axes([0.82, 0.25, 0.13, 0.03]), 'eps', 0.05, 1.5,
                                 valinit=self.dbscan_eps)
        self.min_samples_slider = Slider(self.fig.add_axes([0.82, 0.20, 0.13, 0.03]), 'min pts', 1, 20,
                                         valinit=self.dbscan_min_samples, valstep=1)
        self.eps_slider.on_changed(self.update_dbscan_params)
        self.min_samples_slider.on_changed(self.update_dbscan_params)
        self.ax.add_collection(self.layers['Pairs'])
        self.ax.add_collection(self.layers['Extended pairs'])
        self.ax.set_xlabel('Forward x (m)')
        self.ax.set_ylabel('Left y (m)')
        self.ax.set_title('LiDAR points')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid()
        plt.show(block=False)

    def extract_centerline(self, msg: LaserScan):
        self.current_scan = msg
        # Keep every successfully computed stage available for visualization.
        empty = np.empty((0, 2))
        self.dbscan_noise = empty
        coords = empty
        left_wall = right_wall = empty
        left_wall_resampled = right_wall_resampled = empty
        left_wall_smoothed = right_wall_smoothed = empty
        pairs, extended_pairs = [], []
        reference_line_smoothing_spline = None

        try:
            coords = self.lidar_to_cartesian(msg)
            if len(coords) >= 2:
                left_wall, right_wall = self.extract_walls(coords)
            elif len(coords):
                left_wall = coords

            left_wall_resampled = self.resample_walls(left_wall)
            right_wall_resampled = self.resample_walls(right_wall)
            left_wall_smoothed = self.get_smoothed_wall(left_wall_resampled)
            right_wall_smoothed = self.get_smoothed_wall(right_wall_resampled)

            # Same long/short selection, matching, and extension as before.
            if len(left_wall_resampled) < len(right_wall_resampled):
                long_wall, short_wall = right_wall_resampled, left_wall_resampled
                long_wall_smoothed = right_wall_smoothed
            else:
                long_wall, short_wall = left_wall_resampled, right_wall_resampled
                long_wall_smoothed = left_wall_smoothed

            pairs_by_long_index, unused = self.pair_walls(long_wall, short_wall)
            pairs = [pair for pair in pairs_by_long_index if pair is not None]
            extended_pairs = self.extend_pairs(pairs_by_long_index, unused, long_wall, long_wall_smoothed)
            total_pairs = [pair for pair in pairs_by_long_index if pair is not None]
            reference_line_smoothing_spline = self.get_reference_line_smoothing_spline(total_pairs)
        except (ValueError, IndexError, FloatingPointError, TypeError, np.linalg.LinAlgError) as exc:
            print(f"[centerline] Skipping unavailable stage: {exc}", flush=True)
        finally:
            # Display everything computed before any recoverable failure.
            self.plot(coords, left_wall, right_wall, left_wall_resampled,
                      right_wall_resampled, pairs, extended_pairs,
                      reference_line_smoothing_spline, left_wall_smoothed,
                      right_wall_smoothed)

    def get_midpoints_and_arclength(self, total_pairs):
        midpoints = np.array([(p[0] + p[1]) / 2 for p in total_pairs]).reshape(-1, 2)
        u = np.concatenate((
            [0],
            np.cumsum(np.linalg.norm(np.diff(midpoints, axis=0), axis=1)),
        ))
        return midpoints, u

    def get_reference_line_smoothing_spline(self, total_pairs, smoothing_factor=0.3):
        if len(total_pairs) < 4:
            return None
        midpoints, u = self.get_midpoints_and_arclength(total_pairs)
        if not np.all(np.isfinite(midpoints)) or not np.all(np.diff(u) > 0):
            return None
        spline_x = UnivariateSpline(u, midpoints[:, 0], s=smoothing_factor)
        spline_y = UnivariateSpline(u, midpoints[:, 1], s=smoothing_factor)
        return spline_x, spline_y, u

    def get_smoothed_wall(self, wall, window_length=11, polyorder=3):
        # keep window_length odd and within bounds so savgol_filter doesn't raise on short walls
        window_length = min(window_length, len(wall))
        if window_length % 2 == 0:
            window_length -= 1
        if window_length <= polyorder:
            return wall.copy()  # not enough points to fit this polyorder, skip smoothing
        return np.column_stack((
            savgol_filter(wall[:, 0], window_length, polyorder),
            savgol_filter(wall[:, 1], window_length, polyorder),
        ))

    def lidar_to_cartesian(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=float)
        # positive = left, negative = right, forward = 0
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment
        # reject points behind the car (lidar FOV can exceed +/-90 deg from forward),
        # too far away, or at/below range_min (invalid reading)
        keep = (np.abs(angles) <= np.pi / 2) & (ranges <= self.max_range) & (ranges > msg.range_min)
        ranges = ranges[keep]
        angles = angles[keep]
        cartesian = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        # LiDAR may contain inf/nan ranges. Keep valid readings in scan order.
        return cartesian[np.all(np.isfinite(cartesian), axis=1)]

    def extract_walls(self, coords) -> tuple[np.ndarray, np.ndarray]:
        labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(coords)
        self.dbscan_noise = coords[labels == -1]
        clustered = np.flatnonzero(labels != -1)
        if len(clustered) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        # coords are in scan order: first beams are at -90 deg (right), last at +90 deg (left)
        right_label = labels[clustered[0]]
        left_label = labels[clustered[-1]]
        if right_label == left_label:
            # walls touch (e.g. hairpin or far corner), fall back to cutting the cluster at its largest gap
            return self.split_at_largest_gap(coords[labels == left_label])
        # boolean masks keep scan order, which resampling relies on
        return coords[labels == left_label], coords[labels == right_label]

    def split_at_largest_gap(self, coords) -> tuple[np.ndarray, np.ndarray]:
        vertex_verctors = coords[1:] - coords[:-1]
        dst = np.linalg.norm(vertex_verctors, axis=1)
        if len(dst) == 0:
            return coords.copy(), np.empty((0, 2))
        max_gap_idx = np.argmax(dst)
        if dst[max_gap_idx] < self.max_track_width:
            left_wall = coords
            right_wall = np.empty((0, 2))
        else:
            right_wall = coords[:max_gap_idx + 1]
            left_wall = coords[max_gap_idx + 1:]
        return left_wall, right_wall

    def thin_wall(self, wall):
        # Dense near-range beams are a few mm apart, so LiDAR jitter would inflate the arc length.
        # Keeping only points >= min_point_distance from the last kept one makes the jitter negligible.
        if len(wall) < 2:
            return wall.copy()
        keep = np.zeros(len(wall), dtype=bool)
        keep[0] = True
        last = wall[0]
        for i in range(1, len(wall)):
            if np.hypot(*(wall[i] - last)) >= self.min_point_distance:
                keep[i] = True
                last = wall[i]
        return wall[keep]

    def resample_walls(self, wall):
        wall = self.thin_wall(wall)
        if len(wall) < 2:
            return wall.copy()
        cumulative_distances = np.cumsum(np.linalg.norm(np.diff(wall, axis=0), axis=1))
        cumulative_distances = np.insert(cumulative_distances, 0, 0.0)
        discretized_distances = np.arange(0, cumulative_distances[-1], self.discretization_distance)
        x_interp = np.interp(discretized_distances, cumulative_distances, wall[:, 0])
        y_interp = np.interp(discretized_distances, cumulative_distances, wall[:, 1])
        return np.column_stack((x_interp, y_interp))
    
    def pair_walls(self, long_wall, short_wall):
        pairs = [None] * len(long_wall)
        available = np.ones(len(long_wall), dtype=bool)

        for short_point in short_wall:
            candidate_indices = np.flatnonzero(available)

            # Cannot make unique pairs once all long points are used.
            if len(candidate_indices) == 0:
                break

            candidate_points = long_wall[candidate_indices]
            distances = np.linalg.norm(candidate_points - short_point, axis=1)

            best_local_index = np.argmin(distances)
            best_long_index = candidate_indices[best_local_index]
            
            pairs[best_long_index] = (
                long_wall[best_long_index],
                short_point,
            )

            available[best_long_index] = False  # this long point cannot be used again

        filtered_pairs = []
        if not any(pair is not None for pair in pairs):
            return pairs, available
        median_width = np.median([np.linalg.norm(pair[0] - pair[1]) for pair in pairs if pair is not None])
        filtered_pairs = [pair if pair is not None and np.linalg.norm(pair[0] - pair[1]) < 1.1 * median_width else None for pair in pairs]
        available = np.array([pair is None for pair in filtered_pairs], dtype=bool)
        return filtered_pairs, available

    def extend_pairs(self, pairs, unused, long_wall, tangent_wall):
        valid_pairs = [pair for pair in pairs if pair is not None]
        if not valid_pairs:
            return []
        track_width = np.mean(np.linalg.norm(np.array(valid_pairs)[:, 0] - np.array(valid_pairs)[:, 1], axis=1))
        extended_pairs = []
        for i in range(len(unused) - 1):  # skip the last point, cannot extend it
            if unused[i]:
                # tangent from tangent_wall (resampled or smoothed), anchored at the real (unsmoothed) point
                vector = tangent_wall[i + 1] - tangent_wall[i]
                normal = np.array([-vector[1], vector[0]])
                normal_length = np.linalg.norm(normal)
                if not np.isfinite(normal_length) or normal_length == 0:
                    continue
                unit_normal = normal / normal_length
                new_short_point = long_wall[i] + unit_normal * track_width
                pairs[i] = (
                    long_wall[i],
                    new_short_point,
                )
                extended_pairs.append(pairs[i])
        return extended_pairs

    def save_current_scan(self, event=None):
        msg = self.current_scan
        if msg is None:
            print('[centerline] No scan received yet, nothing to save', flush=True)
            return
        if msg is self.last_saved_scan:
            print('[centerline] Current scan already saved', flush=True)
            return
        scan = {field: getattr(msg, field) for field in SCAN_SCALAR_FIELDS}
        scan['ranges'] = np.asarray(msg.ranges, dtype=np.float32).copy()
        scan['frame_id'] = msg.header.frame_id
        scan['stamp_sec'] = msg.header.stamp.sec
        scan['stamp_nanosec'] = msg.header.stamp.nanosec
        self.saved_scans.append(scan)
        self.last_saved_scan = msg
        self.save_button.label.set_text(f'Save scan ({len(self.saved_scans)})')
        self.fig.canvas.draw_idle()
        print(f'[centerline] Saved scan #{len(self.saved_scans)} (written on exit)', flush=True)

    def write_saved_scans(self):
        if not self.saved_scans:
            return
        os.makedirs(os.path.dirname(self.scan_save_path) or '.', exist_ok=True)
        columns = {key: np.array([scan[key] for scan in self.saved_scans])
                   for key in self.saved_scans[0] if key != 'ranges'}
        np.savez(self.scan_save_path, ranges=np.stack([scan['ranges'] for scan in self.saved_scans]), **columns)
        print(f'[centerline] Wrote {len(self.saved_scans)} scans to {os.path.abspath(self.scan_save_path)}', flush=True)
        self.saved_scans = []  # so the atexit backup doesn't write twice

# DEBUG ONLY
    def update_dbscan_params(self, _value):
        self.dbscan_eps = float(self.eps_slider.val)
        self.dbscan_min_samples = int(self.min_samples_slider.val)
        if self.current_scan is not None:
            self.extract_centerline(self.current_scan)  # re-run on the shown scan so the effect is visible right away

    def toggle_layer(self, label):
        artist = self.layers[label]
        artist.set_visible(not artist.get_visible())
        self.fig.canvas.draw_idle()

    def plot(self, coords, left_wall, right_wall, left_wall_resampled, right_wall_resampled, pairs, extended_pairs,
             reference_line_smoothing_spline, left_wall_smoothed, right_wall_smoothed):
        groups = {
            'Scan': coords,
            'Left wall': left_wall,
            'Right wall': right_wall,
            'Left resampled': left_wall_resampled,
            'Right resampled': right_wall_resampled,
            'DBSCAN noise': self.dbscan_noise,
        }
        for name, points in groups.items():
            self.layers[name].set_offsets(np.asarray(points).reshape(-1, 2))
        self.layers['Pairs'].set_segments(pairs)
        self.layers['Extended pairs'].set_segments(extended_pairs)

        if reference_line_smoothing_spline is not None:
            spline_x, spline_y, u = reference_line_smoothing_spline
            u_dense = np.linspace(u[0], u[-1], 200)
            self.layers['Reference line (smoothing spline)'].set_data(spline_x(u_dense), spline_y(u_dense))
        else:
            self.layers['Reference line (smoothing spline)'].set_data([], [])

        self.layers['Left wall (smoothed)'].set_data(left_wall_smoothed[:, 0], left_wall_smoothed[:, 1])
        self.layers['Right wall (smoothed)'].set_data(right_wall_smoothed[:, 0], right_wall_smoothed[:, 1])
        # Scatter collections do not participate in Axes.relim(), so set limits
        # directly from the complete scan.
        if len(coords) and not self.has_initial_limits:
            lower = coords.min(axis=0)
            upper = coords.max(axis=0)
            padding = np.maximum((upper - lower) * 0.05, 0.1)
            self.ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            self.ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            self.has_initial_limits = True

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Process XQuartz/Tk GUI events during ROS callbacks.
