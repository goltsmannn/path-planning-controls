from sensor_msgs.msg import LaserScan

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import CheckButtons
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

class CenterlineExtractor:
    def __init__(self, max_track_width):
        self.max_track_width = max_track_width
        self.discretization_distance = 1.0 # distance between points in the resampled wall

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
        }
        checkbox_ax = self.fig.add_axes([0.77, 0.43, 0.21, 0.22])
        self.checkboxes = CheckButtons(checkbox_ax, list(self.layers), [True] * len(self.layers))
        self.checkboxes.on_clicked(self.toggle_layer)
        checkbox_ax.set_title('Layers')
        self.ax.add_collection(self.layers['Pairs'])
        self.ax.add_collection(self.layers['Extended pairs'])
        self.ax.set_xlabel('Forward x (m)')
        self.ax.set_ylabel('Left y (m)')
        self.ax.set_title('LiDAR points')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.grid()
        plt.show(block=False)

    def extract_centerline(self, msg: LaserScan):
        # convert lidar to cartesian
        coords = self.lidar_to_cartesian(msg)
        # separate wall points
        left_wall, right_wall = self.extract_walls(coords)
        #resample walls at discretization_distance to get equidistant points along the walls
        left_wall_resampled = self.resample_walls(left_wall)
        right_wall_resampled = self.resample_walls(right_wall)
        left_wall_smoothed = self.get_smoothed_wall(left_wall_resampled)
        right_wall_smoothed = self.get_smoothed_wall(right_wall_resampled)

        # use long/short instead of left/right
        if len(left_wall_resampled) < len(right_wall_resampled):
            long_wall, short_wall = right_wall_resampled, left_wall_resampled
            long_wall_smoothed = right_wall_smoothed
        else:
            long_wall, short_wall = left_wall_resampled, right_wall_resampled
            long_wall_smoothed = left_wall_smoothed

        # pair matching wall parts
        pairs_by_long_index, unused = self.pair_walls(long_wall, short_wall)
        pairs = [pair for pair in pairs_by_long_index if pair is not None] # debug only
        # use normal and tangents (from the smoothed wall) to pair unused points
        extended_pairs = self.extend_pairs(pairs_by_long_index, unused, long_wall, long_wall_smoothed)
        total_pairs = [pair for pair in pairs_by_long_index if pair is not None]
        # fit a reference line through the midpoints
        reference_line_smoothing_spline = self.get_reference_line_smoothing_spline(total_pairs)
        self.plot(coords, left_wall, right_wall, left_wall_resampled, right_wall_resampled, pairs, extended_pairs,
                  reference_line_smoothing_spline, left_wall_smoothed, right_wall_smoothed)

    def get_midpoints_and_arclength(self, total_pairs):
        midpoints = np.array([(p[0] + p[1]) / 2 for p in total_pairs])
        u = np.concatenate((
            [0],
            np.cumsum(np.linalg.norm(np.diff(midpoints, axis=0), axis=1)),
        ))
        return midpoints, u

    def get_reference_line_smoothing_spline(self, total_pairs, smoothing_factor=0.3):
        midpoints, u = self.get_midpoints_and_arclength(total_pairs)
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
        # reject points behind the car (lidar FOV can exceed +/-90 deg from forward)
        forward_facing = np.abs(angles) <= np.pi / 2
        ranges = ranges[forward_facing]
        angles = angles[forward_facing]
        cartesian = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        return cartesian

    def extract_walls(self, coords) -> tuple[np.ndarray, np.ndarray]:
        vertex_verctors = coords[1:] - coords[:-1]
        dst = np.linalg.norm(vertex_verctors, axis=1)
        max_gap_idx = np.argmax(dst)
        if dst[max_gap_idx] < self.max_track_width:
            left_wall = coords
            right_wall = np.empty((0, 2))
        else:
            right_wall = coords[:max_gap_idx + 1]
            left_wall = coords[max_gap_idx + 1:]
        return left_wall, right_wall

    def resample_walls(self, wall):
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
                unit_normal = normal / np.linalg.norm(normal)
                new_short_point = long_wall[i] + unit_normal * track_width
                pairs[i] = (
                    long_wall[i],
                    new_short_point,
                )
                extended_pairs.append(pairs[i])
        return extended_pairs

# DEBUG ONLY
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
        }
        for name, points in groups.items():
            self.layers[name].set_offsets(points)
        self.layers['Pairs'].set_segments(pairs)
        self.layers['Extended pairs'].set_segments(extended_pairs)

        spline_x, spline_y, u = reference_line_smoothing_spline
        u_dense = np.linspace(u[0], u[-1], 200)
        self.layers['Reference line (smoothing spline)'].set_data(spline_x(u_dense), spline_y(u_dense))

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
