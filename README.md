Some of my work on algorithms for path planning, controls, estimation, and perception for F1Tenth vehicle.

Current pipeline:
LiDAR Scan -> Dropping points that are invalid / behind / too far -> DBSCAN Clustering -> Splitting scan into two walls -> Thinning dense points -> Resampling -> Savitsky-Golay Filtering -> Pairing points on walls -> Predicting pairs for points on the wall hidden behind the turn -> Joining midpoints with a smoothing spline

Todos: make use of odom and utilize previously processed path/points for new scans
