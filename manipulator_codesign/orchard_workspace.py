import open3d as o3d
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def extract_prune_points(yaml_path):
    """
    Extracts prune points from a YAML file.

    Parameters:
        yaml_path (str): The path to the YAML file containing prune points.

    Returns:
        list: A list of prune points, each represented as a dictionary with 'x', 'y', and 'z' keys.
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Initialize empty lists
    prune_points = []
    base_directions = []
    base_points = []

    # Extract the arrays
    for entry in data.values():
        prune_points.append(entry["prune_point"])
        base_directions.append(entry["pruned_branch_base_direction"])
        base_points.append(entry["pruned_branch_base_point"])
    
    # Convert to NumPy arrays
    prune_points = np.array(prune_points)
    base_directions = np.array(base_directions)
    base_points = np.array(base_points)

    return prune_points, base_directions, base_points

def filter_prune_points(
    prune_points,
    base_directions,
    base_points,
    window_x_pos,
    window_size=0.5,
    min_y=None,
    max_y=None,
    downsample=False,
    downsample_threshold=0.1,
    min_samples=1
):
    """
    Filters prune points based on their x-position, a specified window size, optional y-bounds,
    and (optionally) downsamples using DBSCAN clustering.

    Parameters:
        prune_points (np.ndarray): Array of prune points with shape (N, 3).
        base_directions (np.ndarray): Array of base directions with shape (N, 3).
        base_points (np.ndarray): Array of base points with shape (N, 3).
        window_x_pos (float): The x-position to filter around.
        window_size (float): The size of the window to use for filtering.
        min_y (float, optional): Minimum y-value to include. Defaults to None.
        max_y (float, optional): Maximum y-value to include. Defaults to None.
        downsample (bool): Whether to apply DBSCAN clustering to reduce close points.
        downsample_threshold (float): Distance threshold (DBSCAN `eps`) for clustering.
        min_samples (int): DBSCAN `min_samples` parameter.

    Returns:
        tuple: Filtered (and optionally downsampled) arrays of prune points, base directions, and base points.
    """
    # Step 1: Apply window filtering
    x_min, x_max = get_filtered_window_bounds(window_x_pos, window_size)
    mask = (prune_points[:, 0] >= x_min) & (prune_points[:, 0] <= x_max)
    if min_y is not None:
        mask &= (prune_points[:, 1] >= min_y)
    if max_y is not None:
        mask &= (prune_points[:, 1] <= max_y)

    filtered_points = prune_points[mask]
    filtered_dirs = base_directions[mask]
    filtered_bases = base_points[mask]

    if filtered_points.shape[0] == 0:
        return filtered_points, filtered_dirs, filtered_bases

    # Step 2: Optionally downsample via DBSCAN
    if downsample:
        clustering = DBSCAN(eps=downsample_threshold, min_samples=min_samples).fit(filtered_points[:, :2])  # x,y only
        labels = clustering.labels_

        unique_labels = np.unique(labels)
        kept_points, kept_dirs, kept_bases = [], [], []

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = filtered_points[cluster_mask]
            cluster_dirs = filtered_dirs[cluster_mask]
            cluster_bases = filtered_bases[cluster_mask]

            if label == -1:
                # Noise points: keep them all
                kept_points.append(cluster_points)
                kept_dirs.append(cluster_dirs)
                kept_bases.append(cluster_bases)
            else:
                # For clusters: keep just one representative (here, the first point)
                kept_points.append(cluster_points[:1])
                kept_dirs.append(cluster_dirs[:1])
                kept_bases.append(cluster_bases[:1])

        filtered_points = np.vstack(kept_points) if kept_points else np.empty((0, 3))
        filtered_dirs = np.vstack(kept_dirs) if kept_dirs else np.empty((0, 3))
        filtered_bases = np.vstack(kept_bases) if kept_bases else np.empty((0, 3))

    return filtered_points, filtered_dirs, filtered_bases

def prune_pose(point, direction, base_point, robot_base,
               num_samples=36, radius=0.1):
    """
    For each angle θ around the branch axis, consider BOTH offset directions
    (+ and – in the perp-plane), and pick the one whose OFFSET POINT is
    closest to robot_base. Then build the frame so that:
      • x-axis = that exact perp direction 
      • z-axis points from prune-point back to robot_base
      • y-axis completes right-hand rule

    Returns:
      p             : the original prune point (3,)
      quat          : orientation as [x, y, z, w]
      best_offset_pt: the offset point associated with the best direction (3,)
    """

    p = np.asarray(point, dtype=float)
    v = np.asarray(direction, dtype=float)
    r = np.asarray(robot_base, dtype=float)

    # 1) Normalize branch vector → axis z0
    z0 = v / np.linalg.norm(v)

    # 2) Build any perp basis (u, w) to z0
    arb = np.array([1,0,0]) if abs(z0[0])<0.9 else np.array([0,1,0])
    u = np.cross(z0, arb)
    u /= np.linalg.norm(u)
    w = np.cross(z0, u)

    best = {
        'dist': np.inf,
        'x_dir': None,
        'offset_pt': None
    }

    # 3) For each sample angle, test both signs
    for k in range(num_samples):
        θ = 2 * np.pi * k / num_samples
        x_cand = np.cos(θ) * u + np.sin(θ) * w
        for sign in (+1, -1):
            x_dir = sign * x_cand
            offset_pt = p + radius * x_dir
            d = np.linalg.norm(offset_pt - r)
            if d < best['dist']:
                best['dist'] = d
                best['x_dir'] = x_dir.copy()
                best['offset_pt'] = offset_pt.copy()

    if best['x_dir'] is None:
        raise RuntimeError("No candidate orientations generated")

    # 4) Build the final frame from the winning x_dir
    x_axis = best['x_dir']
    # z-axis points from prune point back to robot base
    z_axis = -(r - p)
    z_axis /= np.linalg.norm(z_axis)
    # y-axis completes right-hand rule
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # 5) Pack into quaternion [x, y, z, w]
    Rm = np.column_stack((x_axis, y_axis, z_axis))
    quat = R.from_matrix(Rm).as_quat()

    return p, quat, best['offset_pt']

def get_prune_single_pose_from_yaml(yaml_path, robot_base, window_size=0.5, min_y=None, max_y=None):
    """
    Extracts prune poses from a YAML file and computes the best pose for each prune point.

    Parameters:
        yaml_path (str): The path to the YAML file containing prune points.
        robot_base (np.ndarray): The base position of the robot as a 3D vector.
        window_size (float): The size of the window to use for filtering.

    Returns:
        list: A list of tuples containing the prune point and its corresponding quaternion.
    """
    # Check to see if yaml path is valid
    if not yaml_path or not isinstance(yaml_path, str):
        raise ValueError("Invalid YAML path provided.")
    
    prune_points, base_directions, base_points = extract_prune_points(yaml_path)
    
    # Filter prune points based on the x-position
    filtered_points, filtered_directions, filtered_bases = filter_prune_points(
        prune_points, base_directions, base_points, robot_base[0], window_size, min_y=min_y, max_y=max_y
    )

    prune_points = []
    prune_orientations = []
    offset_approach_points = []
    for pt, dv, bp in zip(filtered_points, filtered_directions, filtered_bases):
        prune_point, prune_orientation, prune_approach_point = prune_pose(pt, dv, bp, robot_base)
        prune_points.append(prune_point)
        prune_orientations.append(prune_orientation)
        offset_approach_points.append(prune_approach_point)

    # Package prune points and orientations into a list of (point, orientation) tuples
    poses = list(zip(prune_points, prune_orientations))
    offset_poses = list(zip(offset_approach_points, prune_orientations))

    return poses, offset_poses

def prune_pose_candidates(point, direction, base_point, robot_base,
                          num_samples=36, radius=0.1):
    """
    Sample around the branch axis and return *all* candidate
    (offset_pt, quaternion) on the side of the ring closest to robot_base.

    Returns:
      p                : original prune point, shape (3,)
      offset_pts       : list of offset points (M,3)
      quaternions      : list of corresponding [x,y,z,w] quaternions (M,4)
    """
    p = np.asarray(point, dtype=float)
    v = np.asarray(direction, dtype=float)
    r = np.asarray(robot_base, dtype=float)

    # 1) Axis of branch
    z0 = v / np.linalg.norm(v)

    # 2) Build perp basis (u,w)
    arb = np.array([1,0,0]) if abs(z0[0])<0.9 else np.array([0,1,0])
    u = np.cross(z0, arb);  u /= np.linalg.norm(u)
    w = np.cross(z0, u)

    # Vector from prune point to robot
    to_robot = (r - p) / np.linalg.norm(r - p)

    offset_pts  = []
    quaternions = []

    # 3) Sample ring
    for k in range(num_samples):
        θ = 2*np.pi * k / num_samples
        ring_dir = np.cos(θ)*u + np.sin(θ)*w
        for sign in (+1, -1):
            x_dir = sign * ring_dir
            # only keep if pointing toward robot
            if np.dot(x_dir, to_robot) <= 0:
                continue

            offset_pt = p + radius * x_dir

            # build z-axis (back to robot), y-axis (RH rule)
            z_axis = - (r - p)
            z_axis /= np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_dir)
            y_axis /= np.linalg.norm(y_axis)

            Rm = np.column_stack((x_dir, y_axis, z_axis))
            quat = R.from_matrix(Rm).as_quat()  # [x,y,z,w]

            offset_pts.append(offset_pt)
            quaternions.append(quat)

    if not offset_pts:
        raise RuntimeError("No near‑side candidates found.")

    return p, np.array(quaternions), np.array(offset_pts)

def get_prune_poses_from_yaml(yaml_path, robot_base, window_size=0.5, min_y=None, max_y=None, downsample=False, downsample_threshold=0.1):
    # Check to see if yaml path is valid
    if not yaml_path or not isinstance(yaml_path, str):
        raise ValueError("Invalid YAML path provided.")
    
    prune_points, base_dirs, base_pts = extract_prune_points(yaml_path)
    filt_pts, filt_dirs, filt_bases = filter_prune_points(
        prune_points, base_dirs, base_pts, robot_base[0],
        window_size, min_y=min_y, max_y=max_y,
        downsample=downsample, downsample_threshold=downsample_threshold
    )

    all_results = []
    for pt, dv, bp in zip(filt_pts, filt_dirs, filt_bases):
        p, quats, offsets = prune_pose_candidates(pt, dv, bp, robot_base, num_samples=8)
        # Combine prune point and offset points with orientations
        # prune_poses: (prune_point, orientation) for each orientation
        prune_poses = [(p, quat) for quat in quats]
        # offset_poses: (offset_point, orientation) for each offset/orientation
        offset_poses = list(zip(offsets, quats))
        all_results.append({
            'prune_point': p,
            'offset_points': offsets,        # shape (M,3)
            'orientations': quats,           # shape (M,4)
            'prune_poses': prune_poses,      # list of (prune_point, orientation)
            'offset_poses': offset_poses     # list of (offset_point, orientation)
        })

    return all_results

def package_poses(prune_data):
    N = len(prune_data)
    M = len(prune_data[0]['prune_poses'])   # number of candidates per prune

    # create an empty (N, M, 2) array of Python objects
    target_poses = np.empty((N, M, 2), dtype=object)
    target_offset_poses = np.empty((N, M, 2), dtype=object)

    for i, entry in enumerate(prune_data):
        prune_list = entry['prune_poses']          # [(p, quat), (p, quat), ...]
        for j, (p, quat) in enumerate(prune_list):
            target_poses[i, j, 0] = p              # the 3‑vector
            target_poses[i, j, 1] = quat           # the 4‑vector
    
    for i, entry in enumerate(prune_data):
        prune_list = entry['offset_poses']          # [(p, quat), (p, quat), ...]
        for j, (p, quat) in enumerate(prune_list):
            target_offset_poses[i, j, 0] = p              # the 3‑vector
            target_offset_poses[i, j, 1] = quat           # the 4‑vector
    return target_poses, target_offset_poses

def load_point_cloud(file_path):
    """
    Loads a point cloud from a file.

    Parameters:
        file_path (str): The path to the point cloud file.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.has_points():
        raise ValueError(f"Point cloud at {file_path} is empty or invalid.")
    return pcd

def get_filtered_window_bounds(window_x_pos, window_size=0.5):
    """
    Computes the bounds of a point cloud for filtering based on a specified window size.

    Parameters:
        window_x_pos (float): Starting x-position of the window.
        window_size (float): The size of the window to use for filtering.

    Returns:
        tuple: A tuple containing the minimum and maximum bounds (x_min, x_max).
    """
    
    x_min = np.min(window_x_pos) - window_size / 2
    x_max = np.max(window_x_pos) + window_size / 2
    
    return x_min, x_max

def filter_by_axis(pcd,
                x_min=None, x_max=None,
                y_min=None, y_max=None,
                z_min=None, z_max=None
                ):
    """
    Filters a point cloud by specified axis-aligned bounding box limits.

    Parameters:
        pcd (o3d.geometry.PointCloud): The input point cloud to filter.
        x_min (float, optional): Minimum x-value to include. Points with x < x_min are excluded.
        x_max (float, optional): Maximum x-value to include. Points with x > x_max are excluded.
        y_min (float, optional): Minimum y-value to include. Points with y < y_min are excluded.
        y_max (float, optional): Maximum y-value to include. Points with y > y_max are excluded.
        z_min (float, optional): Minimum z-value to include. Points with z < z_min are excluded.
        z_max (float, optional): Maximum z-value to include. Points with z > z_max are excluded.

    Returns:
        o3d.geometry.PointCloud: A new point cloud containing only the points within the specified bounds.
    """
    points = np.asarray(pcd.points)
    mask = np.ones(points.shape[0], dtype=bool)
    if x_min is not None:
        mask &= points[:, 0] >= x_min
    if x_max is not None:
        mask &= points[:, 0] <= x_max
    if y_min is not None:
        mask &= points[:, 1] >= y_min
    if y_max is not None:
        mask &= points[:, 1] <= y_max
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return filtered_pcd

def downsample_point_cloud(pcd, voxel_size=0.01):
    """
    Downsamples a point cloud using voxel downsampling.

    Parameters:
        pcd (o3d.geometry.PointCloud): The input point cloud to downsample.
        voxel_size (float): The size of the voxel grid for downsampling.

    Returns:
        o3d.geometry.PointCloud: The downsampled point cloud.
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)

def viz_prune_pose_candidates(prune_points, base_directions, base_points,
                              prune_fn, robot_base, idx=0,
                              num_samples=36, radius=0.1,
                              orient_scale=0.05):
    """
    Visualize all robot‑facing candidate offsets and their x‑axes.
    
    Handles both full quaternions (4,) and rotation vectors (3,).
    """
    pt  = prune_points[idx]
    dv  = base_directions[idx]
    bp  = base_points[idx]
    r   = np.asarray(robot_base, float)

    # Get all candidate offsets & quats
    p, quats, offsets = prune_fn(pt, dv, bp, r,
                                  num_samples=num_samples,
                                  radius=radius)
    # --- setup ---
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Prune #{idx}: {len(offsets)} candidates (radius={radius})")

    # plot prune point & robot base
    ax.scatter(*pt, color='k', s=50, label='prune point')
    ax.scatter(*r,  color='g', s=50, label='robot base')

    # plot branch direction at p
    dv_n = dv / np.linalg.norm(dv)
    ax.quiver(*pt, *(dv_n * orient_scale*2),
              color='r', linewidth=2,
              arrow_length_ratio=0.2, label='branch dir')

    # plot all offset points
    ax.scatter(offsets[:,0], offsets[:,1], offsets[:,2],
               color='0.5', s=20, alpha=0.8, label='offset pts')

    # at each offset, draw its local x‑axis
    for off, quat in zip(offsets, quats):
        # detect quat vs rotvec
        quat = np.asarray(quat)
        if quat.size == 3:
            Rm = R.from_rotvec(quat).as_matrix()
        elif quat.size == 4:
            Rm = R.from_quat(quat).as_matrix()
        else:
            raise ValueError(f"Invalid rotation data of size {quat.size}")
        x_axis = Rm[:,0]
        ax.quiver(off[0], off[1], off[2],
              x_axis[0] * orient_scale,
              x_axis[1] * orient_scale,
              x_axis[2] * orient_scale,
              color='b', linewidth=1, arrow_length_ratio=0.3)

    # equalize aspect
    all_pts = np.vstack([pt, r, offsets])
    mins = all_pts.min(axis=0) - radius
    maxs = all_pts.max(axis=0) + radius
    max_range = (maxs - mins).max()
    centers = (maxs + mins) / 2
    ax.set_xlim(centers[0] - max_range/2, centers[0] + max_range/2)
    ax.set_ylim(centers[1] - max_range/2, centers[1] + max_range/2)
    ax.set_zlim(centers[2] - max_range/2, centers[2] + max_range/2)

    ax.legend()
    plt.show()

def publish_point_cloud_collisions(pcd, client, radius=0.005, mass=0, collision_visual=False, rgba_color=[1,0,0,1]):
    """
    Publishes each point in the point cloud as a small spherical collision object in PyBullet.

    Args:
        pcd (o3d.geometry.PointCloud): downsampled point cloud
        client: PyBullet physics client (returned from p.connect)
        radius (float): radius of each sphere collision shape
        mass (float): mass of each collision object (0 = static)
        collision_visual (bool): whether to create a visual shape alongside collision
        rgba_color (list of 4): RGBA color for visual spheres

    Returns:
        list of int: body IDs of the created collision objects
    """
    # extract points
    points = np.asarray(pcd.points)
    body_ids = []
    for idx, pt in enumerate(points):
        # create a spherical collision shape
        col_shape = client.createCollisionShape(
            client.GEOM_SPHERE,
            radius=radius
        )
        vis_shape = -1
        if collision_visual:
            vis_shape = client.createVisualShape(
                client.GEOM_SPHERE,
                radius=radius,
                rgbaColor=rgba_color
            )
        # spawn the multi-body
        body_id = client.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=pt.tolist(),
            useMaximalCoordinates=True
        )
        body_ids.append(body_id)
    return body_ids


if __name__ == "__main__":
    # # Example usage
    # pcd = load_point_cloud("/home/marcus/IMML/manipulator_codesign/point_clouds/before_pcd_transformed.ply")
    # bounds = get_filtered_window_bounds(3, window_size=3.0)
    # filtered_pcd = filter_by_axis(pcd, *bounds)
    # print("Number of points after filtering:", len(filtered_pcd.points))
    # downsampled_pcd = downsample_point_cloud(filtered_pcd, voxel_size=0.15)
    # print("Number of points after downsampling:", len(downsampled_pcd.points))

    # o3d.visualization.draw_geometries([downsampled_pcd])

    yaml_path = "/home/marcus/IMML/manipulator_codesign/prune_data/all_branches_info.yaml"
    prune_points, base_directions, base_points = extract_prune_points(yaml_path)

    # define your robot base (replace or query dynamically)
    robot_base = np.array([-1.0, -0.5, 1.025])

    viz_prune_pose_candidates(
        prune_points,
        base_directions,
        base_points,
        prune_fn   = prune_pose_candidates,
        robot_base = robot_base,
        idx        = 0,        # which prune point to show
        num_samples=36,        # same sampling you used in prune_pose_candidates
        radius     = 1.0,      # same offset radius
        orient_scale = 0.5    # size of the little x‐axis arrows
    )