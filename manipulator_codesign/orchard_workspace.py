import open3d as o3d
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

def filter_prune_points(prune_points, base_directions, base_points, window_x_pos, window_size=0.5, min_y=None, max_y=None):
    """
    Filters prune points based on their x-position, a specified window size, and optional y-bounds.

    Parameters:
        prune_points (np.ndarray): Array of prune points with shape (N, 3).
        base_directions (np.ndarray): Array of base directions with shape (N, 3).
        base_points (np.ndarray): Array of base points with shape (N, 3).
        window_x_pos (float): The x-position to filter around.
        window_size (float): The size of the window to use for filtering.
        min_y (float, optional): Minimum y-value to include. Defaults to None.
        max_y (float, optional): Maximum y-value to include. Defaults to None.

    Returns:
        tuple: Filtered arrays of prune points, base directions, and base points.
    """
    x_min, x_max = get_filtered_window_bounds(window_x_pos, window_size)
    mask = (prune_points[:, 0] >= x_min) & (prune_points[:, 0] <= x_max)
    if min_y is not None:
        mask &= (prune_points[:, 1] >= min_y)
    if max_y is not None:
        mask &= (prune_points[:, 1] <= max_y)
    return prune_points[mask], base_directions[mask], base_points[mask]

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

def get_prune_poses_from_yaml(yaml_path, robot_base, window_size=0.5, min_y=None, max_y=None):
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
                              num_samples=36, radius=0.1, scale=1.0):
    pt = prune_points[idx]
    dv = base_directions[idx]
    bp = base_points[idx]
    r  = np.asarray(robot_base, float)

    # plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Prune #{idx}: candidates={num_samples}, radius={radius}")

    # scatter points
    ax.scatter(*pt, color='k', s=50, label='prune point')
    ax.scatter(*r,  color='g', s=50, label='robot base')

    # branch direction quiver
    dv_n = dv / np.linalg.norm(dv)
    ax.quiver(*pt, *(dv_n * scale),
              color='r', linewidth=2,
              arrow_length_ratio=0.1, label='branch direction')

    # collect candidate endpoints
    endpoints = []
    z = dv_n
    arb = np.array([1,0,0]) if abs(z[0])<0.9 else np.array([0,1,0])
    u = np.cross(z, arb); u /= np.linalg.norm(u)
    w = np.cross(z, u)
    for k in range(num_samples):
        θ = 2*np.pi * k / num_samples
        x_c = np.cos(θ)*u + np.sin(θ)*w
        endpoints.append(pt + radius * x_c)
    endpoints = np.array(endpoints)

    # highlight best
    _, best_quat = prune_fn(pt, dv, bp, r, num_samples=num_samples, radius=radius)
    R_best = R.from_quat(best_quat).as_matrix()
    x_best = R_best[:,0]
    best_endpt = pt + radius * x_best

    # plot all candidates (gray)
    for end in endpoints:
        ax.plot([pt[0], end[0]*scale],
                [pt[1], end[1]*scale],
                [pt[2], end[2]*scale],
                color='0.8', linewidth=1)
    ax.scatter(endpoints[:,0]*scale, endpoints[:,1]*scale,
               endpoints[:,2]*scale, color='0.8', s=10)

    # best candidate (blue)
    ax.plot([pt[0], best_endpt[0]*scale],
            [pt[1], best_endpt[1]*scale],
            [pt[2], best_endpt[2]*scale],
            color='b', linewidth=2, label='best offset')
    ax.scatter(best_endpt[0]*scale, best_endpt[1]*scale,
               best_endpt[2]*scale, color='b', s=50)

    # ---- set truly equal aspect ----
    # gather all points we plotted
    all_pts = np.vstack([
        pt,
        r,
        endpoints * scale,
        best_endpt * scale
    ])
    mins = all_pts.min(axis=0) - radius
    maxs = all_pts.max(axis=0) + radius
    ranges = maxs - mins
    max_range = ranges.max()
    # center
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

    yaml_path = "/home/marcus/IMML/manipulator_codesign/results/all_branches_info.yaml"
    prune_points, base_directions, base_points = extract_prune_points(yaml_path)

    # define your robot base (replace or query dynamically)
    robot_base = np.array([-1.0, -0.5, 1.025])

    viz_prune_pose_candidates(
        prune_points,
        base_directions,
        base_points,
        prune_pose,        # your new selection fn
        robot_base,
        idx=10,
        num_samples=60,
        radius=1.0,
        scale=1.0
    )