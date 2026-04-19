import open3d as o3d
import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def keep_points_in_front_of_any_camera(points_world, poses, use_positive_z=False):
    keep_mask = np.zeros(len(points_world), dtype=bool)

    for T in poses:
        R = T[:3, :3]
        C = T[:3, 3]

        # c2w: Xw = R Xc + C
        # so world -> camera is Xc = R^T (Xw - C)
        points_cam = (points_world - C) @ R

        z = points_cam[:, 2]
        if use_positive_z:
            visible = z > 1e-6
        else:
            visible = z < -1e-6

        keep_mask |= visible

    return ~keep_mask

def estimate_similarity_transform(src, dst):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    s = np.sum(S) / np.sum(src_centered ** 2)
    t = dst_mean - s * (R @ src_mean)
    return s, R, t

def transform_point(p, s, R, t):
    return s * (R @ p) + t

# Load
pcd = o3d.io.read_point_cloud("InstantSplatPP/output_infer/test/Sanjay/4_views/point_cloud/iteration_500/point_cloud.ply")

poses = np.load("InstantSplatPP/output_infer/test/Sanjay/4_views/pose/ours_500/pose_optimized.npy")
camera_centers = poses[:, :3, 3]

camera_centers_fake = poses[:, :3, 3]
camera_centers_real = np.array([
    [0.302, -0.11777, -0.14644],
    [0.66922, -0.37588, -0.9658],
    [0.97686, -0.10539, -0.00422],
    [0.68520, 0.27639, -0.08429],
], dtype=float)

scale, R_align, t_align = estimate_similarity_transform(
    camera_centers_fake,
    camera_centers_real
)



# 1. Statistical outlier removal
pcd, ind = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

# 2. Radius outlier removal
pcd, ind = pcd.remove_radius_outlier(
    nb_points=8,
    radius=0.01
)

# Save
o3d.io.write_point_cloud("cleaned.ply", pcd)

# Segment table plane

plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,
    ransac_n=3,
    num_iterations=1000
)

[a, b, c, d] = plane_model

# Separate plane and non-plane points
plane_cloud = pcd.select_by_index(inliers)
object_cloud = pcd.select_by_index(inliers, invert=True)

# Color plane green
plane_cloud.paint_uniform_color([0.0, 1.0, 0.0])

# Compute signed distance of object points to plane
points = np.asarray(object_cloud.points)

mask = keep_points_in_front_of_any_camera(points, poses, use_positive_z=False)
object_cloud = object_cloud.select_by_index(np.where(mask)[0])

points = np.asarray(object_cloud.points)

normal = np.array([a, b, c])
norm = np.linalg.norm(normal)

distances = (points @ normal + d) / norm

# Flip distances if needed so "above table" is positive
if np.mean(distances) < 0:
    distances = -distances
    normal = -normal

# Find point farthest from plane
idx = np.argmax(distances)
furthest_point = points[idx]
furthest_point_real = transform_point(furthest_point, scale, R_align, t_align)

print("Furthest point from plane:", furthest_point_real)

# Color object cloud gray, with furthest point red
colors = np.zeros_like(points)
colors[:] = [0.7, 0.7, 0.7]
colors[idx] = [1.0, 0.0, 0.0]
object_cloud.colors = o3d.utility.Vector3dVector(colors)

# Add a red sphere marker at the furthest point
bbox = object_cloud.get_axis_aligned_bounding_box()
extent = np.linalg.norm(bbox.get_extent())
sphere_radius = max(extent * 0.01, 0.005)  # scales with cloud size

sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
sphere.translate(furthest_point)
sphere.paint_uniform_color([1.0, 0.0, 0.0])

# Add coordinate frame

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.2)

arbitrary = np.array([1, 0, 0])
if abs(np.dot(arbitrary, normal)) > 0.9:
    arbitrary = np.array([0, 1, 0])

# build orthonormal frame
x_axis = np.cross(arbitrary, normal)
x_axis /= np.linalg.norm(x_axis)

y_axis = np.cross(normal, x_axis)

# rotation matrix (columns = axes)
R = np.stack([x_axis, y_axis, normal], axis=1)

frame.rotate(R, center=(0, 0, 0))
frame.translate(furthest_point)


# --- Visualization setup ---
geoms = []

# Add original clouds (reuse your existing ones if already defined)
plane_cloud.paint_uniform_color([0, 1, 0])     # green
object_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # gray

geoms.append(plane_cloud)
geoms.append(object_cloud)

# Add furthest point sphere (reuse yours if already created)
bbox = object_cloud.get_axis_aligned_bounding_box()
extent = np.linalg.norm(bbox.get_extent())
sphere_radius = max(extent * 0.01, 0.005)

furthest_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
furthest_sphere.translate(furthest_point)
furthest_sphere.paint_uniform_color([1, 0, 0])  # red

geoms.append(furthest_sphere)

# --- Add camera spheres ---
for i, cam_center in enumerate(camera_centers):
    cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 0.7)
    cam_sphere.translate(cam_center)
    cam_sphere.paint_uniform_color([0, 0, 1])  # blue

    # optional: add coordinate frame for each camera
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.1)
    frame.translate(cam_center)

    geoms.append(cam_sphere)
    # geoms.append(frame)

# --- Global coordinate frame ---
# global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent * 0.2)
# geoms.append(global_frame)

# --- Visualize everything ---
o3d.visualization.draw_geometries(
    geoms,
    window_name="Point Cloud + Cameras + Furthest Point"
)

# Visualize
# o3d.visualization.draw_geometries(
#     [plane_cloud, object_cloud, sphere, frame],
#     window_name="Plane, Object Cloud, and Furthest Point"
# )

# o3d.io.write_point_cloud("scene.ply", geoms)