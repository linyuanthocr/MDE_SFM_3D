import numpy as np
import pycolmap
import plotly.graph_objects as go
import os

import open3d as o3d
import numpy as np

def downsample_point_cloud_open3d(point_cloud_np, voxel_size=0.08):
    """
    使用 Open3D 进行点云体素下采样。

    Args:
        point_cloud_np (np.ndarray): 输入点云，形状为 (N, 3)。
        voxel_size (float): 体素的大小。

    Returns:
        np.ndarray: 下采样后的点云。
    """

    # 将 NumPy 数组转换为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    # 体素下采样
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 将 Open3D 点云对象转换为 NumPy 数组
    downsampled_points = np.asarray(downsampled_pcd.points)

    return downsampled_points

def filter_point_cloud_open3d(point_cloud_np, nb_neighbors=100, std_ratio=2.0):
    """
    使用 Open3D 进行点云滤波。

    Args:
        point_cloud_np (np.ndarray): 输入点云，形状为 (N, 3)。
        nb_neighbors (int): 用于统计滤波的邻居数量。
        std_ratio (float): 标准差的倍数，用于确定阈值。

    Returns:
        np.ndarray: 过滤后的点云。
    """

    # 将 NumPy 数组转换为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)

    # 统计滤波
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                std_ratio=std_ratio)
    # 获取过滤后的点云
    filtered_pcd = pcd.select_by_index(ind)

    # 将 Open3D 点云对象转换为 NumPy 数组
    filtered_points = np.asarray(filtered_pcd.points)

    return filtered_points

def undistort_pixel_to_camera(x, y, camera):
    """
    将像素坐标 (x, y) 转换为去畸变的归一化相机坐标 (x_u, y_u)
    支持 SIMPLE_RADIAL, PINHOLE, OPENCV 模型

    Args:
        x (float or np.ndarray): 像素 x 坐标
        y (float or np.ndarray): 像素 y 坐标
        camera (pycolmap.Camera): 相机对象

    Returns:
        x_u, y_u: 去畸变后的归一化坐标（相机方向单位向量的前两维）
    """
    model = camera.model.name
    params = camera.params

    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = params
        x_n = (x - cx) / f
        y_n = (y - cy) / f
        r2 = x_n**2 + y_n**2
        distortion = 1 + k * r2
        x_u = x_n * distortion
        y_u = y_n * distortion

    elif model in ["PINHOLE", "OPENCV"]:
        fx, fy, cx, cy = params[:4]
        x_u = (x - cx) / fx
        y_u = (y - cy) / fy

    else:
        raise ValueError(f"Unsupported camera model: {model}")

    return x_u, y_u

def estimate_similarity_transform(source_points, target_points):
    """
    用 Umeyama 方法估计相似变换（含 scale, rotation, translation）

    Args:
        source_points (np.ndarray): (N, 3)
        target_points (np.ndarray): (N, 3)

    Returns:
        s (float): scale
        R (np.ndarray): (3, 3) rotation matrix
        t (np.ndarray): (3,) translation vector
        T (np.ndarray): (4, 4) transformation matrix
    """
    assert source_points.shape == target_points.shape
    N = source_points.shape[0]

    mu_src = np.mean(source_points, axis=0)
    mu_dst = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    dst_centered = target_points - mu_dst

    cov_matrix = (dst_centered.T @ src_centered) / N
    U, D, Vt = np.linalg.svd(cov_matrix)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    var_src = np.var(src_centered, axis=0).sum()
    scale = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - scale * R @ mu_src

    # 构建 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t

    return scale, R, t, T


def align_mde_to_colmap_with_icp(colmap_model_path, npy_dir):
    """
    使用 ICP 将 MDE 点云与 COLMAP 点云对齐，并将所有 MDE 点云连接起来投影到 COLMAP 重建空间。
    使用 SIMPLE_RADIAL 模型进行投影。

    Args:
        colmap_model_path (str): COLMAP 重建目录的路径。
        npy_dir (str): 包含 _raw_depth_meter.npy 文件的目录的路径。
    """
    try:
        reconstruction = pycolmap.Reconstruction(colmap_model_path)
    except Exception as e:
        print(f"❌ Error loading COLMAP reconstruction: {e}")
        return

    all_colmap_correspondence_points = []
    all_mde_correspondence_points = []
    all_mde_points_transformed = []
    count = 0
    for image_id, image in reconstruction.images.items():
        count = count+1
#        if count != 10:
#            continue
#        if count >3:
#            break
        image_name = image.name
        npy_file_path = os.path.join(npy_dir, os.path.splitext(image_name)[0] + "_raw_depth_meter.npy")

        if not os.path.exists(npy_file_path):
            print(f"⚠️ Warning: _raw_depth_meter.npy file not found for {image_name}")
            continue

        try:
            depth_map = np.load(npy_file_path)
        except Exception as e:
            print(f"❌ Error loading {npy_file_path}: {e}")
            continue

        camera = reconstruction.cameras[image.camera_id]
        if camera.model.name == "SIMPLE_RADIAL":
            f, cx, cy, k = camera.params
        elif camera.model.name in ["PINHOLE", "OPENCV"]:
            fx, fy, cx, cy = camera.params[:4]
            f = fx
        else:
            raise ValueError(f"Unsupported camera model: {camera.model}")

        if f <= 0:
            print(f"⚠️ Skipping {image_name} due to invalid intrinsics.")
            continue

        T_cw_3x4 = np.array(image.cam_from_world.matrix())
        T_cw = np.eye(4)
        T_cw[:3, :4] = T_cw_3x4
        T_wc = np.linalg.inv(T_cw)
        R = T_wc[:3, :3]
        t = T_wc[:3, 3]
        print(R)
        print(t)
        rows, cols = depth_map.shape
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        x = x.flatten().astype(np.float32)
        y = y.flatten().astype(np.float32)
        z = depth_map.flatten()
        valid = np.isfinite(z) & (z > 0)
        x, y, z = x[valid], y[valid], z[valid]
        x_u, y_u = undistort_pixel_to_camera(x, y, camera)
        X = x_u * z
        Y = y_u * z
        Z = z
        points_cam = np.stack([X, Y, Z], axis=1)
        points_world = (R @ points_cam.T + t[:, np.newaxis]).T

        # 找到 COLMAP 重建点在 MDE 深度图中的对应点 (SIMPLE_RADIAL)
        colmap_points_3d = []
        mde_correspondence_points = []
        for point_id, point in reconstruction.points3D.items():
            point_3d = point.xyz
            T_cw_3x4 = np.array(image.cam_from_world.matrix())
            R = T_cw_3x4[:3, :3]
            t = T_cw[:3, 3]
            point_cam = R @ point_3d + t
            x_cam = point_cam[0] / point_cam[2]
            y_cam = point_cam[1] / point_cam[2]
            r2 = x_cam**2 + y_cam**2
            radial_distortion = 1 + k * r2
            x_distorted = x_cam * radial_distortion
            y_distorted = y_cam * radial_distortion
            u = f * x_distorted + cx
            v = f * y_distorted + cy

            if 0 <= int(v) < rows and 0 <= int(u) < cols:
                z_depth = depth_map[int(v), int(u)]
                if np.isfinite(z_depth) and z_depth > 0:
                    x_u_depth, y_u_depth = undistort_pixel_to_camera(u, v, camera)
                    X_depth = x_u_depth * z_depth
                    Y_depth = y_u_depth * z_depth
                    Z_depth = z_depth
                    mde_correspondence_points.append([X_depth, Y_depth, Z_depth])
                    colmap_points_3d.append(point_3d)

        if colmap_points_3d:
            all_colmap_correspondence_points =[]
            all_colmap_correspondence_points.append(np.array(colmap_points_3d))
            all_mde_correspondence_points.append(np.array(mde_correspondence_points))
            
            scale, R, t, T = estimate_similarity_transform(np.array(mde_correspondence_points), np.array(colmap_points_3d))

            # 将 MDE 点预对齐到 COLMAP 空间
            mde_correspondence_points_h = np.concatenate([mde_correspondence_points, np.ones((len(mde_correspondence_points), 1))], axis=1)
            SRT_aligned_mde_points = (T @ mde_correspondence_points_h.T)[:3].T
            
            points_cam_h = np.concatenate([points_cam, np.ones((len(points_world), 1))], axis=1)
            SRT_points_cam = (T @ points_cam_h.T)[:3].T
            
            # ICP 对齐
            source = o3d.geometry.PointCloud()
            target = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(SRT_aligned_mde_points)
            target.points = o3d.utility.Vector3dVector(colmap_points_3d)
            
            reg_result = o3d.pipelines.registration.registration_icp(
                source, target, max_correspondence_distance=0.02, init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            transformation = reg_result.transformation  # ← 关键修复！
            print(transformation)
            # 将 MDE 点投影到 COLMAP 重建空间
            mde_points_transformed = (transformation @ np.concatenate((SRT_points_cam.T, np.ones((1, SRT_points_cam.shape[0]))), axis=0))[:3].T
            all_mde_points_transformed.append(mde_points_transformed)

    if not all_colmap_correspondence_points or not all_mde_correspondence_points:
        print("❌ No valid correspondences found.")
        return

    all_colmap_correspondence_points = np.concatenate(all_colmap_correspondence_points, axis=0)
    print("all_colmap_correspondence_points shape:", all_colmap_correspondence_points.shape)
    all_mde_points_transformed = np.concatenate(all_mde_points_transformed, axis=0)
    print("all_mde_points_transformed shape:", all_mde_points_transformed.shape)

    # 合并点云
    merged_points = np.concatenate((all_colmap_correspondence_points, all_mde_points_transformed), axis=0)
    all_points = merged_points
    # ... (后续的点云处理和可视化部分)
    all_points = downsample_point_cloud_open3d(merged_points)
    all_points = filter_point_cloud_open3d(all_points)

    reconstruction = pycolmap.Reconstruction(colmap_model_path)
    colmap_points = []
    for point_id, point in reconstruction.points3D.items():
        colmap_points.append(point.xyz)
    colmap_points = np.array(colmap_points)

    # === 打印点云坐标统计信息 ===
    print("\n=== 点云坐标统计信息 ===")
    print("Min X:", np.min(all_points[:, 0]))
    print("Max X:", np.max(all_points[:, 0]))
    print("Min Y:", np.min(all_points[:, 1]))
    print("Max Y:", np.max(all_points[:, 1]))
    print("Min Z:", np.min(all_points[:, 2]))
    print("Max Z:", np.max(all_points[:, 2]))
    print("Mean X:", np.mean(all_points[:, 0]))
    print("Mean Y:", np.mean(all_points[:, 1]))
    print("Mean Z:", np.mean(all_points[:, 2]))

    # === Plot with Plotly ===
    fig = go.Figure(data=[
        go.Scatter3d(
            x=all_points[:, 0],
            y=all_points[:, 1],
            z=all_points[:, 2],
            mode='markers',
            marker=dict(size=1.5, color=all_points[:, 2], colorscale='Viridis', opacity=0.8)
        ),
#        go.Scatter3d(
#            x=colmap_points[:, 0],
#            y=colmap_points[:, 1],
#            z=colmap_points[:, 2],
#            mode='markers',
#            marker=dict(size=3, color='gray')
#        ),
#        go.Scatter3d(
#            x=all_colmap_correspondence_points[:, 0],
#            y=all_colmap_correspondence_points[:, 1],
#            z=all_colmap_correspondence_points[:, 2],
#            mode='markers',
#            marker=dict(size=3, color='red')  # 绘制红色球体
#        ),
#        go.Scatter3d(
#            x=SRT_aligned_mde_points[:, 0],
#            y=SRT_aligned_mde_points[:, 1],
#            z=SRT_aligned_mde_points[:, 2],
#            mode='markers',
#            marker=dict(size=3, color='blue')  # 绘制红色球体
#        ),
    ])

    fig.update_layout(
        title="Merged 3D Point Cloud",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()

# === Modify these paths before running ===
colmap_model_path = "/Users/helen/Documents/3DGS_data/small_spiddy_run2/0"
npy_dir = "/Users/helen/Documents/3D/depth_anything_v2/metric_depth/small_spiddy/output"

if __name__ == "__main__":
    align_mde_to_colmap_with_icp(colmap_model_path, npy_dir)
