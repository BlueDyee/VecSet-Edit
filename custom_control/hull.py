from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import numpy as np
import trimesh
import os


def remove_outliers_weighted_sor(
    points_3d,
    values_3d,
    k=10,
    alpha_base=1.5,
    beta=1.5,
    concentration_factor=2.0,
    min_weight_threshold=0.1,
):

    # 正規化權重到 [0, 1]
    weights = (values_3d - values_3d.min()) / (values_3d.max() - values_3d.min() + 1e-8)

    # 增強權重集中度：使用指數放大高權重點
    weights_concentrated = np.power(weights, concentration_factor)
    weights_concentrated = weights_concentrated / (weights_concentrated.max() + 1e-8)
    w_max = weights_concentrated.max()

    # 預先過濾權重過低的點
    weight_mask = weights_concentrated >= min_weight_threshold
    if weight_mask.sum() == 0:
        return np.array([]).reshape(0, 3), np.array([]), np.array([], dtype=bool)

    points_filtered = points_3d[weight_mask]
    weights_filtered = weights_concentrated[weight_mask]

    # 建立 k-NN（在過濾後的點集上）
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(points_filtered))).fit(
        points_filtered
    )
    distances, indices = nbrs.kneighbors(points_filtered)

    # 計算每個點的加權平均鄰距
    weighted_mean_distances = []
    for i in range(len(points_filtered)):
        neighbor_indices = indices[i, 1:]  # 排除自己
        neighbor_distances = distances[i, 1:]
        neighbor_weights = weights_filtered[neighbor_indices]

        # 加權平均距離公式
        if len(neighbor_weights) > 0 and neighbor_weights.sum() > 0:
            weighted_dist = (
                np.sum(neighbor_weights * neighbor_distances) / neighbor_weights.sum()
            )
        else:
            weighted_dist = (
                np.mean(neighbor_distances) if len(neighbor_distances) > 0 else 0
            )
        weighted_mean_distances.append(weighted_dist)

    weighted_mean_distances = np.array(weighted_mean_distances)

    # 計算全域加權均值和標準差
    total_weight = weights_filtered.sum()
    if total_weight > 0:
        mu_w = np.sum(weights_filtered * weighted_mean_distances) / total_weight
        sigma_w = np.sqrt(
            np.sum(weights_filtered * (weighted_mean_distances - mu_w) ** 2)
            / total_weight
        )
    else:
        mu_w = np.mean(weighted_mean_distances)
        sigma_w = np.std(weighted_mean_distances)

    # 更嚴格的動態閾值：權重越低，門檻越嚴格（相反方向）
    alpha_dynamic = (
        alpha_base - beta * weights_filtered
    )  # 改為減法，權重高的點門檻更寬鬆
    thresholds = mu_w + alpha_dynamic * sigma_w

    # 判斷 outliers
    inliers_mask_filtered = weighted_mean_distances <= thresholds

    # 映射回原始數據
    final_mask = np.zeros(len(points_3d), dtype=bool)
    filtered_indices = np.where(weight_mask)[0]
    final_mask[filtered_indices[inliers_mask_filtered]] = True

    return points_3d[final_mask], values_3d[final_mask], final_mask


def remove_outliers_two_stage(
    points_3d,
    values_3d,
    # 第一階段：清除邊緣
    k_stage1=300,
    alpha_stage1=1.0,
    beta_stage1=1.0,
    concentration_factor_stage1=1.0,
    min_weight_threshold_stage1=0.05,
    # 第二階段：聚焦中心
    k_stage2=50,
    alpha_stage2=0.5,
    beta_stage2=2.0,
    concentration_factor_stage2=3.0,
    min_weight_threshold_stage2=0.2,
):

    # 第一階段：用大 k 清除邊緣點
    points_stage1, values_stage1, mask_stage1 = remove_outliers_weighted_sor(
        points_3d,
        values_3d,
        k=k_stage1,
        alpha_base=alpha_stage1,
        beta=beta_stage1,
        concentration_factor=concentration_factor_stage1,
        min_weight_threshold=min_weight_threshold_stage1,
    )

    if len(points_stage1) == 0:
        return np.array([]).reshape(0, 3), np.array([]), np.array([], dtype=bool)

    points_stage2, values_stage2, mask_stage2 = remove_outliers_weighted_sor(
        points_stage1,
        values_stage1,
        k=k_stage2,
        alpha_base=alpha_stage2,
        beta=beta_stage2,
        concentration_factor=concentration_factor_stage2,
        min_weight_threshold=min_weight_threshold_stage2,
    )

    # 合併兩階段的 mask
    final_mask = np.zeros(len(points_3d), dtype=bool)
    stage1_indices = np.where(mask_stage1)[0]
    final_mask[stage1_indices[mask_stage2]] = True

    return points_stage2, values_stage2, final_mask


def find_main_cluster_center(points_3d, values_3d, cluster_radius_factor=0.3):

    # 加權中心：權重高的點影響更大
    weights = (values_3d - values_3d.min()) / (values_3d.max() - values_3d.min() + 1e-8)
    weighted_center = np.average(points_3d, weights=weights, axis=0)

    # 計算每個點到加權中心的距離
    distances = np.linalg.norm(points_3d - weighted_center, axis=1)

    # 動態半徑：基於權重分布
    high_weight_mask = weights > np.percentile(weights, 75)
    high_weight_distances = distances[high_weight_mask]
    cluster_radius = np.percentile(high_weight_distances, 75) * cluster_radius_factor

    # 只保留在聚類半徑內的點
    center_mask = distances <= cluster_radius

    return points_3d[center_mask], values_3d[center_mask], center_mask, weighted_center


def compute_3d_convex_hull(points_3d):
    """計算 3D 點集的凸包"""
    try:
        # 計算凸包
        hull = ConvexHull(points_3d)

        # 獲取凸包頂點的索引
        hull_vertex_indices = hull.vertices

        # 獲取凸包表面的三角形面片
        hull_faces = hull.simplices

        # 創建布爾遮罩，標記哪些點在凸包上
        is_on_hull = np.zeros(len(points_3d), dtype=bool)
        is_on_hull[hull_vertex_indices] = True

        return {
            "hull_vertices": points_3d[hull_vertex_indices],
            "hull_vertex_indices": hull_vertex_indices,
            "hull_faces": hull_faces,
            "is_on_hull_mask": is_on_hull,
            "hull_volume": hull.volume,
            "hull_area": hull.area,
        }
    except Exception as e:
        print(f"計算凸包時出錯: {e}")
        return None


def to_convex_hull(points_3d):

    hull_result = compute_3d_convex_hull(points_3d)
    hull_mesh = trimesh.Trimesh(vertices=points_3d, faces=hull_result["hull_faces"])
    return hull_mesh


def expand_convex_hull_uniform(points_3d, scale_factor=1.2):
    """
    均勻放大凸包：從重心向外按比例放大

    Args:
        points_3d: 凸包頂點坐標 (N, 3)
        scale_factor: 放大倍數，>1為放大，<1為縮小

    Returns:
        expanded_points: 放大後的頂點坐標
    """
    # 計算重心
    centroid = np.mean(points_3d, axis=0)

    # 從重心向外放大
    vectors_from_center = points_3d - centroid
    expanded_points = centroid + vectors_from_center * scale_factor

    return expanded_points


def expand_convex_hull_normal(hull_vertices, hull_faces, expansion_distance=0.1):
    """
    沿法向量方向推出凸包頂點

    Args:
        hull_vertices: 凸包頂點坐標 (N, 3)
        hull_faces: 凸包面片索引 (M, 3)
        expansion_distance: 向外推出的距離

    Returns:
        expanded_vertices: 推出後的頂點坐標
    """
    # 創建trimesh對象來計算頂點法向量
    mesh = trimesh.Trimesh(vertices=hull_vertices, faces=hull_faces)

    # 獲取每個頂點的法向量（向外）
    vertex_normals = mesh.vertex_normals

    # 沿法向量方向推出頂點
    expanded_vertices = hull_vertices + vertex_normals * expansion_distance

    return expanded_vertices


def expand_convex_hull_adaptive(
    hull_vertices, hull_faces, base_distance=0.1, curvature_factor=1.0
):
    """
    自適應擴展：根據局部曲率調整擴展距離

    Args:
        hull_vertices: 凸包頂點坐標 (N, 3)
        hull_faces: 凸包面片索引 (M, 3)
        base_distance: 基礎擴展距離
        curvature_factor: 曲率影響因子

    Returns:
        expanded_vertices: 自適應擴展後的頂點坐標
    """
    # 創建trimesh對象
    mesh = trimesh.Trimesh(vertices=hull_vertices, faces=hull_faces)

    # 獲取頂點法向量
    vertex_normals = mesh.vertex_normals

    # 計算局部曲率（簡化版本：使用相鄰面的角度變化）
    curvatures = np.ones(len(hull_vertices))  # 預設曲率為1

    try:
        # 使用trimesh的離散曲率計算
        curvatures = mesh.vertex_defects  # 使用角度缺陷作為曲率指標
        curvatures = np.abs(curvatures)
        curvatures = curvatures / (curvatures.max() + 1e-8)  # 正規化到[0,1]
    except:
        pass  # 如果計算失敗，使用預設值

    # 根據曲率調整擴展距離
    adaptive_distances = base_distance * (1 + curvature_factor * curvatures)

    # 沿法向量方向推出，距離根據曲率調整
    expanded_vertices = (
        hull_vertices + vertex_normals * adaptive_distances[:, np.newaxis]
    )

    return expanded_vertices


def expand_convex_hull_with_mesh_output(
    hull_mesh, expansion_method="uniform", **kwargs
):
    """
    擴展凸包並返回新的mesh

    Args:
        hull_mesh: 原始凸包mesh (trimesh.Trimesh object)
        expansion_method: 擴展方法 ("uniform", "normal", "adaptive")
        **kwargs: 各種擴展方法的參數

    Returns:
        expanded_mesh: 擴展後的凸包mesh
    """
    # 從mesh對象獲取頂點和面片
    if hull_mesh is None:
        return None

    original_vertices = hull_mesh.vertices
    original_faces = hull_mesh.faces

    # 根據選擇的方法擴展
    if expansion_method == "uniform":
        scale_factor = kwargs.get("scale_factor", 1.2)
        expanded_vertices = expand_convex_hull_uniform(original_vertices, scale_factor)
    elif expansion_method == "normal":
        expansion_distance = kwargs.get("expansion_distance", 0.1)
        expanded_vertices = expand_convex_hull_normal(
            original_vertices, original_faces, expansion_distance
        )
    elif expansion_method == "adaptive":
        base_distance = kwargs.get("base_distance", 0.1)
        curvature_factor = kwargs.get("curvature_factor", 1.0)
        expanded_vertices = expand_convex_hull_adaptive(
            original_vertices, original_faces, base_distance, curvature_factor
        )
    else:
        raise ValueError(f"未知的擴展方法: {expansion_method}")

    # 創建擴展後的mesh
    expanded_mesh = trimesh.Trimesh(vertices=expanded_vertices, faces=original_faces)

    return expanded_mesh


def save_points_cloud(points_3d, values_3d, save_path, colormap="coolwarm"):
    # 使用matplotlib viridis色彩映射 (深藍->綠->黃->紅)
    # 也可以選擇其他色彩映射: 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'coolwarm'
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # 選擇色彩映射 - 提供多種heatmap顏色選擇
    colormap_obj = cm.get_cmap(colormap)

    # 將正規化值映射到RGB顏色
    colors_rgba = colormap_obj(values_3d)
    colors = (colors_rgba[:, :3] * 255).astype(np.uint8)  # 轉換為0-255範圍的RGB
    with open(save_path, "w") as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float value\n")
        f.write("end_header\n")

        for i in range(len(points_3d)):
            f.write(
                f"{points_3d[i,0]:.3f} {points_3d[i,1]:.3f} {points_3d[i,2]:.3f} "
                f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values_3d[i]:.6f}\n"
            )
def find_vertices_in_hull(vertices, faces, hull_mesh):
    """
    检查并列出在3D Hull里的vertices，以及与Hull内vertices通过Face相连的vertices
    
    Args:
        vertices: 所有顶点坐标 (N, 3)
        faces: 面片索引 (M, k) - 每个面包含的顶点索引
        hull_mesh: 3D Hull mesh (trimesh.Trimesh object)
    
    Returns:
        dict: {
            'vertices_in_hull': 在Hull内的顶点索引列表,
            'vertices_in_hull_coords': 在Hull内的顶点坐标,
            'connected_vertices': 与Hull内顶点相连的顶点索引列表,
            'connected_vertices_coords': 与Hull内顶点相连的顶点坐标,
            'all_target_vertices': 所有目标顶点索引（Hull内+相连的）,
            'vertex_status': 每个顶点的状态 ('in_hull', 'connected', 'outside')
        }
    """
    
    # 1. 检查哪些顶点在Hull内
    vertices_in_hull_mask = hull_mesh.contains(vertices)
    vertices_in_hull_indices = np.where(vertices_in_hull_mask)[0]
    
    # 2. 构建顶点邻接图 - 找出通过面片相连的顶点
    vertex_neighbors = {}
    for i in range(len(vertices)):
        vertex_neighbors[i] = set()
    
    # 遍历所有面片，建立顶点之间的连接关系
    for face in faces:
        # 确保face是一维数组
        face_vertices = np.array(face).flatten()
        # 每个面片中的顶点都互相连接
        for i in range(len(face_vertices)):
            for j in range(i + 1, len(face_vertices)):
                v1, v2 = face_vertices[i], face_vertices[j]
                vertex_neighbors[v1].add(v2)
                vertex_neighbors[v2].add(v1)
    
    # 3. 找出与Hull内顶点相连的顶点
    connected_vertices_set = set()
    for hull_vertex_idx in vertices_in_hull_indices:
        # 添加所有与该Hull内顶点相连的顶点
        connected_vertices_set.update(vertex_neighbors[hull_vertex_idx])
    
    # 移除已经在Hull内的顶点（避免重复）
    connected_vertices_set = connected_vertices_set - set(vertices_in_hull_indices)
    connected_vertices_indices = list(connected_vertices_set)
    
    # 4. 创建顶点状态标记
    vertex_status = ['outside'] * len(vertices)
    for idx in vertices_in_hull_indices:
        vertex_status[idx] = 'in_hull'
    for idx in connected_vertices_indices:
        vertex_status[idx] = 'connected'
    
    # 5. 获取所有目标顶点
    all_target_indices = list(vertices_in_hull_indices) + connected_vertices_indices
    
    return {
        'vertices_in_hull': vertices_in_hull_indices.tolist(),
        'vertices_in_hull_coords': vertices[vertices_in_hull_indices],
        'connected_vertices': connected_vertices_indices,
        'connected_vertices_coords': vertices[connected_vertices_indices] if connected_vertices_indices else np.array([]).reshape(0, 3),
        'all_target_vertices': all_target_indices,
        'vertex_status': vertex_status
    }