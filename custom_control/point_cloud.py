# 可視化比較不同操作的效果
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

def save_grid_logits_as_pointcloud(
                                   xyz_samples: torch.Tensor,
                                   grid_logits: torch.Tensor, 
                                   output_path: str = "grid_logits_pointcloud.ply",
                                   format_type: str = "ply",
                                   threshold: float = -5.0,
                                   downsample_factor: int = 1,
                                   colormap: str = "coolwarm",
                                   min_val=None,
                                   max_val=None):
    """
    將3D logits列表轉換為點雲格式並儲存，使用提供的xyz座標
    
    適合的3D點雲格式：
    1. PLY (Polygon File Format) - 最常用，支援顏色
    2. PCD (Point Cloud Data) - PCL library格式
    3. XYZ - 簡單文字格式
    4. OFF - Object File Format
    
    Args:
        xyz_samples: shape (N, 3) 的tensor，包含每個點的實際xyz座標
        grid_logits: shape (N,) 或 (1, N) 的tensor，包含每個點的logit值
        output_path: 輸出檔案路徑（不含副檔名）
        format_type: 格式類型 ("ply", "xyz", "pcd")
        threshold: 只保存小於此閾值的點（過濾背景，SDF通常負值表示內部）
        downsample_factor: 採樣因子，用於減少點的數量（每downsample_factor個點取1個）
        colormap: 色彩映射類型 ("viridis", "plasma", "inferno", "magma", "hot", "coolwarm", "RdYlBu_r")
    """
    
    if isinstance(grid_logits, torch.Tensor):
        grid_logits = grid_logits.cpu().numpy()
    
    if isinstance(xyz_samples, torch.Tensor):
        xyz_samples = xyz_samples.cpu().numpy()
    
    # 處理維度：確保grid_logits是1D數組
    if grid_logits.ndim > 1:
        grid_logits = grid_logits.squeeze()
    
    # 確保xyz_samples是2D數組 (N, 3)
    if xyz_samples.ndim == 3:
        xyz_samples = xyz_samples.squeeze(0)
    
    print(f"XYZ samples shape: {xyz_samples.shape}")
    print(f"Grid logits shape: {grid_logits.shape}")
    print(f"Value range: {grid_logits.min():.3f} to {grid_logits.max():.3f}")
    
    # 確保數據維度匹配
    if len(xyz_samples) != len(grid_logits):
        raise ValueError(f"xyz_samples length {len(xyz_samples)} doesn't match grid_logits length {len(grid_logits)}")
    
    # 使用所有點（因為輸入已經是列表格式，不是網格）
    points = xyz_samples
    values = grid_logits
    
    # 可選的下採樣：每downsample_factor個點取1個
    if downsample_factor > 1:
        indices = np.arange(0, len(points), downsample_factor)
        points = points[indices]
        values = values[indices]
        print(f"Downsampled to {len(points)} points (factor: {downsample_factor})")
    
    # 根據閾值過濾點（對於SDF，通常threshold=0，負值表示物體內部）
    print(f"Keeping points below threshold {threshold}")
    mask = values < threshold
    filtered_points = points[mask]
    filtered_values = values[mask]
    if len(filtered_points) != 0:
        print(f"Filtered points: {len(filtered_points)} / {len(points)}")
        print(f"Point coordinate range: X[{filtered_points[:, 0].min():.3f}, {filtered_points[:, 0].max():.3f}], "
            f"Y[{filtered_points[:, 1].min():.3f}, {filtered_points[:, 1].max():.3f}], "
            f"Z[{filtered_points[:, 2].min():.3f}, {filtered_points[:, 2].max():.3f}]")
    else:
        print("No points remain after filtering. ")
    
    if len(filtered_points) == 0:
        print("Warning: No points remain after filtering!")
        return None
    
    # 將數值映射到顏色（使用matplotlib heatmap顏色映射）
    # 正規化到0-1範圍
    if min_val is None:
        min_val = filtered_values.min()
    if max_val is None:
        max_val = filtered_values.max()
    normalized_values = (filtered_values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(filtered_values)
    
    # 使用matplotlib色彩映射
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # 選擇色彩映射 - 提供多種heatmap顏色選擇
    colormap_obj = cm.get_cmap(colormap)
    
    # 將正規化值映射到RGB顏色
    colors_rgba = colormap_obj(normalized_values)
    colors = (colors_rgba[:, :3] * 255).astype(np.uint8)  # 轉換為0-255範圍的RGB
    
    if format_type.lower() == "ply":
        output_file = f"{output_path}"
        save_ply_pointcloud(filtered_points, colors, filtered_values, output_file)
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    print(f"Point cloud saved to: {output_file}")
    return output_file

def save_ply_pointcloud(points, colors, values, output_path):
    """儲存為PLY格式（最推薦，支援多數3D軟體）"""
    with open(output_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float value\n")
        f.write("end_header\n")
        
        # 點數據
        for i in range(len(points)):
            f.write(f"{points[i,0]:.3f} {points[i,1]:.3f} {points[i,2]:.3f} "
                   f"{colors[i,0]} {colors[i,1]} {colors[i,2]} {values[i]:.6f}\n")

def visualize_point_clouds(original, dilated, eroded, opened, closed, sample_size=1000):
    """
    可視化不同形態學操作的結果
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 隨機採樣以便可視化
    def sample_points(points, n):
        if len(points) > n:
            indices = np.random.choice(len(points), n, replace=False)
            return points[indices]
        return points
    
    # 原始點雲
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    orig_sample = sample_points(original, sample_size)
    ax1.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2], 
               c='blue', s=1, alpha=0.6)
    ax1.set_title(f'Original ({len(original)} points)')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    
    # 膨脹
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    dil_sample = sample_points(dilated, sample_size)
    ax2.scatter(dil_sample[:, 0], dil_sample[:, 1], dil_sample[:, 2], 
               c='red', s=1, alpha=0.6)
    ax2.set_title(f'Dilated ({len(dilated)} points)')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
    
    # 侵蝕
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ero_sample = sample_points(eroded, sample_size)
    ax3.scatter(ero_sample[:, 0], ero_sample[:, 1], ero_sample[:, 2], 
               c='green', s=1, alpha=0.6)
    ax3.set_title(f'Eroded ({len(eroded)} points)')
    ax3.set_xlabel('X'), ax3.set_ylabel('Y'), ax3.set_zlabel('Z')
    
    # 開運算
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    open_sample = sample_points(opened, sample_size)
    ax4.scatter(open_sample[:, 0], open_sample[:, 1], open_sample[:, 2], 
               c='orange', s=1, alpha=0.6)
    ax4.set_title(f'Opened ({len(opened)} points)')
    ax4.set_xlabel('X'), ax4.set_ylabel('Y'), ax4.set_zlabel('Z')
    
    # 閉運算
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    close_sample = sample_points(closed, sample_size)
    ax5.scatter(close_sample[:, 0], close_sample[:, 1], close_sample[:, 2], 
               c='purple', s=1, alpha=0.6)
    ax5.set_title(f'Closed ({len(closed)} points)')
    ax5.set_xlabel('X'), ax5.set_ylabel('Y'), ax5.set_zlabel('Z')
    
    # 統計比較
    ax6 = fig.add_subplot(2, 3, 6)
    operations = ['Original', 'Dilated', 'Eroded', 'Opened', 'Closed']
    point_counts = [len(original), len(dilated), len(eroded), len(opened), len(closed)]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    bars = ax6.bar(operations, point_counts, color=colors, alpha=0.7)
    ax6.set_title('Point Count Comparison')
    ax6.set_ylabel('Number of Points')
    ax6.tick_params(axis='x', rotation=45)
    
    # 在柱狀圖上標註數值
    for bar, count in zip(bars, point_counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(point_counts)*0.01,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./output/tf3de_functions/morphological_operations_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
