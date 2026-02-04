import trimesh
import numpy as np
import imageio
from PIL import Image
import nvdiffrast.torch as dr
import torch
from tqdm import tqdm
import os
import argparse
import cv2

def render_mesh_with_nvdiffrast(mesh_path, output_path="nvdiff_mesh.mp4", resolution=1024, num_frames=120, r=2.0, fov=40,fps=20, rotation_mode='smooth'):
    """
    使用nvdiffrast渲染GLB文件（優化版）
    
    rotation_mode: 'smooth' (原始波動), 'horizontal' (水平旋轉), 'gentle' (輕微波動)
    """
    # 載入mesh
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()  # 使用新的API
    
    # 直接使用原始頂點，不做座標轉換
    vertices = mesh.vertices
    
    # 轉換為torch tensor
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).cuda().contiguous()
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.int32).cuda().contiguous()
    
    # 處理紋理和UV坐標
    has_texture = (hasattr(mesh.visual, 'material') and mesh.visual.material is not None and
                   hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and 
                   hasattr(mesh.visual.material, 'baseColorTexture'))
    
    if has_texture:
        # 檢查UV坐標的格式和範圍
        uv_coords_raw = mesh.visual.uv
        print(f"原始UV坐標形狀: {uv_coords_raw.shape}")
        print(f"UV坐標範圍: U({uv_coords_raw[:, 0].min():.3f}, {uv_coords_raw[:, 0].max():.3f}), V({uv_coords_raw[:, 1].min():.3f}, {uv_coords_raw[:, 1].max():.3f})")
        
        # 確保UV坐標在[0,1]範圍內
        uv_coords_processed = uv_coords_raw.copy()
        
        # 如果UV坐標超出[0,1]範圍，進行歸一化
        if uv_coords_processed.min() < 0 or uv_coords_processed.max() > 1:
            print("UV坐標超出[0,1]範圍，進行歸一化")
            uv_coords_processed[:, 0] = (uv_coords_processed[:, 0] - uv_coords_processed[:, 0].min()) / (uv_coords_processed[:, 0].max() - uv_coords_processed[:, 0].min())
            uv_coords_processed[:, 1] = (uv_coords_processed[:, 1] - uv_coords_processed[:, 1].min()) / (uv_coords_processed[:, 1].max() - uv_coords_processed[:, 1].min())
        
        # 檢查是否需要翻轉V坐標（常見問題）
        # OpenGL/nvdiffrast使用左下角為原點，而許多紋理使用左上角為原點
        uv_coords_processed[:, 1] = 1.0 - uv_coords_processed[:, 1]
        
        uv_coords = torch.tensor(uv_coords_processed, dtype=torch.float32).cuda().contiguous()
        
        # 處理紋理圖像
        texture_image = mesh.visual.material.baseColorTexture
        
        if hasattr(texture_image, 'size'):  # PIL Image
            texture_array = np.array(texture_image)
        else:
            texture_array = texture_image
            
        print(f"原始紋理大小: {texture_array.shape}")
        
        # 轉換紋理為torch tensor
        if len(texture_array.shape) == 3:
            # 確保是RGB格式
            if texture_array.shape[2] == 4:  # RGBA
                texture_array = texture_array[:, :, :3]  # 只取RGB
            elif texture_array.shape[2] == 1:  # 灰度
                texture_array = np.repeat(texture_array, 3, axis=2)
                
            texture_tensor = torch.tensor(texture_array, dtype=torch.float32).cuda() / 255.0
            texture_tensor = texture_tensor.unsqueeze(0).contiguous()  # [1, H, W, 3]
            
        else:
            # 處理灰度圖像
            texture_tensor = torch.tensor(texture_array, dtype=torch.float32).cuda() / 255.0
            texture_tensor = texture_tensor.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0).contiguous()
            
        print(f"處理後紋理大小: {texture_tensor.shape}")
        print(f"處理後UV坐標範圍: U({uv_coords[:, 0].min():.3f}, {uv_coords[:, 0].max():.3f}), V({uv_coords[:, 1].min():.3f}, {uv_coords[:, 1].max():.3f})")
        
    else:
        # 沒有紋理，使用法向量設置顏色
        print("沒有找到紋理，根據法向量設置顏色")
        
        # 計算頂點法向量
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            vertex_normals = mesh.vertex_normals
            print("使用現有的頂點法向量")
        else:
            print("計算頂點法向量")
            # 如果沒有法向量，計算面法向量然後平均到頂點
            face_normals = mesh.face_normals
            vertex_normals = np.zeros_like(vertices)
            
            # 將面法向量累加到對應的頂點
            for i, face in enumerate(mesh.faces):
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += face_normals[i]
            
            # 歸一化頂點法向量
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            vertex_normals = vertex_normals / norms
        
        # 將法向量轉換為顏色 (法向量從[-1,1]映射到[0,1])
        vertex_colors = (vertex_normals + 1.0) / 2.0
        vertex_colors = torch.tensor(vertex_colors, dtype=torch.float32).cuda().contiguous()
        
        print(f"頂點顏色範圍: R({vertex_colors[:, 0].min():.3f}, {vertex_colors[:, 0].max():.3f}), "
              f"G({vertex_colors[:, 1].min():.3f}, {vertex_colors[:, 1].max():.3f}), "
              f"B({vertex_colors[:, 2].min():.3f}, {vertex_colors[:, 2].max():.3f})")
        
        # 為法向量顏色設置UV坐標（實際上不會用到紋理採樣）
        uv_coords = torch.zeros((len(vertices), 2), dtype=torch.float32).cuda().contiguous()
        # 創建一個虛擬的1x1白色紋理
        texture_tensor = torch.ones((1, 1, 1, 3), dtype=torch.float32).cuda().contiguous()
    
    # 創建渲染上下文
    try:
        glctx = dr.RasterizeCudaContext()
        print("使用CUDA渲染上下文")
    except:
        glctx = dr.RasterizeGLContext()
        print("使用OpenGL渲染上下文")
        
    frames = []
    
    # 計算場景邊界盒以調整相機距離
    bbox_min = vertices_tensor.min(dim=0)[0]
    bbox_max = vertices_tensor.max(dim=0)[0]
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = (bbox_max - bbox_min).max()
    
    print(f"模型邊界框中心: {bbox_center}")
    print(f"模型大小: {bbox_size}")
    
    # 根據旋轉模式設定相機軌跡
    # 模仿原始render_video的相機設置，但修正Y/Z軸
    yaws = torch.linspace(0, 2 * np.pi, num_frames + 1)[:-1]  # 完整360度，不重複
    
    if rotation_mode == 'horizontal':
        # 固定水平視角
        pitch = torch.full((num_frames,), 0.25)
    elif rotation_mode == 'gentle':
        # 輕微上下波動
        pitch = 0.25 + 0.15 * torch.sin(torch.linspace(0, 2 * np.pi, num_frames))
    elif rotation_mode == 'overhead':
        # 從正面視角漸變到俯瞰視角，同時保持輕微水平波動
        # 起始角度 0.25 (約14度) 到 1.4 (約80度俯瞰)
        base_pitch = torch.linspace(0.25, 1.4, num_frames)
        gentle_variation = 0.1 * torch.sin(torch.linspace(0, 4 * np.pi, num_frames))
        pitch = base_pitch + gentle_variation
    else:  # 'smooth' 或其他
        # 原始大幅度波動
        pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * np.pi, num_frames))
    
    print(f"旋轉模式: {rotation_mode}")
    print(f"Yaw範圍: 0° 到 360° ({num_frames} 幀)")
    print(f"Pitch範圍: {torch.rad2deg(pitch.min()):.1f}° 到 {torch.rad2deg(pitch.max()):.1f}°")
    
    for i in tqdm(range(num_frames), desc="Rendering frames",total=num_frames):
        # 使用原始render_video的相機計算方式
        yaw = yaws[i]
        pitch_val = pitch[i]
        
        # 相機位置計算（修正Y/Z軸）
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch_val),  # X軸
            torch.sin(pitch_val),                   # Y軸（原來的Z軸）
            torch.cos(yaw) * torch.cos(pitch_val),  # Z軸（原來的Y軸）
        ]).cuda() * r
        
        # 調整相機位置到模型中心
        cam_pos = bbox_center + orig * bbox_size
        
        # lookAt計算 - 看向模型中心，up向量為Y軸
        target = bbox_center
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()  # Y軸向上
        
        # 計算視圖方向
        forward = torch.nn.functional.normalize(target - cam_pos, dim=0)
        right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
        up_corrected = torch.cross(right, forward, dim=0)
        
        # 構建視圖矩陣（修正Y/Z軸順序）
        view_matrix = torch.eye(4, dtype=torch.float32).cuda()
        view_matrix[0, :3] = right        # X軸
        view_matrix[1, :3] = up_corrected # Y軸
        view_matrix[2, :3] = -forward     # Z軸
        view_matrix[0, 3] = -torch.dot(right, cam_pos)
        view_matrix[1, 3] = -torch.dot(up_corrected, cam_pos)
        view_matrix[2, 3] = torch.dot(forward, cam_pos)
        
        # 投影矩陣（使用傳入的fov參數）
        fov_rad = fov * np.pi / 180.0
        aspect = 1.0
        near = bbox_size * 0.1
        far = bbox_size * 10.0
        
        f = 1.0 / np.tan(fov_rad / 2.0)
        proj_matrix = torch.zeros((4, 4), dtype=torch.float32).cuda()
        proj_matrix[0, 0] = f / aspect
        proj_matrix[1, 1] = f
        proj_matrix[2, 2] = (far + near) / (near - far)
        proj_matrix[2, 3] = (2 * far * near) / (near - far)
        proj_matrix[3, 2] = -1.0
        
        # MVP矩陣
        mvp = proj_matrix @ view_matrix
        
        # 變換頂點
        vertices_homo = torch.cat([vertices_tensor, torch.ones((len(vertices_tensor), 1), device=vertices_tensor.device)], dim=1)
        vertices_clip = (mvp @ vertices_homo.T).T.contiguous()
        
        # 光柵化
        rast, _ = dr.rasterize(glctx, vertices_clip.unsqueeze(0), faces_tensor, resolution=(resolution, resolution))
        
        if has_texture:
            # 使用紋理
            # 插值UV坐標
            uv_interp, _ = dr.interpolate(uv_coords.unsqueeze(0), rast, faces_tensor)
            # 紋理採樣
            color = dr.texture(texture_tensor, uv_interp, filter_mode='linear')
        else:
            # 使用法向量顏色
            # 插值頂點顏色
            color, _ = dr.interpolate(vertex_colors.unsqueeze(0), rast, faces_tensor)
        
        # 轉換為圖像
        color = color[0].cpu().numpy()
        
        # 創建純黑背景
        mask = rast[0, :, :, 3].cpu().numpy() > 0  # 獲取有效像素遮罩
        
        # 初始化為純黑背景
        final_image = np.zeros((resolution, resolution, 3), dtype=np.float32)
        
        # 只在有效像素位置填充顏色
        final_image[mask] = color[mask]
        
        # 轉換為uint8
        final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)
        
        # 翻轉Y軸（OpenGL到圖像坐標系）
        final_image = np.flipud(final_image)
        
        frames.append(final_image)
        
    
    # 保存視頻
    imageio.mimsave(output_path, frames, fps=fps)
    return frames



def get_actual_frame_count(reader):
    """安全地獲取影片的實際幀數"""
    try:
        # 嘗試獲取影片長度
        if hasattr(reader, '_meta') and 'nframes' in reader._meta:
            return reader._meta['nframes']
        
        # 如果上述方法失敗，通過實際讀取來計算
        count = 0
        try:
            while True:
                reader.get_data(count)
                count += 1
        except (IndexError, StopIteration):
            pass
        
        # 重置reader
        reader.close()
        return count
    except:
        return 0

def concat_videos_horizontally(video_paths, output_path, fps=20):
    """
    將三個影片的每一幀水平連接成一個大影片
    
    Args:
        video_paths: 三個影片文件的路徑列表
        output_path: 輸出影片的路徑
        fps: 輸出影片的幀率
    """
    print(f"Loading videos: {video_paths}")
    
    # 首先獲取所有影片的實際幀數
    frame_counts = []
    for video_path in video_paths:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # 用臨時reader獲取幀數
        temp_reader = imageio.get_reader(video_path)
        actual_frames = get_actual_frame_count(temp_reader)
        frame_counts.append(actual_frames)
        print(f"Video {os.path.basename(video_path)}: {actual_frames} frames")
    
    # 獲取最小幀數
    min_frames = min(frame_counts)
    print(f"Minimum frames across all videos: {min_frames}")
    
    if min_frames == 0:
        raise ValueError("No valid frames found in videos")
    
    # 重新打開readers
    readers = []
    for video_path in video_paths:
        reader = imageio.get_reader(video_path)
        readers.append(reader)
    
    # 獲取第一個影片的幀尺寸作為參考
    first_frame = readers[0].get_data(0)
    frame_height, frame_width = first_frame.shape[:2]
    print(f"Frame size: {frame_width}x{frame_height}")
    
    # 創建輸出影片寫入器
    # 輸出寬度為三個影片寬度之和，高度保持不變
    output_width = frame_width * 3
    writer = imageio.get_writer(output_path, fps=fps)
    
    try:
        print(f"Concatenating {min_frames} frames...")
        for frame_idx in tqdm(range(min_frames)):
            # 從每個影片讀取對應幀
            frames = []
            for reader in readers:
                frame = reader.get_data(frame_idx)
                # 確保所有幀的高度一致
                if frame.shape[0] != frame_height:
                    # 如果高度不一致，調整到統一高度
                    frame = np.resize(frame, (frame_height, frame_width, frame.shape[2]))
                frames.append(frame)
            
            # 水平連接三個幀
            concatenated_frame = np.concatenate(frames, axis=1)
            
            # 寫入輸出影片
            writer.append_data(concatenated_frame)
    
    finally:
        # 關閉所有讀取器和寫入器
        for reader in readers:
            reader.close()
        writer.close()
    
    print(f"Concatenated video saved to: {output_path}")

def concat_videos_horizontally_cv2(video_paths, output_path, fps=20):
    # 打開影片
    caps = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        caps.append(cap)
    
    # 獲取影片屬性
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    min_frames = min(frame_counts)
    print(f"Frame counts: {frame_counts}")
    print(f"Minimum frames: {min_frames}")
    
    # 獲取第一幀來確定尺寸
    ret, first_frame = caps[0].read()
    if not ret:
        raise ValueError("Cannot read first frame")
    
    height, width = first_frame.shape[:2]
    caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一幀
    
    # 創建輸出影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * len(caps), height))
    
    try:
        for frame_idx in tqdm(range(min_frames)):
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            
            
            concatenated = np.concatenate(frames, axis=1)
            out.write(concatenated)
                
    finally:
        for cap in caps:
            cap.release()
        out.release()
    
    print(f"Video saved to: {output_path}")
    return 


def v2gif(video_path, gif_path, fps=20):
    print(f"Converting video {video_path} to GIF {gif_path} at {fps} FPS")
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']

    # 擷取幀並儲存為 GIF（可調整幀率）
    frames = [frame for frame in reader]
    imageio.mimsave(gif_path, frames, fps=fps)
    return


def rotate(mesh_path, vertical_angle, horizontal_angle, output_mesh_path=None):
    """
    旋轉mesh模型

    Args:
        mesh_path: 輸入mesh文件路徑 (可以是文件路徑或已載入的mesh物件)
        vertical_angle: 垂直角度 (弧度)
        horizontal_angle: 水平角度 (弧度)
        output_mesh_path: 輸出mesh文件路徑 (如果為None則不保存)
        
    Returns:
        rotated_mesh: 旋轉後的mesh物件
        rotation_matrix: 4x4旋轉矩陣
    """
    # 載入或處理mesh
    if isinstance(mesh_path, str):
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
    else:
        # 假設傳入的是mesh物件
        mesh = mesh_path
    
    # 創建旋轉矩陣
    # 先水平旋轉 (繞Y軸)，再垂直旋轉 (繞X軸)
    rotation_y = trimesh.transformations.rotation_matrix(horizontal_angle, [0, 1, 0])
    rotation_x = trimesh.transformations.rotation_matrix(vertical_angle, [1, 0, 0])
    rotation_matrix = np.dot(rotation_x, rotation_y)
    
    # 應用旋轉到mesh
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)

    # 如果指定了輸出路徑，保存旋轉後的mesh
    if output_mesh_path:
        rotated_mesh.export(output_mesh_path)
        print(f"旋轉後的mesh已保存到: {output_mesh_path}")

    print(f"旋轉角度 - 垂直: {np.degrees(vertical_angle):.1f}°, 水平: {np.degrees(horizontal_angle):.1f}°")
    
    return rotated_mesh, rotation_matrix


def render(mesh, resolution=1024, r=2.0, fov=40):
    """
    渲染mesh物件
    
    Args:
        mesh: trimesh mesh物件
        resolution: 渲染解析度
        r: 相機距離倍數
        fov: 視野角度 (度)
        
    Returns:
        rendered_image: 渲染出的圖片 (numpy array)
    """
    # 渲染圖片
    vertices = mesh.vertices
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32).cuda().contiguous()
    faces_tensor = torch.tensor(mesh.faces, dtype=torch.int32).cuda().contiguous()
    
    # 處理紋理和UV坐標
    has_texture = (hasattr(mesh.visual, 'material') and mesh.visual.material is not None and
                   hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and 
                   hasattr(mesh.visual.material, 'baseColorTexture'))
    
    if has_texture:
        # 處理UV坐標
        uv_coords_raw = mesh.visual.uv
        uv_coords_processed = uv_coords_raw.copy()
        
        # 確保UV坐標在[0,1]範圍內
        if uv_coords_processed.min() < 0 or uv_coords_processed.max() > 1:
            uv_coords_processed[:, 0] = (uv_coords_processed[:, 0] - uv_coords_processed[:, 0].min()) / (uv_coords_processed[:, 0].max() - uv_coords_processed[:, 0].min())
            uv_coords_processed[:, 1] = (uv_coords_processed[:, 1] - uv_coords_processed[:, 1].min()) / (uv_coords_processed[:, 1].max() - uv_coords_processed[:, 1].min())
        
        # 翻轉V坐標
        uv_coords_processed[:, 1] = 1.0 - uv_coords_processed[:, 1]
        uv_coords = torch.tensor(uv_coords_processed, dtype=torch.float32).cuda().contiguous()
        
        # 處理紋理圖像
        texture_image = mesh.visual.material.baseColorTexture
        if hasattr(texture_image, 'size'):  # PIL Image
            texture_array = np.array(texture_image)
        else:
            texture_array = texture_image
            
        # 轉換紋理為torch tensor
        if len(texture_array.shape) == 3:
            if texture_array.shape[2] == 4:  # RGBA
                texture_array = texture_array[:, :, :3]  # 只取RGB
            elif texture_array.shape[2] == 1:  # 灰度
                texture_array = np.repeat(texture_array, 3, axis=2)
                
            texture_tensor = torch.tensor(texture_array, dtype=torch.float32).cuda() / 255.0
            texture_tensor = texture_tensor.unsqueeze(0).contiguous()
        else:
            texture_tensor = torch.tensor(texture_array, dtype=torch.float32).cuda() / 255.0
            texture_tensor = texture_tensor.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0).contiguous()
            
    else:
        # 沒有紋理，使用法向量設置顏色
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            vertex_normals = mesh.vertex_normals
        else:
            # 計算頂點法向量
            face_normals = mesh.face_normals
            vertex_normals = np.zeros_like(vertices)
            
            for i, face in enumerate(mesh.faces):
                for vertex_idx in face:
                    vertex_normals[vertex_idx] += face_normals[i]
            
            norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vertex_normals = vertex_normals / norms
        
        # 將法向量轉換為顏色
        vertex_colors = (vertex_normals + 1.0) / 2.0
        vertex_colors = torch.tensor(vertex_colors, dtype=torch.float32).cuda().contiguous()
        
        uv_coords = torch.zeros((len(vertices), 2), dtype=torch.float32).cuda().contiguous()
        texture_tensor = torch.ones((1, 1, 1, 3), dtype=torch.float32).cuda().contiguous()
    
    # 創建渲染上下文
    try:
        glctx = dr.RasterizeCudaContext()
    except:
        glctx = dr.RasterizeGLContext()
        
    # 計算場景邊界盒
    bbox_min = vertices_tensor.min(dim=0)[0]
    bbox_max = vertices_tensor.max(dim=0)[0]
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = (bbox_max - bbox_min).max()
    
    # 設置相機位置 (正面視角)
    yaw = 0.0
    pitch = 0.25  # 輕微俯視角度
    
    orig = torch.tensor([
        np.sin(yaw) * np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw) * np.cos(pitch),
    ], dtype=torch.float32).cuda() * r
    
    cam_pos = bbox_center + orig * bbox_size
    target = bbox_center
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
    
    # 計算視圖矩陣
    forward = torch.nn.functional.normalize(target - cam_pos, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    up_corrected = torch.cross(right, forward, dim=0)
    
    view_matrix = torch.eye(4, dtype=torch.float32).cuda()
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up_corrected
    view_matrix[2, :3] = -forward
    view_matrix[0, 3] = -torch.dot(right, cam_pos)
    view_matrix[1, 3] = -torch.dot(up_corrected, cam_pos)
    view_matrix[2, 3] = torch.dot(forward, cam_pos)
    
    # 投影矩陣
    fov_rad = fov * np.pi / 180.0
    aspect = 1.0
    near = bbox_size * 0.1
    far = bbox_size * 10.0
    
    f = 1.0 / np.tan(fov_rad / 2.0)
    proj_matrix = torch.zeros((4, 4), dtype=torch.float32).cuda()
    proj_matrix[0, 0] = f / aspect
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0
    
    # MVP矩陣
    mvp = proj_matrix @ view_matrix
    
    # 變換頂點
    vertices_homo = torch.cat([vertices_tensor, torch.ones((len(vertices_tensor), 1), device=vertices_tensor.device)], dim=1)
    vertices_clip = (mvp @ vertices_homo.T).T.contiguous()
    
    # 光柵化
    rast, _ = dr.rasterize(glctx, vertices_clip.unsqueeze(0), faces_tensor, resolution=(resolution, resolution))
    
    if has_texture:
        # 使用紋理
        uv_interp, _ = dr.interpolate(uv_coords.unsqueeze(0), rast, faces_tensor)
        color = dr.texture(texture_tensor, uv_interp, filter_mode='linear')
    else:
        # 使用法向量顏色
        color, _ = dr.interpolate(vertex_colors.unsqueeze(0), rast, faces_tensor)
    
    # 轉換為圖像
    color = color[0].cpu().numpy()
    mask = rast[0, :, :, 3].cpu().numpy() > 0
    
    # 創建純黑背景
    final_image = np.zeros((resolution, resolution, 3), dtype=np.float32)
    final_image[mask] = color[mask]
    
    # 轉換為uint8
    final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)
    # 翻轉Y軸
    final_image = np.flipud(final_image)
    
    return Image.fromarray(final_image)


def rotate_and_render_glb(glb_path, vertical_angle, horizontal_angle, output_glb_path=None, resolution=1024, r=2.0, fov=40):
    """
    旋轉GLB模型並渲染圖片 (保持向後兼容性的包裝函數)
    
    Args:
        glb_path: 輸入GLB文件路徑
        vertical_angle: 垂直角度 (弧度)
        horizontal_angle: 水平角度 (弧度)
        output_glb_path: 輸出GLB文件路徑 (如果為None則不保存)
        resolution: 渲染解析度
        r: 相機距離倍數
        fov: 視野角度 (度)
        
    Returns:
        rotated_mesh: 旋轉後的mesh物件
        rotation_matrix: 4x4旋轉矩陣
        rendered_image: 渲染出的圖片 (numpy array)
    """
    # 先旋轉
    rotated_mesh, rotation_matrix = rotate_glb(glb_path, vertical_angle, horizontal_angle, output_glb_path)
    
    # 再渲染
    rendered_image = render_mesh(rotated_mesh, resolution, r, fov)
    
    return rotated_mesh, rotation_matrix, rendered_image
