import os
import torch
import random
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import time

# 定义 reshape_transform 函数
def reshape_transform(tensor, height=37, width=37):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]
    return result

def check_file_complete(file_path, min_size=1024):
    """检查文件是否完整（存在且大小合理）"""
    if not os.path.exists(file_path):
        return False
    try:
        return os.path.getsize(file_path) >= min_size
    except:
        return False

def process_image(source_path, target_path, model, cam, transform):
    """处理单张图片"""
    try:
        # 加载并预处理图像
        image = Image.open(source_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        
        # 使用 Grad-CAM 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # 准备原始图像（调整大小并归一化）
        original_img = np.array(image.resize((518, 518))) / 255.0
        
        # 生成热力图叠加图像
        heatmap = show_cam_on_image(original_img, grayscale_cam, use_rgb=False)
        
        # 保存结果
        cv2.imwrite(target_path, heatmap)
        return True
    except Exception as e:
        print(f"    处理失败: {e}")
        return False

# 设置随机种子
seed = 14
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((518, 518)),  # 确保输入图像大小为 518x518
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载模型
print("正在加载模型...")
start_time = time.time()
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False)
local_weights_path = '/public/home/houkaiji/flower/university/hkj/vit_base_patch14_dinov2.lvd142m/real95.14/weights_e10_0.9514.pth'
model.load_state_dict(torch.load(local_weights_path, map_location='cpu'), strict=False)
model.eval()
print(f"模型加载完成! 用时: {time.time() - start_time:.2f}秒")

# 定义目标层
target_layers = [model.blocks[-1].norm1]

# 创建 Grad-CAM 对象
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

# 要处理的文件夹列表
folders_to_process = [
    # ("gallery_drone", "heatmap_gallery_drone"),
    # ("query_drone", "heatmap_query_drone"),
    # ("gallery_satellite", "heatmap_gallery_satellite"),
    # ("query_satellite", "heatmap_query_satellite")
    # ("query_satellite_mirrortwo20","heatmap_drone_mirrortwo20"),
    # ("query_satellite_mirrortwo40","heatmap_drone_mirrortwo40"),
    # ("query_satellite_mirrortwo60","heatmap_drone_mirrortwo60"),
    # ("query_satellite_mirrortwo80","heatmap_drone_mirrortwo80"),
    # ("query_satellite_mirrortwo100","heatmap_drone_mirrortwo100")
]

# 基础路径
base_source_dir = "/public/home/houkaiji/Sample4Geo/data/query_drone_mirrorLeft"
base_target_dir = "/public/home/houkaiji/flower"

# 确保基础目标目录存在
os.makedirs(base_target_dir, exist_ok=True)

# 总统计
total_processed = 0
total_skipped = 0
total_failed = 0

# 处理每个文件夹类型
for source_subdir, target_subdir in folders_to_process:
    source_dir = os.path.join(base_source_dir, source_subdir)
    target_dir = os.path.join(base_target_dir, target_subdir)
    
    if not os.path.exists(source_dir):
        print(f"\n⚠️ 警告: 源目录不存在，跳过: {source_dir}")
        continue
    
    print(f"\n{'='*60}")
    print(f"处理: {source_subdir} -> {target_subdir}")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"{'='*60}")
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 当前文件夹统计
    processed_images = 0
    skipped_images = 0
    failed_images = 0
    processed_dirs = 0
    
    # 收集所有需要处理的文件
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        rel_path = os.path.relpath(root, source_dir)
        current_target_dir = os.path.join(target_dir, rel_path)
        os.makedirs(current_target_dir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png', '.jfif', '.bmp')):
                source_image_path = os.path.join(root, file)
                file_name, file_ext = os.path.splitext(file)
                target_image_path = os.path.join(current_target_dir, f"heatmap_{file_name}{file_ext}")
                all_files.append((source_image_path, target_image_path, rel_path))
    
    print(f"找到 {len(all_files)} 个图像文件")
    
    # 处理文件
    start_time_dir = time.time()
    for i, (source_path, target_path, rel_dir) in enumerate(all_files):
        # 检查是否已存在
        if check_file_complete(target_path):
            skipped_images += 1
            continue
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == len(all_files):
            print(f"  进度: {i+1}/{len(all_files)} | "
                  f"处理: {processed_images} | "
                  f"跳过: {skipped_images} | "
                  f"失败: {failed_images}")
        
        # 处理图像
        success = process_image(source_path, target_path, model, cam, transform)
        if success:
            processed_images += 1
            if processed_images % 50 == 0:
                print(f"  ✓ 已处理 {processed_images} 张图片...")
        else:
            failed_images += 1
    
    # 更新总统计
    total_processed += processed_images
    total_skipped += skipped_images
    total_failed += failed_images
    
    # 计算目录数量（基于实际处理的目录）
    if processed_images > 0:
        processed_dirs = len(set([os.path.dirname(t) for s, t, d in all_files 
                                 if not check_file_complete(t)]))
    
    elapsed_time = time.time() - start_time_dir
    
    print(f"\n✅ 完成 {source_subdir}:")
    print(f"   用时: {elapsed_time:.2f}秒")
    print(f"   处理目录: {processed_dirs} 个")
    print(f"   处理图片: {processed_images} 张")
    print(f"   跳过图片: {skipped_images} 张")
    print(f"   失败图片: {failed_images} 张")
    if processed_images > 0:
        print(f"   平均速度: {processed_images/elapsed_time:.2f} 张/秒")

# 最终统计
print(f"\n{'='*60}")
print("🎉 所有处理完成!")
print(f"{'='*60}")
print(f"📊 总统计:")
print(f"   总处理图片: {total_processed} 张")
print(f"   总跳过图片: {total_skipped} 张")
print(f"   总失败图片: {total_failed} 张")
print(f"   总用时: {time.time() - start_time:.2f}秒")
print(f"📁 热力图保存在: {base_target_dir}")
print(f"{'='*60}")