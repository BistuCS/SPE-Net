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

# 定义 reshape_transform 函数
def reshape_transform(tensor, height=37, width=37):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)  # [batch_size, channels, height, width]
    return result

# 设置输出目录
output_dir = "gradcam_results_sate"
os.makedirs(output_dir, exist_ok=True)

# 定义输入图像路径
# image_path = "/public/home/chuyanhao/Sample4Geo/data/U1652/test/gallery_satellite/0099/0099.jpg"
# image_path = "/public/home/chuyanhao/Sample4Geo/data/U1652/test/query_drone/0015/image-53.jpeg"

# image_path = "/public/home/chuyanhao/Sample4Geo/data/SUES-200-512x512/Testing/150/gallery_satellite/0008/0.png"
image_path = "/public/home/houkaiji/Sample4Geo/data/query_drone_mirrorLeft/query_satellite_mirrortwo40/0788/image-01.jpeg"


# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((518, 518)),  # 确保输入图像大小为 518x518
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试多个随机种子
seed_range = range(100)  # 测试 100 个不同的随机种子
for seed in seed_range:
    print(f"Processing with seed: {seed}")
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 重新加载模型
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False)
    # local_weights_path = '/public/home/chuyanhao/Sample4Geo/university/1652_vit_base_patch14_dinov2.lvd142m/111612/weights_e10_0.9501.pth'
    
    # local_weights_path = '/public/home/chuyanhao/Sample4Geo/university/SUES200_vit_base_patch14_dinov2.lvd142m/150_5/weights_e1_0.9877.pth'
    local_weights_path = '/public/home/houkaiji/flower/university/hkj/vit_base_patch14_dinov2.lvd142m/real95.14/weights_e10_0.9514.pth'
    # local_weights_path = '/public/home/chuyanhao/Sample4Geo/university/SUES200_vit_base_patch14_dinov2.lvd142m/250_5/weights_e5_0.9902.pth'
    

    model.load_state_dict(torch.load(local_weights_path, map_location='cpu'), strict=False)
    model.eval()

    # 定义目标层
    target_layers = [model.blocks[-1].norm1]

    # 创建 Grad-CAM 对象
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # 重新加载并预处理图像
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # 使用 Grad-CAM 生成热力图
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    heatmap = show_cam_on_image(np.array(image.resize((518, 518))) / 255.0, grayscale_cam, use_rgb=False)

    # 保存结果
    output_image_path = os.path.join(output_dir, f"gradcam_seed_{seed}.jpg")
    cv2.imwrite(output_image_path, heatmap)

print(f"All Grad-CAM visualizations saved to {output_dir}")