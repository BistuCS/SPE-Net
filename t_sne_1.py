import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from openTSNE import TSNE
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import umap

# ==================== 配置文件 ====================
class Config:
    # 路径配置
    data_root = "/public/home/chuyanhao/Sample4Geo/data/"
    model_path = "/public/home/chuyanhao/Sample4Geo/university/1652_vit_base_patch14_dinov2.lvd142m/111612/weights_e10_0.9501.pth"
    
    # 数据参数
    target_modality = 'gallery_satellite'  # 只处理无人机视角
    test_split = "test"  
    sample_per_class = 1  # 每个地点采样数量
    
    # 模型参数
    target_layer = 'blocks[-1]'  
    img_size = 518
    
    # t-SNE参数
    perplexity = 1
    learning_rate = 150
    n_iter = 2000

# ==================== 数据集加载 ====================
class DroneOnlyDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = transforms.Compose([
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 收集无人机图像路径
        self.samples = []
        self.class_ids = []
        
        modality_path = os.path.join(
            self.cfg.data_root, 
            self.cfg.test_split,
            self.cfg.target_modality
        )
        
        # 遍历所有地点类别
        for place_id in sorted(os.listdir(modality_path)):
            place_dir = os.path.join(modality_path, place_id)
            if not os.path.isdir(place_dir):
                continue
            
            # 获取所有无人机图像
            drone_images = [f for f in os.listdir(place_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 随机采样
            selected = np.random.choice(
                drone_images,
                size=min(self.cfg.sample_per_class, len(drone_images)),
                replace=False
            )
            
            for img_name in selected:
                img_path = os.path.join(place_dir, img_name)
                self.samples.append(img_path)
                self.class_ids.append(int(place_id))  # 地点ID作为类别标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        return self.transform(img)
        

# ==================== 特征提取器 ====================
class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_model()
        
    def _init_model(self):
        """初始化DINOv2模型"""
        self.model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m', 
            pretrained=False, 
            num_classes=0  
        )
        
        state_dict = torch.load(self.cfg.model_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device).eval()
        
        # 注册特征钩子
        self.features = {}
        layer = eval(f'self.model.{self.cfg.target_layer}')
        layer.register_forward_hook(lambda m, i, o: self.features.update({'embedding': o[:, 0].detach()}))
    
    def extract_features(self, dataloader):
        """批量提取特征"""
        all_features = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                _ = self.model(batch)
                feat = self.features['embedding'].cpu().numpy()
                all_features.append(feat)
        return np.concatenate(all_features, axis=0)

# ==================== 可视化引擎 ====================
class TSNEVisualizer:
    def __init__(self, features, labels, cfg):
        self.features = features
        self.labels = labels
        self.cfg = cfg
        
    def visualize(self):
        # 数据标准化
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(self.features)
        
        # 运行t-SNE
        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=self.cfg.perplexity,
        #     learning_rate=self.cfg.learning_rate,
        #     n_iter=self.cfg.n_iter,
        #     random_state=42
        # )

        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=self.cfg.perplexity,
        #     metric="cosine",  # 使用余弦距离
        #     learning_rate=self.cfg.learning_rate,
        #     n_iter=self.cfg.n_iter,
        #     n_jobs=8,  # 利用多核加速
        #     random_state=42,
        # )
        # embeddings = tsne.fit(features_norm)


        # 创建UMAP对象
        umap_reducer = umap.UMAP(
            n_neighbors=2,
            min_dist=1.5,
            spread=1.5,      # 嵌入点的有效尺度
            n_components=2,  # 降维后的维度数
            metric='cosine',  # 距离度量方式
            n_epochs=200,    # 训练轮数
            learning_rate=1.0)  # 学习率
        # 训练UMAP模型
        embeddings = umap_reducer.fit_transform(features_norm)

        for i in range(0, len(embeddings), 4):
            print(embeddings[i][0],embeddings[i][1])
            print(embeddings[i+1][0],embeddings[i+1][1])
            print(embeddings[i+2][0],embeddings[i+2][1])
            print(embeddings[i+3][0],embeddings[i+3][1])
            
            # print()

        # 创建可视化
        plt.figure(figsize=(16, 12))
        scatter = plt.scatter(
            embeddings[:, 0], 
            embeddings[:, 1],
            c=self.labels,
            cmap='viridis',
            alpha=0.8,
            s=20,
            edgecolor='k',
            linewidth=0.2
        )
        
        # 添加样式
        plt.title('Drone View Feature Distribution', fontsize=14, pad=20)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(alpha=0.2)
        
        # 保存并显示
        plt.savefig('drone_only_tsne.png', dpi=350, bbox_inches='tight')
        plt.show()

# ==================== 主流程 ====================
def main():
    cfg = Config()
    
    # 1. 加载无人机数据集
    print("Loading drone images...")
    dataset = DroneOnlyDataset(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0
    )
    
    # 2. 特征提取
    print("Extracting features...")
    extractor = FeatureExtractor(cfg)
    features = extractor.extract_features(dataloader)
    
    print(features.shape)

    # 3. 可视化
    print("Generating visualization...")
    visualizer = TSNEVisualizer(features, dataset.class_ids, cfg)
    visualizer.visualize()

if __name__ == '__main__':
    main()