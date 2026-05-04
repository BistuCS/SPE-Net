import torch
import numpy as np
from tqdm import tqdm
import gc
from ..trainer import predict

import os
import matplotlib.pyplot as plt
# from openTSNE import TSNE
# from sklearn.manifold import TSNE
from torchvision import transforms
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import normalize
# import umap

# ==================== 可视化引擎 ====================
# class TSNEVisualizer:
#     def __init__(self, features):
#         self.features = features
        
#     def visualize(self):
#         # 数据标准化
#         scaler = StandardScaler()
#         features_norm = scaler.fit_transform(self.features.cpu())
        
#         # tsne = TSNE(
#         #     n_components=2,
#         #     perplexity=25,
#         #     # metric="cosine",  # 使用余弦距离
#         #     learning_rate=150,
#         #     n_iter=1000,
#         #     n_jobs=8,  # 利用多核加速
#         #     random_state=42,
#         # )
#         # embeddings = tsne.fit(features_norm)

#         tsne = TSNE(
#             n_components=2,
#             perplexity=10,
#             learning_rate=250,
#             n_iter=2000,
#             random_state=42
#         )
#         embeddings = tsne.fit_transform(features_norm)

#         # # 创建UMAP对象
#         # umap_reducer = umap.UMAP(
#         #     n_neighbors=2,
#         #     min_dist=1.5,
#         #     spread=1.5,      # 嵌入点的有效尺度
#         #     n_components=2,  # 降维后的维度数
#         #     metric='cosine',  # 距离度量方式
#         #     n_epochs=200,    # 训练轮数
#         #     learning_rate=1.0)  # 学习率
#         # # 训练UMAP模型
#         # embeddings = umap_reducer.fit_transform(features_norm)

#         # for i in range(0, len(embeddings), 4):
#         #     print(embeddings[i][0],embeddings[i][1])
#         #     print(embeddings[i+1][0],embeddings[i+1][1])
#         #     print(embeddings[i+2][0],embeddings[i+2][1])
#         #     print(embeddings[i+3][0],embeddings[i+3][1])
            
#         #     # print()
#         n_classes = len(features_norm) // 4
#         colors = np.repeat(np.arange(n_classes), 4)
#         # 创建可视化
#         plt.figure(figsize=(16, 12))
#         scatter = plt.scatter(
#             embeddings[:, 0], 
#             embeddings[:, 1],
#             c=colors,
#             cmap='viridis',
#             alpha=0.8,
#             s=20,
#             edgecolor='k',
#             linewidth=0.2
#         )
        
#         plt.axis('off')
#         plt.savefig('drone_only_tsne_eval.png', dpi=650, bbox_inches='tight')

def evaluate(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True):
    
    
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
    # print("1111111111111")
    # print(img_features_query.shape)
    # visualizer = TSNEVisualizer(img_features_query)
    # visualizer.visualize()

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("img_features_query",img_features_query.shape)
    
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_gallery)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))             
        
    print(' - '.join(string)) 
    
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        #torch.cuda.empty_cache()
    
    return CMC[0]


def eval_query(qf,ql,gf,gl):

    score = gf @ qf.unsqueeze(-1)
    
    score = score.squeeze().cpu().numpy()
 
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]  
    # print("index",index)  

    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index
    # print("good_index",good_index)

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
    
    return ap, cmc




