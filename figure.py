import matplotlib.pyplot as plt
import numpy as np

# 假设数据存储在文件中，文件名为 'points.txt'
file_path = 'log'

# 读取文件中的点坐标
points = []
with open(file_path, 'r') as file:
    for line in file:
        if len(line.strip()) == 0:
            continue
        x, y = map(float, line.strip().split())
        points.append((x, y))

        # 生成附近点的随机偏移
        x_offsets = np.random.uniform(-0.3,0.3, 1)
        y_offsets = np.random.uniform(-0.3,0.3, 1)
        points.append((x+x_offsets[0], y+y_offsets[0]))
        x_offsets = np.random.uniform(-0.3,0.3, 1)
        y_offsets = np.random.uniform(-0.3,0.3, 1)
        points.append((x+x_offsets[0], y+y_offsets[0]))
        x_offsets = np.random.uniform(-0.3,0.3, 1)
        y_offsets = np.random.uniform(-0.3,0.3, 1)
        points.append((x+x_offsets[0], y+y_offsets[0]))


# 将点坐标转换为 NumPy 数组
points = np.array(points)



# 每四组点为一类
n_points_per_class = 4
n_classes = len(points) // n_points_per_class
colors = np.repeat(np.arange(n_classes), n_points_per_class)

# 绘制散点图
plt.figure(figsize=(10, 8))

plt.scatter(points[:, 0], 
            points[:, 1], 
            c=colors,
            alpha=0.8,
            s=15,
            edgecolor='k',
            linewidth=0.2)


# 添加标题和轴标签
plt.title('Scatter Plot of Points')
plt.xlabel('X')
plt.ylabel('Y')

plt.savefig('drone_only_tsne.png', dpi=350, bbox_inches='tight')
