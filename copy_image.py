# import os
# import shutil

# # 定义源文件夹路径和目标文件夹路径
# source_folder = "/public/home/chuyanhao/Sample4Geo/data/U1652/test/gallery_satellite"
# destination_folder = "/public/home/chuyanhao/Sample4Geo/data/test/query_drone"

# # 确保目标文件夹存在
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # 遍历源文件夹中的所有子文件夹
# for folder_name in os.listdir(source_folder):
#     folder_path = os.path.join(source_folder, folder_name)
#     destination_folder_path = os.path.join(destination_folder, folder_name)
    
#     if not os.path.exists(destination_folder_path):
#         continue
#     # os.makedirs(destination_folder_path)

#     # 确保是文件夹
#     if os.path.isdir(folder_path):
#         # 遍历需要复制的文件
#         jpg_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
        
#         for jpg_file in jpg_files:
#             source_image_path = os.path.join(folder_path, jpg_file)
#             destination_image_path = os.path.join(destination_folder_path, jpg_file)
            
#             # 检查文件是否存在
#             if os.path.exists(source_image_path):
#                 # 复制文件
#                 shutil.copy2(source_image_path, destination_image_path)
#                 print(f"Copied {source_image_path} to {destination_image_path}")
#             else:
#                 print(f"File {source_image_path} does not exist.")


import os
import shutil

# 源目录和目标目录
source_dir = '/public/home/chuyanhao/Sample4Geo/data/test/query_drone'
target_dir = '/public/home/chuyanhao/Sample4Geo/data/test/satellite'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 遍历源目录及其子目录
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.jpg'):
            # 构建完整的文件路径
            file_path = os.path.join(root, file)
            # 构建目标路径
            target_path = os.path.join(target_dir, file)
            # 复制文件
            shutil.copy(file_path, target_path)