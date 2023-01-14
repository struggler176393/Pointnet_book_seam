import open3d as o3d
import numpy as np
'''
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))
'''
 
txt_path1 = '/home/lin/CV_AI_learning/Pointnet_Pointnet2_pytorch-master/results/partseg/label_txt/0_label.txt'
txt_path2 = 'data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/Area_5/office_8/office_8.txt'
# 通过numpy读取txt点云
pcd = np.genfromtxt(txt_path2, delimiter=" ")
 
pcd_vector = o3d.geometry.PointCloud()
# 加载点坐标
pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
pcd_vector.colors = o3d.utility.Vector3dVector(pcd[:, 3:6]/255)

# colors = np.random.randint(255, size=(50,3))/255
# pcd_vector.colors = o3d.utility.Vector3dVector(colors[list(map(int,pcd[:, 6])),:])
o3d.visualization.draw_geometries([pcd_vector])




