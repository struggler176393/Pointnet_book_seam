from plot3dUtils import Open3dVisualizer
import open3d as o3d
import pybullet as p
import time
from visualize_partseg import PartNormalDataset,Generate_txt_and_3d_img,generate_pointcloud,load_models
import torch
import numpy as np
import matplotlib.pyplot as plt

def save_ply(image_num,npoints):
    num_classes = 2 

    from models.pointnet2_part_seg_msg import get_model as pointnet2
    model1 = pointnet2(num_classes=num_classes,normal_channel=True).eval()
    model_dict = {'PonintNet': [model1,r'./log/part_seg/pointnet2_part_seg_msg/checkpoints']}
    loaded_model = load_models(model_dict)
    for i in range(1,image_num):
        color_image = o3d.io.read_image('image/rgb'+str(i)+'.jpg')
        depth_image = o3d.io.read_image('image/depth'+str(i)+'.png')
        point_cloud = generate_pointcloud(color_image=color_image, depth_image=depth_image)

        TEST_DATASET = PartNormalDataset(point_cloud,npoints=npoints, normal_channel=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
        
        predict = Generate_txt_and_3d_img(num_classes,testDataLoader,loaded_model,visualize = False)
        o3d.io.write_point_cloud('video_ply/ply_'+str(i)+'.ply', predict[0])

def video_visualize(ply_num,dt):
    pcd = []
    for i in range(1,ply_num):
        pcd.append(o3d.io.read_point_cloud('video_ply/ply_'+str(i)+'.ply'))
    open3dVisualizer = Open3dVisualizer()
    for i in range(1,ply_num):
        open3dVisualizer(pcd[i-1].points,pcd[i-1].colors)
        time.sleep(dt)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],window_name="removal_pcd")

def process_ply():
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.15, origin = [0,0,0])

    # # 显示原始点云
    # color_image = o3d.io.read_image('image/rgb2.jpg')
    # depth_image = o3d.io.read_image('image/depth2.png')
    # origin_pcd = generate_pointcloud(color_image=color_image, depth_image=depth_image)
    # o3d.visualization.draw_geometries([origin_pcd],window_name="origin_pcd")

    # 显示分割点云
    segmented_pcd = o3d.io.read_point_cloud('video_ply/ply_1.ply')
    colors = np.array(segmented_pcd.colors)
    points = np.array(segmented_pcd.points)
    # o3d.visualization.draw_geometries([segmented_pcd],window_name="segmented_pcd")

    # 分离出目标点云
    seam_index = colors[:,0]==1
    background_index = colors[:,0]!=1
    seam_pcd = o3d.geometry.PointCloud()
    seam_pcd.points = o3d.utility.Vector3dVector(points[seam_index])
    seam_pcd.colors = o3d.utility.Vector3dVector(colors[seam_index])
    seam_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([seam_pcd],window_name="seam_pcd")

    # 基于统计方式剔除离群点
    inlier_pcd, ind = seam_pcd.remove_statistical_outlier(nb_neighbors=15,std_ratio=3)
    # display_inlier_outlier(seam_pcd, ind)
    # o3d.visualization.draw_geometries([inlier_pcd],window_name="inlier_pcd")

    # DBSCAN 聚类
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(inlier_pcd.cluster_dbscan(eps=0.03, min_points=10, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    segment_colors = np.array([[0,0,1],[0,1,0],[1,0,0],[1,1,0]])
    # colors_map = plt.get_cmap("gist_rainbow")(labels / (max_label if max_label > 0 else 1))
    colors_map = segment_colors[labels]
    print(colors_map.shape)
    colors_map[labels < 0] = 0
    # print(labels / (max_label if max_label > 0 else 1))
    # print(colors_map.shape)
    clustered_pcd = o3d.geometry.PointCloud()
    clustered_pcd.points = o3d.utility.Vector3dVector(np.vstack([np.array(inlier_pcd.points),points[background_index]]))
    clustered_pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors_map[:, :3],colors[background_index]]))
    o3d.visualization.draw_geometries([clustered_pcd],window_name="clustered_pcd")
    o3d.io.write_point_cloud('video_ply/processed_ply_1.pcd', clustered_pcd)

    # index = labels == 3
    # colors_map = plt.get_cmap("gist_rainbow")(labels / (max_label if max_label > 0 else 1))
    # colors_map[labels < 0] = 0
    # clustered_points = np.array(inlier_pcd.points)[index]
    # clustered_pcd = o3d.geometry.PointCloud()
    # clustered_pcd.points = o3d.utility.Vector3dVector(clustered_points)
    # clustered_pcd.colors = o3d.utility.Vector3dVector(colors_map[index, :3])
    # o3d.visualization.draw_geometries([clustered_pcd],window_name="clustered_pcd")


    # # 平面分割
    # segmented_plane_pcd = o3d.geometry.PointCloud()
    # segmented_plane_pcd.points = clustered_pcd.points
    # segmented_plane_pcd.colors = clustered_pcd.colors

    # plane_model, inliers = segmented_plane_pcd.segment_plane(distance_threshold=0.01,
    #                                         ransac_n=3,
    #                                         num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # inlier_cloud = segmented_plane_pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = segmented_plane_pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud,coord_mesh])

if __name__ =='__main__':
    image_num = 2
    ply_num = image_num
    npoints = 60000
    dt = 0.5
    # 存储点云并制作视频
    # save_ply(image_num,npoints)
    # video_visualize(ply_num,dt)
    process_ply()



#---------------------------------------------------------------------------#
    '''
    接下来的工作：
    1、弄清楚点云的位姿以及转换       OK
    2、找个好点的位姿录视频       OK
    3、在训练过程中注重对数据集进行平移旋转裁剪，修改相应的代码
    4、对分割得到的点云进行处理
    5、pybullet中加入机械臂进行仿真
    6、在gazebo中搭建环境，机械臂末端添加涂胶头，添加涂胶平板，进行仿真涂胶操作
    '''