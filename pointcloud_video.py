import open3d as o3d
import pybullet as p
import time
from visualize_partseg import PartNormalDataset,Generate_txt_and_3d_img,generate_pointcloud,load_models,Open3dVisualizer
import torch
import numpy as np
import video_utils


if __name__ =='__main__':
    # image_num = 2
    # ply_num = image_num
    # npoints = 60000
    # dt = 0.5
    # 存储点云并制作视频
    # save_ply(image_num,npoints)
    # video_visualize(ply_num,dt)
    # process_ply()

    video_utils.setting()
    labeled_board_address = r'data/dataset_generator/label_urdf/urdf/label_urdf.urdf'
    origin_board_address = r'data/dataset_generator/plane_urdf/urdf/plane_urdf.urdf'
    body_uid_now = []
    num_classes = 2

    # 随机选取相机位姿
    pos_num = 1
    pos_list = video_utils.generate_camera_pos(pos_num)

    # 加载无标签壁板的模型
    body_uid_now = video_utils.load_body(labeled_board_address, origin_body = body_uid_now)
    
    i = 1
    for pos in pos_list:
        p.stepSimulation()
        
        cameraPos = [pos[0], pos[1], pos[2]]
        targetPos = [pos[0], 0, pos[2]]
        cameraupPos = [pos[3], pos[4], pos[5]]

        color_image, depth_image, point_cloud = video_utils.capture(cameraPos,targetPos,cameraupPos)
        a = np.array(point_cloud.points)
        o3d.visualization.draw_geometries([point_cloud])
        
        TEST_DATASET = PartNormalDataset(point_cloud,npoints=30000, normal_channel=True)
        testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
        predict_pcd = Generate_txt_and_3d_img(num_classes,testDataLoader,load_models(),visualize = True)



        # dataset_utils.view_image(cameraPos,targetPos,cameraupPos)
        # video_utils.save_origin_RGBimage(i,cameraPos,targetPos,cameraupPos)
        time.sleep(1)
        i = i + 1


