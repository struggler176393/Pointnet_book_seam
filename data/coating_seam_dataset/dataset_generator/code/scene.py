import pybullet as p
import pybullet_data
import numpy as np
import cv2
import PIL.Image as Image
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import open3d as o3d
import time
import dataset_utils
from shuffle import shuffle_data

if __name__ == '__main__':
    dataset_utils.setting()
    origin_board_address = r'data/coating_seam_dataset/dataset_generator/plane_urdf/urdf/plane_urdf.urdf'
    labeled_board_address = r'data/coating_seam_dataset/dataset_generator/label_urdf/urdf/label_urdf.urdf'
    body_uid_now = []

    # 随机选取相机位姿
    pos_num = 10
    pos_list = dataset_utils.generate_camera_pos(pos_num)

    # # 加载无标签壁板的模型,保存未标注图像
    # body_uid_now = dataset_utils.load_body(origin_board_address, origin_body = body_uid_now)
    
    # i = 1
    # for pos in pos_list:
    #     p.stepSimulation()
        
    #     cameraPos = [pos[0], pos[1], pos[2]]
    #     targetPos = [pos[0], 0, pos[2]]
    #     cameraupPos = [pos[3], pos[4], pos[5]]


    #     # dataset_utils.view_image(cameraPos,targetPos,cameraupPos)
    #     dataset_utils.save_origin_RGBimage(i,cameraPos,targetPos,cameraupPos)
    #     time.sleep(1)
    #     i = i + 1

    # 加载有标签壁板的模型，保存标注后的pcd和txt
    body_uid_now = dataset_utils.load_body(labeled_board_address, origin_body = body_uid_now)

    i = 1
    for pos in pos_list:
        p.stepSimulation()
        
        cameraPos = [pos[0], pos[1], pos[2]]
        targetPos = [pos[0], 0, pos[2]]
        cameraupPos = [pos[3], pos[4], pos[5]]


        # dataset_utils.view_image(cameraPos,targetPos,cameraupPos)
        dataset_utils.generate_pcd_save_labeled_txt(i,cameraPos,targetPos,cameraupPos,
        save_color=False,save_label=True,txt_path = 'data/coating_seam_dataset/12345678/')

        # time.sleep(1)
        i = i + 1
    
    shuffle_data(pos_num)
