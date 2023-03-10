import pybullet as p
import numpy as np
import pybullet_data
import cv2
import PIL.Image as Image
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import time
from visualize_partseg import PartNormalDataset,Generate_txt_and_3d_img,generate_pointcloud,load_models,Open3dVisualizer
import torch

# pybullet初始设置
def setting():  
    _ = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=180,
            cameraPitch=-30,
            cameraTargetPosition=[0,0,0])

# 删除pybullet中的物体
def remove_body(body_uid=[]):
    for body in body_uid:
        p.removeBody(body)

# 删除pybullet中的物体
def load_body(file_path, origin_body = []):
    r = R.from_rotvec(np.pi/2 * np.array([-1, 0, 0]))
    if origin_body != []:
        remove_body(origin_body)
    plane_uid = p.loadURDF("plane.urdf", useMaximalCoordinates=True)  # 加载一个地面
    body_uid = [plane_uid]
    object_uid = p.loadURDF(file_path, basePosition=[0, 0, 0],baseOrientation=r.as_quat(),useFixedBase=1)
    body_uid.append(object_uid)
    return body_uid

# 随机加载多组相机位姿
def generate_camera_pos(pos_num):
    # Pos_x范围：[-4.5,0]
    # Pos_y范围：[0.3,0.7]
    # Pos_z范围：[0.1,1.4]
    # Pos_rx范围：[-1,1]
    # Pos_ry范围：[-1,1]
    # Pos_rz范围：[-1,1]
    Pos_x = [-4.5,0]
    Pos_y = [0.3,0.7]
    Pos_z = [0.1,1.4]
    Pos_rx = [-1,1]
    Pos_ry = [-1,1]
    Pos_rz = [-1,1]
    pos_range_list = [Pos_x,Pos_y,Pos_z,Pos_rx,Pos_ry,Pos_rz]
    rand_list = np.random.rand(pos_num,6)
    pos_list = np.zeros([pos_num,6])
    for i in range(pos_num):
        for j in range(6):
            pos_list[i][j] = (pos_range_list[j][1]-pos_range_list[j][0]) * rand_list[i][j] + pos_range_list[j][0]
    return pos_list

# 保存未标注模型的RGB图像
def save_origin_RGBimage(i,cameraPos,targetPos,cameraupPos,width=1280,height=720,fov=50,near=0.01,far=5):

    aspect = width / height

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=cameraupPos,
        physicsClientId=0
    )
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    images = p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # 存储彩色图像
    rgbImg = cv2.cvtColor(images[2], cv2.COLOR_BGR2RGB)      
    cv2.imwrite('origin_rgb/'+str(i)+'.jpg',rgbImg)






# 利用RGB图像和深度图像加载点云
def generate_pointcloud(color_image, depth_image,width=1280,height=720,fov=50,near=0.01,far=5):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault )

    aspect = width / height

    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    intrinsic.set_intrinsics(width=width, height=height, fx=projection_matrix[0]*width/2, fy=projection_matrix[5]*height/2, cx=width/2, cy=height/2)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    return point_cloud

# 将点云保存为标注后的数据集txt文件(7列，3列位置，3列法向，1列标签)
def save_txt(pcd,file_path):
    pcd = pcd
    all_colors = np.array(pcd.colors)
    label_index = (all_colors[:,0]>=0.2) & (all_colors[:,1]<0.2) & (all_colors[:,2]<0.2)
    if label_index.any():
        all_points = np.array(pcd.points)
        all_normals = np.array(pcd.normals)

        
        label_points = all_points[label_index]
        label_normals = all_normals[label_index]
        label_ID = np.ones([label_points.shape[0],1])

        label_data = np.hstack([label_points,label_normals,label_ID])

        background_index = label_index == False
        background_points = all_points[background_index]
        background_normals = all_normals[background_index]
        background_ID = np.zeros([background_points.shape[0],1])
        background_data = np.hstack([background_points,background_normals,background_ID])

        all_data = np.vstack([background_data,label_data])
        np.savetxt(file_path, all_data, delimiter = ' ', fmt = '%.6f')

def capture(cameraPos,targetPos,cameraupPos,width=1280,height=720,fov=50,near=0.01,far=5):
    aspect = width / height

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=cameraupPos,
        physicsClientId=0
    )
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    images = p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # 彩色图像
    rgbImg = cv2.cvtColor(images[2], cv2.COLOR_BGRA2BGR)
    # 深度图像
    depImg = far * near / (far - (far - near) * images[3])
    depImg = np.asanyarray(depImg).astype(np.float32) * 1000.  
    depImg = depImg.astype(np.uint16)

    color_image = o3d.geometry.Image(rgbImg)
    depth_image = o3d.geometry.Image(depImg)
    point_cloud = generate_pointcloud(color_image=color_image, depth_image=depth_image)
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    return color_image, depth_image, point_cloud


# 从相机获取图像生成点云并保存标签后的数据集txt文件
# save_color:保存标注后模型的pcd文件
# save_label:保存标签后的数据集txt文件
def generate_pcd_save_labeled_txt(i,cameraPos,targetPos,cameraupPos,save_color=True,save_label=True):
    _, _, point_cloud = capture(cameraPos,targetPos,cameraupPos)
    if save_color:
        o3d.io.write_point_cloud('labeled_colorpcd/'+str(i)+'.pcd', point_cloud)
    if save_label:
        save_txt(point_cloud,"labeled_txt/"+str(i)+".txt")

# 在pybullet中显示相机拍摄的图像
def view_image(cameraPos,targetPos,cameraupPos,width=1280,height=720,fov=50,near=0.01,far=5):
    aspect = width / height

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=cameraupPos,
        physicsClientId=0
    )
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    p.getCameraImage(width, height, viewMatrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

# txt文件open3d可视化
def visualize_txt(file_path):
    data = np.loadtxt(file_path).astype(float)

    color_map = np.array([[0,0,1],[1,0,0]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,0:3])
    pcd.colors = o3d.utility.Vector3dVector(color_map[data[:,6].astype(int)])
    o3d.visualization.draw_geometries([pcd])

def visualize_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])


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
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
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


if __name__ == '__main__':
    # visualize_ply(r"labeled_colorpcd\1.pcd")
    visualize_txt(r"labeled_txt\1.txt")