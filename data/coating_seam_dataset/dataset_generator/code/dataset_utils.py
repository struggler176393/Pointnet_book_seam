import pybullet as p
import numpy as np
import pybullet_data
import cv2
import PIL.Image as Image
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import time

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
def save_origin_RGBimage(i,cameraPos,targetPos,cameraupPos,width=1280,height=720,fov=50,near=0.01,far=5,img_path = 'data/coating_seam_dataset/dataset_generator/origin_rgb/'):

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
    cv2.imwrite(img_path+str(i)+'.jpg',rgbImg)






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
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
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
    return color_image, depth_image, point_cloud


# 从相机获取图像生成点云并保存标签后的数据集txt文件
# save_color:保存标注后模型的pcd文件
# save_label:保存标签后的数据集txt文件
def generate_pcd_save_labeled_txt(i,cameraPos,targetPos,cameraupPos,save_color=True,save_label=True,
pcd_path = 'data/coating_seam_dataset/dataset_generator/labeled_colorpcd/',
txt_path = 'data/coating_seam_dataset/dataset_generator/labeled_txt/'):
    _, _, point_cloud = capture(cameraPos,targetPos,cameraupPos)
    if save_color:
        o3d.io.write_point_cloud(pcd_path+str(i)+'.pcd', point_cloud)
    if save_label:
        save_txt(point_cloud,txt_path+str(i)+".txt")

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


if __name__ == '__main__':
    # visualize_ply(r"labeled_colorpcd\1.pcd")
    visualize_txt(r"labeled_txt\1.txt")