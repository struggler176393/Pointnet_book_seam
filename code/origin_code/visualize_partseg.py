import tqdm
import matplotlib
import torch
import os
import warnings
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import pybullet as p

import time

warnings.filterwarnings('ignore')
matplotlib.use("Agg")
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc,centroid,m

def generate_pointcloud(color_image, depth_image,width=1280,height=720,fov=50,near=0.01,far=5):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault )

    aspect = width / height

    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    intrinsic.set_intrinsics(width=width, height=height, fx=projection_matrix[0]*width/2, fy=projection_matrix[5]*height/2, cx=width/2, cy=height/2)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    point_cloud.estimate_normals( search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    return point_cloud

class PartNormalDataset(Dataset):
    def __init__(self, point_cloud, npoints=2500, normal_channel=False):
        self.npoints = npoints # 采样点数
        self.cat = {}
        self.normal_channel = normal_channel # 是否使用法向信息

        position_data = np.asarray(point_cloud.points)
        normal_data = np.asarray(point_cloud.normals)
        self.raw_pcd = np.hstack([position_data,normal_data]).astype(np.float32)

        self.cat = {'board':'12345678'}
        # 输出的是元组，('Airplane',123.txt)

        self.classes = {'board': 0} 

        data = self.raw_pcd

        if not self.normal_channel:  # 判断是否使用法向信息
            self.point_set = data[:, 0:3]
        else:
            self.point_set = data[:, 0:6]

        self.point_set[:, 0:3],self.centroid,self.m = pc_normalize(self.point_set[:, 0:3]) # 做一个归一化

        choice = np.random.choice(self.point_set.shape[0], self.npoints, replace=True) # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        # resample
        self.point_set =  self.point_set[choice, :] # 根据索引采样

    def __getitem__(self, index):

        cat = list(self.cat.keys())[0]
        cls = self.classes[cat] # 将类名转换为索引
        cls = np.array([cls]).astype(np.int32)

        return self.point_set, cls, self.centroid, self.m # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return 1



class Generate_txt_and_3d_img:
    def __init__(self,num_classes,testDataLoader,model,visualize = False):
        self.testDataLoader = testDataLoader
        self.num_classes = num_classes
        self.heat_map = False # 控制是否输出heatmap
        self.visualize = visualize # 是否open3d可视化
        self.model = model

        self.generate_predict()
        self.o3d_draw_3d_img()

    def __getitem__(self, index):
        return self.predict_pcd_colored

    def generate_predict(self):

        for _, (points, label,centroid,m) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                                      total=len(self.testDataLoader),smoothing=0.9):

            #点云数据、整个图像的标签、每个点的标签、  没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])
            points = points.transpose(2, 1)
            #print('1',target.shape) # 1 torch.Size([1, 2048])
            xyz_feature_point = points[:, :6, :]

            model = self.model

            seg_pred, _ = model(points, self.to_categorical(label, 1))
            seg_pred = seg_pred.cpu().data.numpy()

            if self.heat_map:
                out =  np.asarray(np.sum(seg_pred,axis=2))
                seg_pred = ((out - np.min(out) / (np.max(out) - np.min(out))))
            else:
                seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c

            seg_pred = np.concatenate([np.asarray(xyz_feature_point), seg_pred[:, None, :]],
                    axis=1).transpose((0, 2, 1)).squeeze(0) 

            self.predict_pcd = seg_pred
            self.centroid = centroid
            self.m = m


    def o3d_draw_3d_img(self):

        pcd = self.predict_pcd
        pcd_vector = o3d.geometry.PointCloud()
        # 加载点坐标
        pcd_vector.points = o3d.utility.Vector3dVector(self.m * pcd[:, :3] + self.centroid)
        # colors = np.random.randint(255, size=(2,3))/255
        colors = np.array([[0.8, 0.8, 0.8],[1,0,0]])
        pcd_vector.colors = o3d.utility.Vector3dVector(colors[list(map(int,pcd[:, 6])),:])

        if self.visualize:
            coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1, origin = [0,0,0])
            o3d.visualization.draw_geometries([pcd_vector,coord_mesh])
        self.predict_pcd_colored = pcd_vector

    def to_categorical(self,y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y

def load_models(model_dict):
    model = list(model_dict.values())[0][0]
    checkpoints_dir = list(model_dict.values())[0][1]
    weight_dict = torch.load(os.path.join(checkpoints_dir,'best_model.pth'))
    model.load_state_dict(weight_dict['model_state_dict'])
    return model

if __name__ =='__main__':
    
    num_classes = 2 # 填写数据集的类别数 如果是s3dis这里就填13   shapenet这里就填50

    from models.pointnet2_part_seg_msg import get_model as pointnet2
    model1 = pointnet2(num_classes=num_classes,normal_channel=True).eval()

    
    color_image = o3d.io.read_image('image/rgb1.jpg')
    depth_image = o3d.io.read_image('image/depth1.png')
    
    point_cloud = generate_pointcloud(color_image=color_image, depth_image=depth_image)

    TEST_DATASET = PartNormalDataset(point_cloud,npoints=30000, normal_channel=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,drop_last=True)
    
    model_dict = {'PonintNet': [model1,r'./log/part_seg/pointnet2_part_seg_msg/checkpoints']}
    loaded_model = load_models(model_dict)
    predict_pcd = Generate_txt_and_3d_img(num_classes,testDataLoader,loaded_model,visualize = True)


    # o3d.visualization.draw_geometries([inv_pcd])


