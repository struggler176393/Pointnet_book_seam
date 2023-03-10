"传入模型权重文件，读取预测点，生成预测的txt文件"

import tqdm
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import warnings
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import time




def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/book_seam_dataset', npoints=2500, class_choice=None, normal_channel=False,path='1.txt'):
        self.npoints = npoints # 采样点数
        self.root = root # 文件根路径
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt') # 类别和文件夹名字对应的路径
        self.cat = {}
        self.normal_channel = normal_channel # 是否使用法向信息


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()} #{'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
        self.classes_original = dict(zip(self.cat, range(len(self.cat)))) #{'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        if not class_choice is  None:  # 选择一些类别进行训练  好像没有使用这个功能
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)
        
        self.datapath = ('book',path)
        # 输出的是元组，('Airplane',123.txt)

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        ## self.classes  将类别的名称和索引对应起来  例如 飞机 <----> 0
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        """
        shapenet 有16 个大类，然后每个大类有一些部件 ，例如飞机 'Airplane': [0, 1, 2, 3] 其中标签为0 1  2 3 的四个小类都属于飞机这个大类
        self.seg_classes 就是将大类和小类对应起来
        """
        self.seg_classes = {'book': [0,1]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):

        fn = self.datapath # 根据索引 拿到训练数据的路径self.datepath是一个元组（类名，路径）
        cat = self.datapath[0] # 拿到类名
        cls = self.classes[cat] # 将类名转换为索引
        cls = np.array([cls]).astype(np.int32)
        # 读取modelnet40
        data = np.loadtxt(fn[1],dtype=np.float32,delimiter=' ') 
        # 读取shapenet
        # data = np.loadtxt(fn[1]).astype(np.float32) # size 20488,7 读入这个txt文件，共20488个点，每个点xyz rgb +小类别的标签
        if not self.normal_channel:  # 判断是否使用法向信息
            point_set = data[:, 0:3]
        else:
            point_set = data[:, 0:6]
        seg = data[:, -1].astype(np.int32) # 拿到小类别的标签
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls, seg)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3]) # 做一个归一化
        choice = np.random.choice(len(seg), self.npoints, replace=False) # 对一个类别中的数据进行随机采样 返回索引，不允许重复采样
        # resample
        point_set =  point_set[choice, :] # 根据索引采样
    
        seg = seg[choice]

        return point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return len(self.datapath)



class Generate_txt_and_3d_img:
    def __init__(self,img_root,target_root,num_classes,testDataLoader,model_dict,path='123.txt'):
        self.img_root = img_root # 点云数据路径
        self.target_root = target_root  # 生成txt标签和预测结果路径
        self.testDataLoader = testDataLoader
        self.num_classes = num_classes
        self.heat_map = False # 控制是否输出heatmap
        self.label_path_txt = os.path.join(self.target_root, 'label_txt') # 存放label的txt文件，指标注
        self.path=path
        self.make_dir(self.label_path_txt)

        # 拿到模型 并加载权重
        self.model_name = []
        self.model = []
        self.model_weight_path = []

        for k,v in model_dict.items():
            self.model_name.append(k)
            self.model.append(v[0])
            self.model_weight_path.append(v[1])

        # 加载权重
        self.load_cheackpoint_for_models(self.model_name,self.model,self.model_weight_path)

        # 创建文件夹
        self.all_pred_image_path = [] # 所有预测结果的路径列表
        self.all_pred_txt_path = [] # 所有预测txt的路径列表
        for n in self.model_name:
            self.make_dir(os.path.join(self.target_root,n+'_predict_txt'))
            self.make_dir(os.path.join(self.target_root, n + '_predict_image'))
            self.all_pred_txt_path.append(os.path.join(self.target_root,n+'_predict_txt'))
            self.all_pred_image_path.append(os.path.join(self.target_root, n + '_predict_image'))
        "将模型对应的预测txt结果和img结果生成出来，对应几个模型就在列表中添加几个元素"

        self.generate_predict_to_txt() 
        self.o3d_draw_3d_img()

    def generate_predict_to_txt(self): # 生成预测txt

        for batch_id, (points, label, target) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                                      total=len(self.testDataLoader),smoothing=0.9):

            #点云数据、整个图像的标签、每个点的标签、  没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])
            points = points.transpose(2, 1)
            #print('1',target.shape) # 1 torch.Size([1, 2048])
            xyz_feature_point = points[:, :6, :]
            # 将标签保存为txt文件
            point_set_without_normal = np.asarray(torch.cat([points.permute(0, 2, 1),target[:,:,None]],dim=-1)).squeeze(0)  # 代标签 没有归一化的点云数据  的numpy形式
            np.savetxt(os.path.join(self.label_path_txt,f'{batch_id}_label.txt'), point_set_without_normal, fmt='%.04f') # 将其存储为txt文件
            " points  torch.Size([16, 2048, 6])  label torch.Size([16, 1])  target torch.Size([16, 2048])"

            assert len(self.model) == len(self.all_pred_txt_path) , '路径与模型数量不匹配，请检查'

            for n,model,pred_path in zip(self.model_name,self.model,self.all_pred_txt_path):

                seg_pred, trans_feat = model(points, self.to_categorical(label, 1))
                seg_pred = seg_pred.cpu().data.numpy()
                #=================================================
                #seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                if self.heat_map:
                    out =  np.asarray(np.sum(seg_pred,axis=2))
                    seg_pred = ((out - np.min(out) / (np.max(out) - np.min(out))))
                else:
                    seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                #=================================================
                seg_pred = np.concatenate([np.asarray(xyz_feature_point), seg_pred[:, None, :]],
                        axis=1).transpose((0, 2, 1)).squeeze(0)  # 将点云与预测结果进行拼接，准备生成txt文件
                save_path = os.path.join(pred_path, f'{n}_{batch_id}.txt')
                np.savetxt(save_path,seg_pred, fmt='%.04f')


    def o3d_draw_3d_img(self):
        result_path = os.path.join(self.all_pred_txt_path[0], f'PonintNet_0.txt')
        pcd = np.genfromtxt(result_path, delimiter=" ") 
        pcd_vector = o3d.geometry.PointCloud()
        print(self.path)
        # 加载点坐标
        pcd_vector.points = o3d.utility.Vector3dVector(pcd[:, :3])
        # colors = np.random.randint(255, size=(2,3))/255
        colors = np.array([[1,0,0],[0,0,1]])
        pcd_vector.colors = o3d.utility.Vector3dVector(colors[list(map(int,pcd[:, 6])),:])
        o3d.visualization.draw_geometries([pcd_vector])

    def pc_normalize(self,pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def make_dir(self, root):
        if os.path.exists(root):
            print(f'{root} 路径已存在 无需创建')
        else:
            os.mkdir(root)
    def to_categorical(self,y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
        if (y.is_cuda):
            return new_y.cuda()
        return new_y

    def load_cheackpoint_for_models(self,name,model,cheackpoints):

        assert cheackpoints is not None,'请填写权重文件'
        assert model is not None, '请实例化模型'

        for n,m,c in zip(name,model,cheackpoints):
            print(f'正在加载{n}的权重.....')
            weight_dict = torch.load(os.path.join(c,'best_model.pth'))
            m.load_state_dict(weight_dict['model_state_dict'])
            print(f'{n}权重加载完毕')


if __name__ =='__main__':
    import copy
    time_start=time.time()
    

    img_root = r'./data/book_seam_dataset' # 数据集路径
    target_root = r'./results/444' # 输出结果路径

    path='data/book_seam_dataset/12345678/43.txt'
    
    num_classes = 2 # 填写数据集的类别数 如果是s3dis这里就填13   shapenet这里就填50

    # 导入模型  部分
    "所有的模型以PointNet++为标准  输入两个参数 输出两个参数，如果模型仅输出一个，可以将其修改为多输出一个None！！！！"
    #==============================================
    from models.pointnet2_part_seg_msg import get_model as pointnet2

    model1 = pointnet2(num_classes=num_classes,normal_channel=True).eval()

    #============================================
    # 实例化数据集
    "Dataset同理，都按ShapeNet格式输出三个变量 point_set, cls, seg # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别"
    "不是这个格式的话就手动添加一个"

    print('实例化ShapeNet')
    TEST_DATASET = PartNormalDataset(root=img_root, npoints=5000, normal_channel=True,path=path)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,
                                                    drop_last=True)
        

    model_dict = {'PonintNet': [model1,r'./log/part_seg/pointnet2_part_seg_msg/checkpoints']}
    c = Generate_txt_and_3d_img(img_root,target_root,num_classes,testDataLoader,model_dict,path=path)

    time_end=time.time()
    print('time cost',time_end-time_start,'s')