import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    
    B:batchsize, N:第一组点个数, M:第二组点个数, C:输入点通道数(xyz.C=3)
    Input:
        src: source points, [B, N, C] 
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
        batchsize个[N,M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1)) # permute:转换维度
    dist += torch.sum(src ** 2, -1).view(B, N, 1) # view:按维度填充
    dist += torch.sum(dst ** 2, -1).view(B, 1, M) # 数组广播机制，右边的式子复制N组后与dist叠加
    return dist


def index_points(points, idx):  # i按照输入的点云数据和索引返回由索引的点云数据。
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)         #view_shape=[B,S]
    view_shape[1:] = [1] * (len(view_shape) - 1)    #[1] * (len(view_shape) - 1) -> [1],即view_shape=[B,1]
    repeat_shape = list(idx.shape)    #repeat_shape=[B,S]
    repeat_shape[0] = 1    #repeat_shape=[1,S]
    #.view(view_shape)=.view(B,1)
    #.repeat(repeat_shape)=.view(1,S)
    #batch_indices的维度[B,S]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    '''
    FPS的逻辑如下：

        假设一共有n个点,整个点集为N = {f1, f2,…,fn}, 目标是选取n1个起始点做为下一步的中心点:

        随机选取一个点fi为起始点，并写入起始点集 B = {fi};
        选取剩余n-1个点计算和fi点的距离，选择最远点fj写入起始点集B={fi,fj};
        选取剩余n-2个点计算和点集B中每个点的距离, 将最短的那个距离作为该点到点集的距离, 这样得到n-2个到点集的距离，选取最远的那个点写入起始点B = {fi, fj ,fk},同时剩下n-3个点, 如果n1=3 则到此选择完毕;
        如果n1 > 3则重复上面步骤直到选取n1个起始点为止.
    '''
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape     # B:BatchSize, N:ndataset(点云中点的个数), C:dimension
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # 提取得到中心点的集合
    distance = torch.ones(B, N).to(device) * 1e10                   # 记录某个样本中所有点到某一个点的距离，先取很大
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 当前最远的点，随机初始化，范围为0~N，初始化B个，对应到每个样本都随机有一个初始最远点，B列的行向量
    batch_indices = torch.arange(B, dtype=torch.long).to(device)    # batch的索引，0~(B-1)的数组
    for i in range(npoint):
        centroids[:, i] = farthest  # 第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 取出最远点xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)      # 计算距离，-1代表行求和
        mask = dist < distance  # 一个bool值的张量数组
        distance[mask] = dist[mask]  # True的会留下，False删除
        farthest = torch.max(distance, -1)[1]  # 返回一个张量，第一项是最大值，第二项是索引,-1代表列索引
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    '''
    
    '''
    """
    Input:
        radius: local region radius                      # radius为半径，new_xyz为中心，取nsample个点
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]            # 所有点
        new_xyz: query points, [B, S, 3]      # farthest_point_sample得到S个中心点, new_xyz为中心点xyz
    Return:
        group_idx: grouped points index, [B, S, nsample]      # nsameple个点的索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # torch.arange得到索引，view转换为三维，repeat使其复制成[B,S,N]
    sqrdists = square_distance(new_xyz, xyz)          # 计算中心点与所有点之间的欧几里德距离
    group_idx[sqrdists > radius ** 2] = N       # 大于半径的点设置成N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点. 0代表输出值，1代表索引
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, nsample]， 实际就是把group_idx中的第一个点的值复制到[B, S, nsample]的维度，便利于后面的替换
    # 这里要用view是因为group_idx[:, :, 0]取出之后的tensor相当于二维Tensor，因此需要用view变成三维tensor
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 找到group_idx中值等于N的点，会输出0,1构成的三维Tensor，维度为[B,S,nsample]
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C] 中心点
    new_xyz = index_points(xyz, fps_idx)    # 中心点位置
    idx = query_ball_point(radius, nsample, xyz, new_xyz)   # 球查询得到点的索引
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]    # 球查询点的位置
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)    # 计算与中心点距离

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D] C=3，D为点的特征维度（位置、法向、颜色）
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)       #new_xyz代表中心点，用原点表示
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))   # MLP就相当于是1x1卷积
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        N是输入点的数量，C是坐标维度(C=3)，D是特征维度（除坐标维度以外的其他特征维度)
        S是输出点的数量，C是坐标维度，D'是新的特征维度
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # [B, N, 3]
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]    # pytorch的通道顺序是NCHW
        # N - Batch
        # C - Channel
        # H - Height
        # W - Width
        # 对[3+D, nsample]的维度上做逐像素的卷积，结果相当于对单个C+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):    
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        # 针对多个radius和nsample取点
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]   # 所有点
            xyz2: sampled input points position data, [B, C, S]   # 采样点
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        "  将B C N 转换为B N C 然后利用插值将高维点云数目S 插值到低维点云数目N (N大于S)"
        "  xyz1 低维点云  数量为N   xyz2 高维点云  数量为S"
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        
        "如果最后只有一个点，就将S直复制N份后与与低维信息进行拼接"
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2) # [B,N,S]
            dists, idx = dists.sort(dim=-1)    # 找到距离最近的三个邻居
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B,N,3],N个点与这S个距离最近的前三个点的索引

            dist_recip = 1.0 / (dists + 1e-8)    # 求距离的倒数 2,512,3 对应论文中的 Wi(x)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)   # 也就是将距离最近的三个邻居的加起来  此时对应论文中公式的分母部分
            weight = dist_recip / norm   
            """
            这里的weight是计算权重  dist_recip中存放的是三个邻居的距离  norm中存放是距离的和  
            两者相除就是每个距离占总和的比重 也就是weight
            """
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)  # 点乘

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

