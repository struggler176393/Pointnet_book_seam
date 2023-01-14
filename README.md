# 一个利用pointnet++做的书缝识别项目（内含数据集）
网络结构有一定修改，因为pointnet原网络结构是适应shapenet中16种物体的，本数据集只有一种物体。
数据集是把两本书并排放，中间留条缝用来识别，中间的缝标签值为1，其余为0。
data/book_seam_dataset/12345678/1.txt到40.txt有标签值，总共七列，前三列位置，中间三列法向量，最后一列标签。
data/book_seam_dataset/12345678/41.txt到43.txt无标签值，总共六列，前三列位置，后三列法向量

## 训练（用零件分割的网络）
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg

## 测试（用open3d做的可视化）
python visualize_partseg_open3d.py 