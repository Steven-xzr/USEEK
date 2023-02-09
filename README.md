# USEEK
USEEK: Unsupervised SE(3)-Equivariant 3D Keypoints for Generalizable Manipulation.
Project website: https://sites.google.com/view/useek.
The paper is available at https://arxiv.org/abs/2209.13864.

## A map of the repository
+ The `merger/pointnetpp` folder contains the [Pytorch Implementation of PointNet and PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) repository with some minor changes. It is adapted to make compatible relative imports.
+ The `merger/composed_chamfer.py` file contains an efficient implementation of proposed Composite Chamfer Distance (CCD).
+ The `merger/data_flower.py` file is for data loading and preprocessing.
+ The `merger/merger_net.py` file contains the `Skeleton Merger` implementation.
+ The root folder contains several scripts for training and testing.

## Dataset
The ShapeNetCore.v2 dataset used in the paper is available from the [Point Cloud Datasets](https://github.com/AnTao97/PointCloudDatasets) repository.


https://drive.google.com/drive/folders/1vifujGJ-Aq3-jZKpHN9Aqi3gvIyb0Paw?usp=sharing
