# USEEK: Unsupervised SE(3)-Equivariant 3D Keypoints for Generalizable Manipulation (ICRA2023)
Project website: https://sites.google.com/view/useek.
The paper is available at https://arxiv.org/abs/2209.13864.

## A map of the repository
+ `1_train_merger.py` trains the teacher network, following the procedure in [Skeleton Merger](https://github.com/eliphatfs/SkeletonMerger) with minor adaptations.
+ `2_prepare_segment.py` prepares the segmentation pseudo labels for the student network.
+ `3_train_useek.py` trains the student network, which is adapted from the segmentation network of [SPRIN](https://github.com/qq456cvb/SPRIN).
+ `4_predictor_keypointnet.py` and `5_eval_keypointnet.py` test the student network on the [KeypointNet](https://github.com/qq456cvb/KeypointNet) dataset with SE(3) transformations.

## Dataset
The ShapeNetCore.v2 dataset used for training is available from the [Point Cloud Datasets](https://github.com/AnTao97/PointCloudDatasets) repository.

## Pre-trained models
The pre-trained models for both the teacher and student networks are available at [Google Drive](https://drive.google.com/drive/folders/1vifujGJ-Aq3-jZKpHN9Aqi3gvIyb0Paw?usp=sharing).
