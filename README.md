# HBA: A Globally Consistent and Efficient Large-Scale LiDAR Mapping Module

## **1. Introduction**
**HBA** is developed to resolve the issue that the divergence within the point cloud map cannot be fully eliminated after pose graph optimization (PGO). Though PGO is time-efficient, it does not directly optimize the mapping consistency. LiDAR bundle adjustment (BA) could mitigate this issue; however, it is too time-consuming on large-scale maps. HBA proposes a hierarchical structure to divide the huge LidAR BA problem into multiple smaller BA problems and use the PGO to smoothly update the entire LiDAR poses. HBA could achieve similar accuracy compared with the original BA method but with much less computation time.

<div align="center">
  <div align="center">
    <img src="figure/lidar_frame.png"  width="100%" />
  </div>
  <font color=#a0a0a0 size=2>Pyramid structure of our proposed Hierarchical Bundle Adjustment.</font>
</div>

## **2. Paper and Video**
Our paper has been publised on [IEEE RA-L](https://ieeexplore.ieee.org/abstract/document/10024300) and the corresponding video could be watched on [Bilibili](https://www.bilibili.com/video/BV1Qg41127j9/?spm_id_from=333.999.0.0) or [YouTuBe](https://youtu.be/CuLnTnXVujw). If you find our work useful in your research, please consider citing:

```
@ARTICLE{10024300,
  author={Liu, Xiyuan and Liu, Zheng and Kong, Fanze and Zhang, Fu},
  journal={IEEE Robotics and Automation Letters}, 
  title={Large-Scale LiDAR Consistent Mapping Using Hierarchical LiDAR Bundle Adjustment}, 
  year={2023},
  volume={8},
  number={3},
  pages={1523-1530},
  doi={10.1109/LRA.2023.3238902}}
```

## **3. Run the Code**
### 3.1 Prerequisites
Our code is tested under [Ubuntu 20.04](https://releases.ubuntu.com/focal/) with [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) and [Ubuntu 18.04](https://releases.ubuntu.com/18.04/) with [ROS Melodic](https://wiki.ros.org/melodic/Installation/Ubuntu), [PCL 1.10.0](https://github.com/PointCloudLibrary/pcl/releases), [Eigen 3.3.7](https://gitlab.com/libeigen/eigen/-/releases/3.3.7), and [GTSAM 4.1.1](https://github.com/borglab/gtsam).

### 3.2 File Structure
To test on your own data, please prepare the file structure as follows: a folder named `pcd` containing point clouds and a `pose.json` file containing the initial poses of each LiDAR scan. Noted the format of the pose is `tx ty tz qw qx qy qz`.
```
.
├── pcd
│   ├── 0.pcd
│   └── 1.pcd
└── pose.json
```
### 3.3 Important Parameters
#### in `hba.launch` file
* `total_layer_num`: number of layers used in HBA. Default value is `3`.
* `pcd_name_fill_num`: number of prefix zeros in pcd file name (e.g., if pcd starts with `00000.pcd`, please fill in `5`). Default value is `0`.
* `thread_num`: number of threads used in cpu parallel computation. Default value is `16`.
#### in `hba.hpp` file
* `downsample_size`: leaf size used for point cloud downsampling of each LiDAR scan. Default value is `0.1`.
* `voxel_size`: initial voxel size used in LiDAR BA. Default value is `4.0`.
* `eigen_ratio`: threshold used to determine whether this voxel contains valid plane feature. Larger value means less strict threshold. Default value is `0.1`.
* `reject_ratio`: threshold used to reject the largest certain amount of voxels (residuals) used in optimization. Default value is `0.05`.
#### in `ba.hpp` file
* `WIN_SIZE`: window size used in local BA. Default value is `10`.
* `GAP`: stride size between the start of each two adjacent windows. Default value is `5`.
* `layer_limit`: the maximum number of voxel recut times in LiDAR BA. Default value is `2`.

Remark: in the `global_ba` function, we use a slightly larger `eigen_ratio=0.2` parameter than that in local BA, which usually leads to a faster convergence. However, you could adjust them in `cut_voxel` function from the `hba.cpp` file. Smaller voxel size and eigen ratio parameters usually lead to a better accuracy but a longer computation time.

### 3.4 Try Our Data
The zipped pcd files and their initial poses of our self-collected [park](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/xliuaa_connect_hku_hk/EbIOZhucOxJMs_m5_4JheNUBIj1FILxNmgVX0vt4C7madw?e=LD1XQh) and the public [KITTI 07](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/xliuaa_connect_hku_hk/EdDPCY06Xa5CnmKG7_fbmq8BW9jpNZGRsVxtHLt80voKaw?e=nJWZjF) dataset have been uploaded to OneDrive. You could download them and directly run the code with the parameters provided.

Remark: `hba.launch` only optimizes the LiDAR poses but does not visualize the point cloud map and `visualize.launch` is used to view the point cloud map. One more thing, when you launch the `hba.launch`, when the launch completes, it only optimizes the poses once. So if you are not satisfied with the result, you could execute the launch again.

## **4. Applications**

### 4.1 Globally Optimize the Point Cloud Mapping Consistency

#### 4.1.1 Close the Loop in Odometry (See Our Results Below on KITTI Sequences)

<div align="center"><img src="figure/kitti.gif"  width="100%" /></div>
<div align="center"><img src="figure/kitti05.jpg"  width="100%" /></div>

#### 4.1.2 Further Optimize the Mapping Consistency (See Our Results on Self-Collected Dataset)
<div align="center"><img src="figure/lg02.jpg"  width="100%" /></div>

### 4.2 Provide a Point Cloud Map with Centimeter-Level of Precision

<!-- #### 3.2.1 HILTI SLAM Challenge
HBA has participated in [HILTI SLAM Challenge](https://hilti-challenge.com/) and achieved centimeter level of precision (millimeter level in some sequences). Currently, HBA ranked #1 on leaderboard of the [HILTI SLAM Challenge 2022](https://hilti-challenge.com/leader-board-2022.html) and #2 on leaderboard of HILTI SLAM Challenge 2023.

<!-- #### 3.2.2 MARSIM -->
[MARSIM](https://github.com/hku-mars/MARSIM) is a lightweight point-realistic simulator for LiDAR-based UAVs, to which **HBA** has contributed more than ten real-world point cloud maps with centimeter-level of precision.

## **5. Acknowledgements**
In the development of **HBA**, we stand on the state-of-the-art work: [BALM2](https://github.com/hku-mars/BALM).

## **6. License**
The source code is released under [GPLv2](LICENSE) license.

We are still working on improving the performance and reliability of our codes. For any technical issues, please contact us via email xliuaa@connect.hku.hk. For commercial use, please contact Dr. Fu Zhang fuzhang@hku.hk.