#!/usr/bin/env python3

import pcl
import rospy
import lidar_pcl_helper
from sensor_msgs.msg import PointCloud2

def do_voxel_grid_downssampling(pcl_data,leaf_size):
    '''
    Create a VoxelGrid filter object for a input point cloud
    :param pcl_data: point cloud data subscriber
    :param leaf_size: voxel(or leaf) size
    :return: Voxel grid downsampling on point cloud
    :https://github.com/fouliex/RoboticPerception
    '''
    vox = pcl_data.make_voxel_grid_filter()
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size) # The bigger the leaf size the less information retained
    return vox.filter()

def do_passthrough(pcl_data,filter_axis,axis_min,axis_max):
    '''
    Create a PassThrough  object and assigns a filter axis and range.
    :param pcl_data: point could data subscriber
    :param filter_axis: filter axis
    :param axis_min: Minimum  axis to the passthrough filter object
    :param axis_max: Maximum axis to the passthrough filter object
    :return: passthrough on point cloud
    '''
    passthrough = pcl_data.make_passthrough_filter()
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    return passthrough.filter()

def do_ransac_plane_normal_segmentation(point_cloud, input_max_distance):
    segmenter = point_cloud.make_segmenter_normals(ksearch=50)
    segmenter.set_optimize_coefficients(True)
    segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  #pcl_sac_model_plane
    segmenter.set_normal_distance_weight(0.1)
    segmenter.set_method_type(pcl.SAC_RANSAC) #pcl_sac_ransac
    segmenter.set_max_iterations(100)
    segmenter.set_distance_threshold(input_max_distance) #0.03)  #max_distance
    indices, coefficients = segmenter.segment()

    inliers = point_cloud.extract(indices, negative=False)
    outliers = point_cloud.extract(indices, negative=True)

    return indices, inliers, outliers

def do_statistical_outlier_filtering(pcl_data,mean_k,tresh):
    '''
    :param pcl_data: point could data subscriber
    :param mean_k:  number of neighboring points to analyze for any given point
    :param tresh:   Any point with a mean distance larger than global will be considered outlier
    :return: Statistical outlier filtered point cloud data
    eg) cloud = do_statistical_outlier_filtering(cloud,10,0.001)
    : https://github.com/fouliex/RoboticPerception
    '''
    outlier_filter = pcl_data.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(mean_k)
    outlier_filter.set_std_dev_mul_thresh(tresh)
    return outlier_filter.filter()

def do_euclidean_clustering(white_cloud):
        tree = white_cloud.make_kdtree()

        # Create Cluster-Mask Point Cloud to visualize each cluster separately
        ec = white_cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.1)
        ec.set_MinClusterSize(10)
        ec.set_MaxClusterSize(25000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()
        cluster_color = lidar_pcl_helper.get_color_list(len(cluster_indices))

        color_cluster_point_list = []

        for j, indices in enumerate(cluster_indices):
            for i, indice in enumerate(indices):
                color_cluster_point_list.append([white_cloud[indice][0],
                                                white_cloud[indice][1],
                                                white_cloud[indice][2],
                                                lidar_pcl_helper.rgb_to_float(cluster_color[j])])

        cluster_cloud = pcl.PointCloud_PointXYZRGB()
        cluster_cloud.from_list(color_cluster_point_list)
        return cluster_cloud,cluster_indices

class LiDARProcessing():
    def __init__(self) -> None:
        rospy.init_node("lidar_clustering")
        rospy.Subscriber("/lidar3D", PointCloud2, self.callback)
        self.roi_pub = rospy.Publisher("/lidar3D_ROI", PointCloud2, queue_size=1)
        self.ransac_pub = rospy.Publisher("/lidar3D_ransac", PointCloud2, queue_size=1)
        self.filt_pub = rospy.Publisher("/lidar3D_filtered", PointCloud2, queue_size=1)
        self.cluster_pub = rospy.Publisher("/lidar3D_clustered", PointCloud2, queue_size=1)

    def callback(self, data):
        cloud = lidar_pcl_helper.ros_to_pcl(data)

        # downsampling
        cloud = do_voxel_grid_downssampling(cloud,0.1)
        # x 값이 0부터 10인 것까지 ROI
        filter_axis = 'x'
        axis_min = 0.0
        axis_max = 10.0
        cloud = do_passthrough(cloud, filter_axis, axis_min, axis_max)
        # y 값이 -5부터 5인 것까지 ROI
        filter_axis = 'y'
        axis_min = -5.0
        axis_max = 5.0
        cloud = do_passthrough(cloud, filter_axis, axis_min, axis_max)
        cloud_new = lidar_pcl_helper.pcl_to_ros(cloud)
        cloud_new.header.frame_id = "velodyne"
        self.roi_pub.publish(cloud_new)

        # RANSAC 적용, inliers & outliers 추출
        _, _, cloud= do_ransac_plane_normal_segmentation(cloud,0.15)
        cloud_new = lidar_pcl_helper.pcl_to_ros(cloud)
        cloud_new.header.frame_id = "velodyne"
        self.ransac_pub.publish(cloud_new)
        
        # 군집화
        cloud = lidar_pcl_helper.XYZRGB_to_XYZ(cloud)
        cloud, _ = do_euclidean_clustering(cloud)
        cloud_new = lidar_pcl_helper.pcl_to_ros(cloud)
        cloud_new.header.frame_id = "velodyne"
        self.cluster_pub.publish(cloud_new)
        
        rospy.loginfo("Publishing!")


if __name__ == "__main__":
    try:
        LiDARProcessing()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass