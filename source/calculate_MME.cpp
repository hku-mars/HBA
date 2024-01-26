#include <mutex>
#include <string>
#include <fstream>
#include <iostream>

#include <assert.h>
#include <ros/ros.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/console/parse.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/geometry.h>
#include <pcl/common/centroid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

string file_path;
int THR_NUM;

double computeEntropy(pcl::PointCloud< PointType >::Ptr cloud)
{
	Eigen::Vector4f centroid;
	Eigen::Matrix3f covarianceMatrixNormalized = Eigen::Matrix3f::Identity();;
	pcl::compute3DCentroid (*cloud, centroid);
	pcl::computeCovarianceMatrixNormalized (*cloud, centroid, covarianceMatrixNormalized);
	double determinant = static_cast<double>(((2 * M_PI * M_E) * covarianceMatrixNormalized).determinant());
	return 0.5f*log(determinant);
}

void PC2Entropy(double& Entropy, int part_num)
{
  pcl::PointCloud<PointType>::Ptr full_cloud(new pcl::PointCloud<PointType>);
  pcl::io::loadPCDFile(file_path, *full_cloud);

  pcl::KdTreeFLANN<PointType> kdtree;
	kdtree.setInputCloud(full_cloud);
  double Entropy_ = 0;

  int partSize = full_cloud->points.size()/THR_NUM;
  pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
  for(size_t i = part_num*partSize; i < (part_num+1)*partSize; i++)
    pc->points.push_back(full_cloud->points[i]);
  
  for(size_t i = 0; i < pc->points.size(); i++)
  {
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    int numberOfNeighbors = kdtree.radiusSearch(pc->points[i], 0.3, pointIdxRadiusSearch, pointRadiusSquaredDistance);
    double localEntropy = 0;
    double localPlaneVariance = 0;
    if(numberOfNeighbors > 15)
    {
      pcl::PointCloud< PointType>::Ptr localCloud (new pcl::PointCloud< PointType>);
      for(size_t iz = 0; iz < pointIdxRadiusSearch.size(); ++iz)
        localCloud->points.push_back(full_cloud->points[pointIdxRadiusSearch[iz]]);
      localEntropy = computeEntropy(localCloud);
      // localPlaneVariance = computePlaneVariance(localCloud);
    }
    Entropy_ += localEntropy;
  }
  
  Entropy = Entropy_/pc->points.size();
  cout<<"complete"<<endl;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "calculate_MME");
  ros::NodeHandle nh("~");
  
  pcl::PointCloud<PointType>::Ptr full_cloud(new pcl::PointCloud<PointType>);
  pcl::PointCloud<PointType>::Ptr tree_frame(new pcl::PointCloud<PointType>);
  
  double MME = 0;

  nh.getParam("file_path", file_path);
  nh.getParam("THR_NUM", THR_NUM);

  vector<thread*> mthreads(THR_NUM);
  vector<double> Entropys;
  Entropys.resize(THR_NUM);
  
  for(int i = 0; i < THR_NUM; i++)
    mthreads[i] = new thread(PC2Entropy, ref(Entropys[i]), i);
  
  for(int i = 0; i < THR_NUM; i++)
  {
    mthreads[i]->join();
    delete mthreads[i];
  }

  double sum = std::accumulate(std::begin(Entropys), std::end(Entropys), 0.0);  
  double mean =  sum / Entropys.size(); //均值  
  double accum  = 0.0;  
  std::for_each (std::begin(Entropys), std::end(Entropys), [&](const double d)
  {
    accum  += (d-mean)*(d-mean);  
  });  
  double stdev = sqrt(accum/(Entropys.size()-1));
  cout<<"MME mean "<<mean<<" std "<<stdev<<endl;
}