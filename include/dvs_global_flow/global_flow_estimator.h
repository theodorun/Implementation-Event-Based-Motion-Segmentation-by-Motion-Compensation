#pragma once

#include <cv_bridge/cv_bridge.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <deque>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "image_warped_events.h"

enum
{
  VARIANCE_CONTRAST = 1,
  MEAN_SQUARE_CONTRAST = 2,
  MEAN_POWER_CONTRAST_NON_ZERO = 3,
  MEAN_SQUARE_CONTRAST_NON_ZERO = 4,
};

namespace dvs_global_flow
{

  // Options of the method
  struct OptionsMethod
  {
    // Sliding Window options
    // Number of events used to synthetize an image of warped events
    int num_events_per_image_ = 15000;

    // Amount of overlap between consecutive packets of events: a number >= 0. (no
    // overlap) and <1.0 (full)
    int num_events_slide_ = 15000;

    // Objective function to be optimized: 0=Variance, 1=RMS, etc.
    int contrast_measure_ = VARIANCE_CONTRAST;

    // Options of the image of warped events
    OptionsWarp opts_warp_;

    // Verbosity / printing level
    unsigned int verbose_ = 0;

    // Number of motions
    int num_motions_ = 1;

    // Pixel distance for the pool of motion initialization
    int pixel_distance_ = 20;

    // Number of directions for the pool of motion initialization
    int number_of_directions_ = 4;

    // Number of steps for the pool of motion initialization
    int number_of_pool_steps_ = 5;

    bool clustering_enabled_ = false;

    double pixel_displaying_and_clustering_threshold_ = 0.05;

    int max_percentage_difference = 0;
  };

  class GlobalFlowEstimator
  {
  public:
    GlobalFlowEstimator(ros::NodeHandle &nh);
    ~GlobalFlowEstimator();

  private:
    ros::NodeHandle nh_;  // Node handle used to subscribe to ROS topics
    ros::NodeHandle pnh_; // Private node handle for reading parameters

    // Subscribers
    ros::Subscriber event_sub_;
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);

    // Publishers
    image_transport::Publisher image_pub_;
    cv_bridge::CvImage cv_event_image_;
    std::vector<ros::Publisher> vel_publishers_;
    void publishGlobalFlow();
    void publishEventImage();
    ros::Time time_packet_;

    // Sliding window of events
    std::deque<dvs_msgs::Event> events_;
    std::vector<dvs_msgs::Event> events_subset_;
    int idx_first_ev_; // index of first event of processing window
    void getSubsetOfEvents();
    void slideWindow();

    // Motion estimation

    OptionsMethod opts_; // Options of the method
    cv::Mat event_image_warped_;

    std::vector<cv::Mat> images_warped_;
    std::vector<char> motionsFinished;

    int counterForSaving = 0;

    // std::vector<cv::Mat> images_warped_2;
    // std::vector<char> motions_finished_;
    //  Colored version of the warped image

    // Probability Map:
    std::vector<std::vector<double>> probabilityVectors;
    std::vector<cv::Point2d> velocities; // global flow (vx,vy)ev_warped_pt
    std::vector<cv::Point2d> old_velocities;

    std::vector<std::vector<std::vector<double>>> clusterData;
    std::unordered_map<cv::Point2i, PixelData, PointHash, PointComparator>
        pixel_dict_;
    double max_events_at_Pixel_overall;

    void findInitialFlow();
    void maximizeContrast();
    void maximizeContrastOneMotion(
        cv::Point2d &velocity,
        std::vector<dvs_msgs::Event> *events_subset_remaining);

    // Calculate Probabilitys
    const bool calculateProbability();

    // Camera information (size, intrinsics, lens distortion)
    cv::Size img_size_;
  };

} // namespace dvs_global_flow
