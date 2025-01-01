#pragma once

#include <dvs_msgs/Event.h>

#include <opencv2/core/core.hpp>
#include <unordered_map>
// Structure collecting the options for warping the events onto
// a histogram or image: the "image of warped events" (IWE)
struct OptionsWarp {
  // Whether to use polarity or not in the IWE
  bool use_polarity_ = true;

  // Amounf ot Gaussian blur (in pixels) to make the IWE smoother,
  // and consequently, optimize a smoother objective function
  double blur_sigma_ = 1.0;
};

// PointCompartor to use cv::Point inside map

struct PixelData {
  int indexOfMotion;
  double highestValue;
  PixelData(int index, double value) {
    indexOfMotion = index;
    highestValue = value;
  };
  PixelData() {
    indexOfMotion = 0;
    highestValue = 0.;
  };
};
struct PointHash {
  std::size_t operator()(const cv::Point2i &p) const {
    return std::hash<int>()(p.x + (p.y << 10));
  }
};
struct PointComparator {
  bool operator()(const cv::Point2i &p1, const cv::Point2i &p2) const {
    return p1.x == p2.x && p1.y == p2.y;
  }
};

void warpEvent(const cv::Point2d &vel, const dvs_msgs::Event &event,
               const double t_ref, cv::Point2d *warped_pt);
void computeImageOfWarpedEvents(
    const cv::Point2d &vel, const std::vector<dvs_msgs::Event> &events_subset,
    const cv::Size &img_size, cv::Mat *image_warped,
    const OptionsWarp &optsWarp, int motionIndex,
    std::vector<std::vector<double>> *probabilityVectors);
int findMostMatchingVelocity(const cv::Point2d vel,
                             const std::vector<cv::Point2d> &old_velocities,
                             std::vector<bool> &old_velocities_flags);
void computeImageOfWarpedEventsColored(
    std::vector<cv::Point2d> *velocities,
    std::vector<cv::Point2d> *old_velocities,
    const std::vector<dvs_msgs::Event> &events_subset, const cv::Size &img_size,
    std::vector<cv::Mat> *images_warped, cv::Mat *event_image_warped_colored,
    const int number_of_motions, const double max_events_at_Pixel_overall,
    std::unordered_map<cv::Point2i, PixelData, PointHash, PointComparator>
        *pixel_dict,
    std::vector<std::vector<std::vector<double>>> *cluster_data,
    const bool clustering_enabled, const double threshold);
void accumulateWarpedEvent(const int img_width, const int img_height,
                           const cv::Point2d &ev_warped_pt,
                           cv::Mat *image_warped, const double probability);
void accumulateWarpedEventColored(const int x, const int y,
                                  const int motion_index,
                                  const int number_of_motions,
                                  const double value,
                                  cv::Mat *event_image_warped_colored);
void computeOriginalImage(const std::vector<dvs_msgs::Event> &events_subset,
                          const cv::Size &img_size, cv::Mat *image_warped);

cv::Vec3b HSVtoRGB(cv::Vec3d hsv_vector);
cv::Vec3b getColorFromIndex(int motion_index_of_pixel, int number_of_Motions,
                            double value);