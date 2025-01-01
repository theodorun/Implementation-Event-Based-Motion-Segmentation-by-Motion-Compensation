#include "dvs_global_flow/image_warped_events.h"

#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/imgproc/imgproc.hpp>
void warpEvent(const cv::Point2d &vel, const dvs_msgs::Event &event,
               const double t_ref, cv::Point2d *warped_pt)
{
  double tDelta = event.ts.toSec() - t_ref;
  warped_pt->x = event.x - tDelta * vel.x;
  warped_pt->y = event.y - tDelta * vel.y;
}

void accumulateWarpedEvent(const int img_width, const int img_height,
                           const cv::Point2d &ev_warped_pt,
                           cv::Mat *image_warped, const double probability)
{
  // Accumulate warped events, using bilinear voting (polarity or count)
  const int xx = static_cast<int>(ev_warped_pt.x);
  const int yy = static_cast<int>(ev_warped_pt.y);

  // if warped point is within the image, accumulate polarity
  if (0 <= xx && xx < img_width - 1 && 0 <= yy && yy < img_height - 1)
  {
    // Accumulate warped events on the IWE
    double dX = ev_warped_pt.x - xx;
    double dY = ev_warped_pt.y - yy;
    image_warped->at<double>(yy, xx) += ((1 - dX) * (1 - dY)) * probability;
    image_warped->at<double>(yy, xx + 1) += (dX * (1 - dY)) * probability;
    image_warped->at<double>(yy + 1, xx) += ((1 - dX) * dY) * probability;
    image_warped->at<double>(yy + 1, xx + 1) += (dX * dY) * probability;
  }
}

// here we need to claculate the Image of Warped Events
void computeImageOfWarpedEvents(
    const cv::Point2d &vel, const std::vector<dvs_msgs::Event> &events_subset,
    const cv::Size &img_size, cv::Mat *image_warped,
    const OptionsWarp &opts_warp, int motion_index,
    std::vector<std::vector<double>> *probability_vectors)
{
  const int width = img_size.width;
  const int height = img_size.height;

  *image_warped = cv::Mat(height, width, CV_64FC1, cv::Scalar(0.));
  // Loop through all events
  const double tRef = events_subset.front().ts.toSec(); // warp wrt 1st event

  for (int j = 0; j < events_subset.size(); j++)
  {
    //  Warp event according to candidate flow and accumulate on the IWE
    cv::Point2d evWarpedPt;
    warpEvent(vel, events_subset[j], tRef, &evWarpedPt);
    double probability = (*probability_vectors)[motion_index][j];
    accumulateWarpedEvent(width, height, evWarpedPt, image_warped,
                          probability);
  }

  // Smooth the IWE (to spread the votes)
  if (opts_warp.blur_sigma_ > 0)
  {
  cv:
    GaussianBlur(*image_warped, *image_warped, cv::Size(0, 0),
                 opts_warp.blur_sigma_);
  }
}

void computeImageOfWarpedEventsColored(
    std::vector<cv::Point2d> *velocities,
    std::vector<cv::Point2d> *old_velocities,
    const std::vector<dvs_msgs::Event> &events_subset, const cv::Size &img_size,
    std::vector<cv::Mat> *images_warped, cv::Mat *event_image_warped_colored,
    const int number_of_motions, const double max_events_at_pixel_overall,
    std::unordered_map<cv::Point2i, PixelData, PointHash, PointComparator>
        *pixel_dict,
    std::vector<std::vector<std::vector<double>>> *cluster_data,
    const bool clustering_enabled, const double threshold)
{
  // Init old_velocities_flags
  //  int indexOfNearestMotion = findMostMatchingVelocity(vel, old_velocities,
  //  old_velocities_flags);
  std::unordered_map<int, int> matchingDict;
  std::vector<bool> oldVelocitiesFlags(old_velocities->size(), false);
  for (int k = 0; k < number_of_motions; k++)
  {
    matchingDict[k] = findMostMatchingVelocity(
        (*velocities)[k], *old_velocities, oldVelocitiesFlags);
  }
  const int width = img_size.width;
  const int height = img_size.height;

  double maxNumberOfEventsAtPixel = 0;
  int numberOfMotionWithHighestValue = 0;

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      numberOfMotionWithHighestValue = 0;
      maxNumberOfEventsAtPixel = 0;
      for (int indexMotion = 0; indexMotion < number_of_motions;
           indexMotion++)
      {
        if ((*images_warped)[indexMotion].at<double>(y, x) >
            maxNumberOfEventsAtPixel)
        {
          maxNumberOfEventsAtPixel =
              (*images_warped)[indexMotion].at<double>(y, x);
          numberOfMotionWithHighestValue = indexMotion;
        }
      }
      if (max_events_at_pixel_overall > 0)
      {
        maxNumberOfEventsAtPixel /= max_events_at_pixel_overall;
      }
      else
      {
        LOG(INFO)
            << "There is an error, max_events_at_Pixel_overall need to be > 0 ";
      }
      if (maxNumberOfEventsAtPixel > 1)
      {
        LOG(INFO) << "Errror Max number of Events at one Pixel greater than 1 "
                  << maxNumberOfEventsAtPixel;
      }
      // Only cluster and accumulate if they are above Treshhold
      if (maxNumberOfEventsAtPixel >= threshold)
      {
        if (clustering_enabled)
        {
          // Insert the indexOf the best velocity and the value at that pixel
          // into Dict
          (*pixel_dict)[cv::Point2i(y, x)] =
              PixelData(numberOfMotionWithHighestValue,
                        maxNumberOfEventsAtPixel);
          (*cluster_data)[numberOfMotionWithHighestValue].push_back(
              {static_cast<double>(y), static_cast<double>(x)});
          // constructImage
        }
        else
        {
          // accumulateEvents
          accumulateWarpedEventColored(
              x, y, matchingDict[numberOfMotionWithHighestValue],
              number_of_motions, maxNumberOfEventsAtPixel,
              event_image_warped_colored);
        }
      }
    }
  }
}

int findMostMatchingVelocity(const cv::Point2d vel,
                             const std::vector<cv::Point2d> &old_velocities,
                             std::vector<bool> &old_velocities_flags)
{
  if (old_velocities.empty())
  {
    throw std::runtime_error("Array of velocities is empty!");
  }

  double minDistance = 0;

  int bestMatch = 0;
  minDistance = cv::norm(vel - old_velocities[0]);

  // Iterate through the array to find the closest point
  for (int i = 0; i < old_velocities.size(); ++i)
  {
    if (old_velocities_flags[i])
    {
      continue;
    }
    double distance = cv::norm(vel - old_velocities[i]);
    if (distance < minDistance)
    {
      minDistance = distance;
      bestMatch = i;
    }
  }
  if (bestMatch <= -1 || bestMatch >= old_velocities.size())
  {
    throw std::runtime_error("Array of points is empty!");
  }
  old_velocities_flags[bestMatch] = true;

  return bestMatch;
}

void accumulateWarpedEventColored(const int x, const int y,
                                  const int motion_index,
                                  const int number_of_motions,
                                  const double value,
                                  cv::Mat *event_image_warped_colored)

{
  (*event_image_warped_colored).at<cv::Vec3b>(y, x) =
      getColorFromIndex(motion_index, number_of_motions, value);

  // Accumulate warped events, using bilinear voting (polarity or count)
}

void computeOriginalImage(const std::vector<dvs_msgs::Event> &events_subset,
                          const cv::Size &img_size, cv::Mat *image_warped)
{
  const int width = img_size.width;
  const int height = img_size.height;

  *image_warped = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int j = 0; j < events_subset.size(); j++)
  {
    cv::Vec3b colorVec;
    colorVec[0] = 80;
    colorVec[1] = 255;
    colorVec[2] = 175;
    const int xx = static_cast<int>(events_subset[j].x);
    const int yy = static_cast<int>(events_subset[j].y);
    // if warped point is within the image
    if (0 <= xx && xx < width && 0 <= yy && yy < height)
    {
      image_warped->at<cv::Vec3b>(yy, xx) = colorVec;
    }
  }
}
cv::Vec3b HSVtoRGB(cv::Vec3d hsv_vector)
{
  cv::Vec3b bgrVec;
  // SV is between 0-1 H between 0-360
  double red = 0.;
  double blue = 0.;
  double green = 0.;
  double fC = hsv_vector[2] * hsv_vector[1]; // Chroma
  double fHPrime = fmod(hsv_vector[0] / 60.0, 6);
  double fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
  double fM = hsv_vector[2] - fC;

  if (0 <= fHPrime && fHPrime < 1)
  {
    red = fC;
    green = fX;
    blue = 0;
  }
  else if (1 <= fHPrime && fHPrime < 2)
  {
    red = fX;
    green = fC;
    blue = 0.;
  }
  else if (2 <= fHPrime && fHPrime < 3)
  {
    red = 0.;
    green = fC;
    blue = fX;
  }
  else if (3 <= fHPrime && fHPrime < 4)
  {
    red = 0.;
    green = fX;
    blue = fC;
  }
  else if (4 <= fHPrime && fHPrime < 5)
  {
    red = fX;
    green = 0.;
    blue = fC;
  }
  else if (5 <= fHPrime && fHPrime < 6)
  {
    red = fC;
    green = 0.;
    blue = fX;
  }
  else
  {
    red = 0.;
    green = 0.;
    blue = 0.;
  }

  red += fM;
  green += fM;
  blue += fM;

  red *= 255;
  green *= 255;
  blue *= 255;

  bgrVec[0] = static_cast<int>(blue);
  bgrVec[1] = static_cast<int>(green);
  bgrVec[2] = static_cast<int>(red);
  return bgrVec;
}
cv::Vec3b getColorFromIndex(int motion_index_of_pixel, int number_of_motions,
                            double value)
{
  if (number_of_motions > 0)
  {
    double angle = (360 / number_of_motions) * motion_index_of_pixel;
    cv::Vec3d colorVec;
    colorVec[0] = angle;
    colorVec[1] = 1.;
    colorVec[2] = value;
    return HSVtoRGB(colorVec);
  }
  else
  {
    cv::Vec3d colorVec;
    colorVec[0] = 10.;
    colorVec[1] = 1.;
    colorVec[2] = 1.;
    return HSVtoRGB(colorVec);
  }
}