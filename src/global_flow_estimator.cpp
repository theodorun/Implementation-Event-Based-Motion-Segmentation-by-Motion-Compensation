#include "dvs_global_flow/global_flow_estimator.h"

#include <geometry_msgs/PointStamped.h>
#include <glog/logging.h>

#include "../../hdbscan/Hdbscan/hdbscan.hpp"
#include "dvs_global_flow/image_warped_events.h"
#include "dvs_global_flow/util.h"

namespace dvs_global_flow
{

  OptionsMethod loadBaseOptions(ros::NodeHandle &pnh)
  {
    OptionsMethod opts;
    const int defaultNumberOfEvents = 5000;
    // Sliding window parameters
    opts.num_events_per_image_ =
        pnh.param("num_events_per_image", defaultNumberOfEvents);
    LOG(INFO) << "Found parameter: num_events_per_image = "
              << opts.num_events_per_image_;

    opts.num_events_slide_ =
        pnh.param("num_events_slide", defaultNumberOfEvents);
    LOG(INFO) << "Found parameter: num_events_slide = " << opts.num_events_slide_;

    // Objective function parameters
    opts.contrast_measure_ = pnh.param("contrast_measure", 0);
    LOG(INFO) << "Found parameter: contrast_measure = " << opts.contrast_measure_;

    // Event warping parameters
    opts.opts_warp_.blur_sigma_ = pnh.param("gaussian_smoothing_sigma", 1.0);
    LOG(INFO) << "Found parameter: gaussian_smoothing_sigma = "
              << opts.opts_warp_.blur_sigma_;

    // Verbosity / printing level
    opts.verbose_ = pnh.param("verbosity", 0);
    LOG(INFO) << "Found parameter: verbosity = " << opts.verbose_;
    // Number of motions
    opts.num_motions_ = pnh.param("num_motions", 1);
    LOG(INFO) << "Found parameter: num_motions = " << opts.num_motions_;

    // Pixel distance for the pool of motion initialization
    opts.pixel_distance_ = pnh.param("pixel_distance", 20);
    LOG(INFO) << "Found parameter: pixel_distance = " << opts.pixel_distance_;
    // Number of directions for the pool of motion initialization
    opts.number_of_directions_ = pnh.param("number_of_directions", 16);
    LOG(INFO) << "Found parameter: number_of_directions = " << opts.number_of_directions_;
    // Number of steps for the pool of motion initialization
    opts.number_of_pool_steps_ = pnh.param("number_of_pool_steps", 50);
    LOG(INFO) << "Found parameter: number_of_pool_steps = " << opts.number_of_pool_steps_;

    opts.clustering_enabled_ = pnh.param("clustering_enabled", false);
    LOG(INFO) << "Found parameter: clustering_enabled = "
              << opts.clustering_enabled_;

    opts.pixel_displaying_and_clustering_threshold_ =
        pnh.param("pixel_displaying_and_clustering_threshold", 0.);
    LOG(INFO) << "Found parameter: pixel_displaying_and_clustering_threshold = "
              << opts.pixel_displaying_and_clustering_threshold_;

    opts.max_percentage_difference =
        pnh.param("max_percentage_difference", 0.);
    LOG(INFO) << "Found parameter: max_percentage_difference = "
              << opts.max_percentage_difference;

    return opts;
  }

  GlobalFlowEstimator::GlobalFlowEstimator(ros::NodeHandle &nh)
      : nh_(nh),
        pnh_("~"),
        event_sub_(nh_.subscribe("events", 0,
                                 &GlobalFlowEstimator::eventsCallback, this)),
        img_size_(0, 0)
  {
    //  Set up subscribers

    // Set up publishers
    image_transport::ImageTransport imageTransport(nh_);
    image_pub_ = imageTransport.advertise("dvs_motion_compensated", 1);

    opts_ = loadBaseOptions(pnh_);
    for (int num = 0; num < opts_.num_motions_; num++)
    {
      // set velocities to = 0
      velocities.push_back(cv::Point2d(0, 0));
      old_velocities.push_back(cv::Point2d(0, 0));

      std::string velPublisherString = "velocity_" + std::to_string(num);
      vel_publishers_.push_back(
          nh_.advertise<geometry_msgs::PointStamped>(velPublisherString, 1));
    }

    // Sliding window
    idx_first_ev_ = 0; // Index of first event of processing window
    time_packet_ = ros::Time(0);
  }

  GlobalFlowEstimator::~GlobalFlowEstimator()
  {
    image_pub_.shutdown();
    for (int num = 0; num < opts_.num_motions_; num++)
    {
      vel_publishers_[num].shutdown();
    }
  }

  void GlobalFlowEstimator::publishGlobalFlow()
  {
    for (int num = 0; num < opts_.num_motions_; num++)
    {
      geometry_msgs::PointStamped globalFlowMsg;
      globalFlowMsg.point.x = velocities[num].x;
      globalFlowMsg.point.y = velocities[num].y;
      globalFlowMsg.point.z = 0.;
      globalFlowMsg.header.stamp = time_packet_;
      vel_publishers_[num].publish(globalFlowMsg);
    }
  }

  const bool GlobalFlowEstimator::calculateProbability()
  {
    bool changedMinimal = true;
    const double tRef = events_subset_.front().ts.toSec();

    for (int i = 0; i < events_subset_.size(); i++)
    {
      // For each motion warp to position to get coordinates
      std::vector<cv::Point2i> positions;
      std::vector<cv::Point2d> positionsDoubles;
      double eventsTotal = 0.;
      for (int num = 0; num < opts_.num_motions_; num++)
      {
        cv::Point2d evWarpedPt;
        warpEvent(velocities[num], events_subset_[i], tRef, &evWarpedPt);
        const int xx = static_cast<int>(evWarpedPt.x);
        const int yy = static_cast<int>(evWarpedPt.y);
        double dX = evWarpedPt.x - xx;
        double dY = evWarpedPt.y - yy;
        cv::Point2i evWarpedPtInteger(xx, yy);
        positions.push_back(evWarpedPtInteger);
        positionsDoubles.push_back(evWarpedPt);
        if (xx < 0 || yy < 0 || xx >= images_warped_[num].cols - 1 ||
            yy >= images_warped_[num].rows - 1)
        {
          continue;
        }

        eventsTotal +=
            ((1 - dX) * (1 - dY)) * images_warped_[num].at<double>(yy, xx);
        eventsTotal +=
            (dX * (1 - dY)) * images_warped_[num].at<double>(yy, xx + 1);
        eventsTotal +=
            ((1 - dX) * dY) * images_warped_[num].at<double>(yy + 1, xx);
        eventsTotal +=
            (dX * dY) * images_warped_[num].at<double>(yy + 1, xx + 1);
      }

      for (int num = 0; num < opts_.num_motions_; num++)
      {
        const int xx = positions[num].x;
        const int yy = positions[num].y;
        double dX = positionsDoubles[num].x - xx;
        double dY = positionsDoubles[num].y - yy;
        if (xx < 0 || yy < 0 || xx >= img_size_.width - 1 ||
            yy >= img_size_.height - 1)
        {
          probabilityVectors[num][i] = 0.;
        }
        else if (eventsTotal == 0)
        {
          probabilityVectors[num][i] = 1. / opts_.num_motions_;
        }
        else
        {
          double prob =
              ((1 - dX) * (1 - dY)) * images_warped_[num].at<double>(yy, xx);
          prob += (dX * (1 - dY)) * images_warped_[num].at<double>(yy, xx + 1);
          prob += ((1 - dX) * dY) * images_warped_[num].at<double>(yy + 1, xx);
          prob += (dX * dY) * images_warped_[num].at<double>(yy + 1, xx + 1);
          prob /= eventsTotal;
          double difference = abs(probabilityVectors[num][i] - prob);
          if (difference > 0.1) // if probabilities changed by more then 10 percent (it doesnt break the loop)
          {
            changedMinimal = false;
          }

          probabilityVectors[num][i] = prob;
        }
      }
    }
    return changedMinimal;
  }

  void GlobalFlowEstimator::eventsCallback(
      const dvs_msgs::EventArray::ConstPtr &msg)
  {
    // Assume events are sorted in time and event messages come in correct order.
    // Read events, split the events into packets of constant number of events,
    // generate an image from those events and save/publish that image.

    // Append events of current message to the queue
    for (const dvs_msgs::Event &ev : msg->events)
    {
      events_.push_back(ev);
    }

    // Set image size with the arrival of the first messsage
    if (img_size_.height == 0)
    {
      img_size_ = cv::Size(msg->width, msg->height);
    }

    static unsigned int packet_number = 0;
    static unsigned long total_event_count = 0;
    total_event_count += msg->events.size();

    // If there are enough events in the queue, get subset of events
    while (events_.size() >= idx_first_ev_ + opts_.num_events_per_image_)
    {
      probabilityVectors = {
          size_t(opts_.num_motions_),
          std::vector<double>(size_t(opts_.num_events_per_image_),
                              double(1. / double(opts_.num_motions_)))};
      images_warped_ = {
          size_t(opts_.num_motions_),
          cv::Mat(msg->height, msg->width, CV_64FC1, cv::Scalar(0.))};

      getSubsetOfEvents();
      //  Initial value of global flow
      if (packet_number == 0)
      {
        findInitialFlow();
        // calculateProbability();
      }
      packet_number++;

      if (opts_.verbose_ >= 1)
      {
        LOG(INFO) << "Packet # " << packet_number << "  event# "
                  << total_event_count;
      }

      // Process the events
      int iterations = 0;

      maximizeContrast();
      while (!calculateProbability() && iterations < 30)
      {
        if (opts_.verbose_ >= 2)
        {
          LOG(INFO) << "Iter: " << iterations << "\n";
          for (int i = 0; i < opts_.num_motions_; i++)
          {
            LOG(INFO) << "velocity " << i << ": " << velocities[i].x << " "
                      << velocities[i].y;
          }
        }
        maximizeContrast();
        iterations++;
      }

      bool velocityWasChanged = false;

      for (int j = 0; j < opts_.num_motions_; j++)
      {
        for (int k = j + 1; k < opts_.num_motions_; k++)
        {
          double percentageDifferenceX =
              (abs(velocities[j].x - velocities[k].x));
          double percentageDifferenceY =
              (abs(velocities[j].y - velocities[k].y));
          if (percentageDifferenceX <= opts_.max_percentage_difference &&
              percentageDifferenceY <= opts_.max_percentage_difference)
          {
            velocities[k].x = 0.;
            velocities[k].y = 0.;
            if (opts_.verbose_ > 1)
            {
              LOG(INFO) << "Set one velocity to 0 because they are similar";
            }
            velocityWasChanged = true;
          }
        }
        std::vector<std::vector<double>> temp2D;
        clusterData.push_back(temp2D);
      }
      // Calculate Probaility with the right motions if something was changed
      if (velocityWasChanged)
      {
        calculateProbability();
      }

      if (opts_.verbose_ >= 2)
      {
        // Count how many events are in each CLuster
        std::vector<int> numOfEventsPerCluster(
            size_t(opts_.num_events_per_image_), 0);

        for (int j = 0; j < events_subset_.size(); j++)
        {
          for (int i = 0; i < opts_.num_motions_; i++)
          {
            if ((probabilityVectors)[i][j] > 0.5)
            {
              numOfEventsPerCluster[i] += 1;
            }
          }
        }
        for (int i = 0; i < opts_.num_motions_; i++)
        {
          LOG(INFO) << "velocity " << i << ": " << velocities[i].x << " "
                    << velocities[i].y
                    << " Corresponding Events: " << numOfEventsPerCluster[i];
        }
      }

      // Compute publication timestamp: midpoint between timestamps of events in
      // the packet
      ros::Time timeFirst = events_subset_.front().ts;
      ros::Time timeLast = events_subset_.back().ts;
      ros::Duration timeDt = timeLast - timeFirst;
      time_packet_ = timeFirst + timeDt * 0.5;

      // Publish estimated motion parameters
      publishGlobalFlow();

      // Save / Publish image
      publishEventImage();

      // Slide the window, for next subset of events
      slideWindow();
    }
  }

  void GlobalFlowEstimator::publishEventImage()
  {
    /*
    if (image_pub_.getNumSubscribers() <= 0)
      return;
    */

    static cv::Mat image_original;
    static cv::Mat image_warped;
    static cv::Mat image_stacked;
    static cv::Mat event_image_warped_colored;

    // Options to visualize the image of raw events and the image of warped events
    // without blur

    // Compute image of events assuming zero flow (i.e., without motion
    // compensation)
    event_image_warped_colored =
        cv::Mat(img_size_.height, img_size_.width, CV_8UC3, cv::Scalar(0, 0, 0));

    // Compute image of events assuming zero flow (i.e., without motion
    // compensation)
    computeOriginalImage(events_subset_, img_size_, &image_original);

    // Warp events without sigma blur
    OptionsWarp optsWarpDisplay = opts_.opts_warp_;
    optsWarpDisplay.blur_sigma_ = 0.;
    for (int indexOfMotion = 0; indexOfMotion < opts_.num_motions_;
         indexOfMotion++)
    {
      computeImageOfWarpedEvents(velocities[indexOfMotion], events_subset_,
                                 img_size_, &images_warped_[indexOfMotion],
                                 optsWarpDisplay, indexOfMotion,
                                 &probabilityVectors);
    }

    max_events_at_Pixel_overall = 0;
    for (int y = 0; y < img_size_.height; y++)
    {
      for (int x = 0; x < img_size_.width; x++)
      {
        for (int indexOfMotion = 0; indexOfMotion < opts_.num_motions_;
             indexOfMotion++)
        {
          if (images_warped_[indexOfMotion].at<double>(y, x) >
              max_events_at_Pixel_overall)
          {
            max_events_at_Pixel_overall =
                images_warped_[indexOfMotion].at<double>(y, x);
          }
        }
      }
    }

    computeImageOfWarpedEventsColored(
        &velocities, &old_velocities, events_subset_, img_size_, &images_warped_,
        &event_image_warped_colored, opts_.num_motions_,
        max_events_at_Pixel_overall, &pixel_dict_, &clusterData,
        opts_.clustering_enabled_,
        opts_.pixel_displaying_and_clustering_threshold_);

    if (opts_.clustering_enabled_)
    {
      std::vector<std::vector<int>> labelsVector;
      int minLabel = 0;
      for (int i = 0; i < clusterData.size(); i++)
      {
        if (clusterData[i].size() == 0)
        {
          continue;
        }
        if (opts_.verbose_ > 1)
          LOG(INFO) << "start clustering number:" << i << " with data size: " << clusterData[i].size();
        Hdbscan hdbscan("");
        hdbscan.dataset = clusterData[i];
        hdbscan.execute(100, 100, "Euclidean");
        if (opts_.verbose_ > 1)
          LOG(INFO) << "Clustering contains " << hdbscan.numClusters_ << " clusters with " << hdbscan.noisyPoints_ << " noise Points and" << clusterData[i].size() << "total Points.";

        for (int &num : hdbscan.normalizedLabels_)
        {
          if (num > 0)
            num += minLabel;
          if (num == -1)
            num = 0;
        }
        labelsVector.push_back(hdbscan.normalizedLabels_);
        minLabel += hdbscan.numClusters_;
        if (opts_.verbose_ > 1)
          LOG(INFO) << "end clustering number:" << i;
      }
      for (int j = 0; j < labelsVector.size(); j++)
      {
        for (int i = 0; i < labelsVector[j].size(); i++)
        {
          int y = static_cast<int>(clusterData[j][i][0]);
          int x = static_cast<int>(clusterData[j][i][1]);
          double value = pixel_dict_[cv::Point2i(y, x)].highestValue;
          int motionIndex = labelsVector[j][i];
          event_image_warped_colored.at<cv::Vec3b>(y, x) = getColorFromIndex(motionIndex, minLabel + 1, value);
        }
      }
    }
    concatHorizontal(image_original, event_image_warped_colored, &image_stacked);
    cv::bitwise_not(image_stacked, image_stacked);

    // Publish images (without and with motion compensation)
    cv_event_image_.encoding = "bgr8";
    cv_event_image_.image = image_stacked;
    auto aux = cv_event_image_.toImageMsg();
    aux->header.stamp = time_packet_;
    image_pub_.publish(aux);

    // Set the old velocities to the curr
    old_velocities = velocities;
    clusterData.clear();
  }

  /** \brief Select vector(s) of events from the queue storing the events
   * \note In a separte function to remove clutter from main function
   */
  void GlobalFlowEstimator::getSubsetOfEvents()
  {
    // LOG(INFO) << "------------_" <<  idx_first_ev_ << "-----------------";
    // LOG(INFO) << "------------_" << idx_first_ev_+ opts_.num_events_per_image_
    // << "-----------------";
    events_subset_ = std::vector<dvs_msgs::Event>(
        events_.begin() + idx_first_ev_,
        events_.begin() + idx_first_ev_ + opts_.num_events_per_image_);
    // LOG(INFO) << "events_ after getSubset_: " <<  events_.size() <<
    // "-----------------"; LOG(INFO) << "events_subset_ after getSubset_: " <<
    // events_subset_.size() << "-----------------";
  }

  /** \brief Slide the window and remove those old events from the queue
   * \note In a separte function to remove clutter from main function
   */
  void GlobalFlowEstimator::slideWindow()
  {
    if (opts_.num_events_slide_ <= events_.size())
    {
      events_.erase(events_.begin(), events_.begin() + opts_.num_events_slide_);
      idx_first_ev_ = 0;
    }
    else
    {
      idx_first_ev_ += opts_.num_events_slide_;
    }
  }

} // namespace dvs_global_flow