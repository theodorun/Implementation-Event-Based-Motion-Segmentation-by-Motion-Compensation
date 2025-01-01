#include <glog/logging.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

#include <cmath>
#include <opencv2/highgui/highgui.hpp>

#include "dvs_global_flow/global_flow_estimator.h"
#include "dvs_global_flow/image_warped_events.h"
#include "dvs_global_flow/numerical_deriv.h"

double contrast_MeanSquare(const cv::Mat &image)
{
  // Compute mean square value of the image
  double contrast = cv::norm(image, cv::NORM_L2SQR) /
                    static_cast<double>(image.rows * image.cols);
  return contrast;
}
double contrast_Pow_8_NonZero_Greater_0(const cv::Mat &image)
{
  // Compute mean square value of the image

  cv::Mat tmpMat = image.mul(image);
  tmpMat = tmpMat.mul(tmpMat);
  tmpMat = tmpMat.mul(tmpMat);
  double nonZeroPixels = 0;
  double totalValues = 0;
  for (int row = 0; row < image.rows; row++)
  {
    for (int col = 0; col < image.cols; col++)
    {
      if (tmpMat.at<double>(row, col) > 0)
      {
        nonZeroPixels++;
        totalValues += tmpMat.at<double>(row, col);
      }
    }
  }
  if (nonZeroPixels == 0)
  {
    return 0.;
  }
  // cv::imshow("some", tmp_mat);
  // cv::waitKey();

  //  used to reward better warps even with fewer events (as value inc
  //  exponential)
  const double contrast = totalValues / nonZeroPixels;

  return contrast;
}

double contrast_Pow_4_NonZero_Greater_0(const cv::Mat &image)
{
  // Compute mean square value of the image

  cv::Mat tmpMat = image.mul(image);
  tmpMat = tmpMat.mul(tmpMat);
  double nonZeroPixels = 0;
  double totalValues = 0;
  for (int row = 0; row < image.rows; row++)
  {
    for (int col = 0; col < image.cols; col++)
    {
      if (tmpMat.at<double>(row, col) > 0)
      {
        nonZeroPixels++;
        totalValues += tmpMat.at<double>(row, col);
      }
    }
  }
  if (nonZeroPixels == 0)
  {
    return 0.;
  }
  //  used to reward better warps even with fewer events (as value inc
  //  exponential)
  const double contrast = totalValues / nonZeroPixels;

  return contrast;
}

double contrast_Variance(const cv::Mat &image)
{
  // Compute variance of the image
  double contrast;
  cv::Mat mean;
  cv::Mat stddev;
  cv::meanStdDev(image, mean, stddev);
  contrast = stddev.at<double>(0) * stddev.at<double>(0);
  return contrast;
}

double computeContrast(const cv::Mat &image, const int contrast_measure)
{
  // Branch according to contrast measure
  double contrast;
  switch (contrast_measure)
  {
  case MEAN_SQUARE_CONTRAST:
    contrast = contrast_MeanSquare(image);
    break;
  case MEAN_POWER_CONTRAST_NON_ZERO:
    contrast = contrast_Pow_8_NonZero_Greater_0(image);
    break;
  case MEAN_SQUARE_CONTRAST_NON_ZERO:
    contrast = contrast_Pow_4_NonZero_Greater_0(image);
  default:
    contrast = contrast_Variance(image);
    break;
  }

  return contrast;
}

/**
 * @brief Auxiliary data structure for optimization algorithm
 */
typedef struct
{
  std::vector<dvs_msgs::Event> *poEvents_subset;
  cv::Size *img_size;
  dvs_global_flow::OptionsMethod *opts;
  int motionIndex;
  cv::Mat *warpedImage;
  std::vector<std::vector<double>> *probabilityVectors;
} AuxdataBestFlow;

/**
 * @brief Main function used by optimization algorithm.
 * Maximize contrast, or equivalently, minimize (-contrast)
 */
double contrast_f_numerical(const gsl_vector *v, void *adata)
{
  // Extract auxiliary data of cost function
  AuxdataBestFlow *poAuxData = (AuxdataBestFlow *)adata;

  // Parameter vector (from GSL to OpenCV)
  cv::Point2d vel(gsl_vector_get(v, 0), gsl_vector_get(v, 1));

  // Compute cost
  double contrast = 0;

  computeImageOfWarpedEvents(
      vel, *(poAuxData->poEvents_subset), *(poAuxData->img_size),
      (poAuxData->warpedImage), poAuxData->opts->opts_warp_,
      poAuxData->motionIndex, poAuxData->probabilityVectors);

  contrast = computeContrast(*(poAuxData->warpedImage),
                             poAuxData->opts->contrast_measure_);

  return -contrast;
}

void contrast_fdf_numerical(const gsl_vector *v, void *adata, double *f,
                            gsl_vector *df)
{
  // Finite difference approximation
  *f = vs_gsl_Gradient_ForwardDiff(v, adata, contrast_f_numerical, df, 1e0);
}

void contrast_df_numerical(const gsl_vector *v, void *adata, gsl_vector *df)
{
  double cost;
  contrast_fdf_numerical(v, adata, &cost, df);
}

namespace dvs_global_flow

{

  void GlobalFlowEstimator::findInitialFlow()
  {
    if (opts_.verbose_)
      LOG(INFO) << "finding Initial Flow ---------------------";
    // init velocities with zero and wapred images with velocities
    for (int indexOfMotion = 0; indexOfMotion < opts_.num_motions_;
         indexOfMotion++)
    {
      computeImageOfWarpedEvents(cv::Point2d(0, 0), events_subset_, img_size_,
                                 &images_warped_[indexOfMotion],
                                 opts_.opts_warp_, indexOfMotion,
                                 &probabilityVectors);
    }
    // create pool of motions
    std::vector<cv::Point2d> poolOfMotions;
    poolOfMotions.push_back(cv::Point2d(0, 0));

    for (int num = 0; num < opts_.number_of_directions_; ++num)
    {
      const double angle = (2 * M_PI) / opts_.number_of_directions_ * num;
      const double xDirection = opts_.pixel_distance_ * cos(angle);
      const double yDirection = opts_.pixel_distance_ * sin(angle);
      for (int multiplier = 1; multiplier <= opts_.number_of_pool_steps_;
           multiplier += 1)
      {
        poolOfMotions.push_back(
            cv::Point2d(xDirection * multiplier, yDirection * multiplier));
      }
    }

    std::vector<dvs_msgs::Event> remainingEvents = events_subset_;

    double tRef = events_subset_.front().ts.toSec();
    if (opts_.verbose_ > 1)
      LOG(INFO) << "t_ref:" << tRef;

    for (int indexOfMotion = 0; indexOfMotion < opts_.num_motions_; indexOfMotion++)
    {
      std::vector<double> contrastPoolMotions;
      // calculate contrast for every pool motion
      std::vector<cv::Mat> imagesOfPoolMotions;
      // We compute this again for each motion to prevent that the initalize takes
      // two motions witch are close by
      for (int j = 0; j < poolOfMotions.size(); j++)
      {
        cv::Mat imageOfPoolMotionTemp;
        computeImageOfWarpedEvents(poolOfMotions[j], remainingEvents,
                                   img_size_, &imageOfPoolMotionTemp,
                                   opts_.opts_warp_, indexOfMotion, &probabilityVectors);
        double contrast =
            contrast_Pow_4_NonZero_Greater_0(imageOfPoolMotionTemp);
        contrastPoolMotions.push_back(contrast);
        imagesOfPoolMotions.push_back(imageOfPoolMotionTemp);
      }

      // get pool motion with highest contrast auto maxElementIterator =
      // std::max_element(vec.begin(), vec.end()); Calculate the index of the
      // maximum element
      auto maxElementIterator = std::max_element(contrastPoolMotions.begin(),
                                                 contrastPoolMotions.end());
      int maxIndex =
          std::distance(contrastPoolMotions.begin(), maxElementIterator);

      computeImageOfWarpedEvents(poolOfMotions[maxIndex], remainingEvents,
                                 img_size_, &imagesOfPoolMotions[maxIndex],
                                 opts_.opts_warp_, indexOfMotion, &probabilityVectors);
      // cv::imshow("Events with velocity for max contrast",
      //            imagesOfPoolMotions[max_index]);
      // cv::waitKey();
      //  Calculate the Contrast for the maximum contrast
      double maxContrast = contrastPoolMotions[maxIndex];
      if (opts_.verbose_ > 2)
        LOG(INFO) << "max sqaure contrast non zero of pool motions: "
                  << maxContrast;
      velocities[indexOfMotion] = poolOfMotions[maxIndex];

      // Delete the motion with the highes contrast from the pool
      poolOfMotions.erase(poolOfMotions.begin() + maxIndex);

      //  go trough events and check if local contrast is higher then global
      //  contrast
      std::vector<dvs_msgs::Event> usedEventsThisIteration;
      std::vector<dvs_msgs::Event> viableEventsNextIteration;

      // Push tref event along
      viableEventsNextIteration.push_back(events_subset_[0]);
      // Push tref event along
      usedEventsThisIteration.push_back(events_subset_[0]);

      for (int indexEvent = 0; indexEvent < remainingEvents.size(); indexEvent++)
      {
        cv::Point2d evWarpedPt;
        warpEvent(velocities[indexOfMotion], remainingEvents[indexEvent], tRef, &evWarpedPt);
        const int xx = static_cast<int>(evWarpedPt.x);
        const int yy = static_cast<int>(evWarpedPt.y);
        double dX = evWarpedPt.x - xx;
        double dY = evWarpedPt.y - yy;
        if (0 <= xx && xx < img_size_.width - 1 && 0 <= yy &&
            yy < img_size_.height - 1)
        {
          double localContrast =
              ((1 - dX) * (1 - dY)) *
              imagesOfPoolMotions[maxIndex].at<double>(yy, xx); //* polarity
          localContrast +=
              (dX * (1 - dY)) *
              imagesOfPoolMotions[maxIndex].at<double>(yy, xx + 1);
          localContrast +=
              ((1 - dX) * dY) *
              imagesOfPoolMotions[maxIndex].at<double>(yy + 1, xx);
          localContrast +=
              (dX * dY) *
              imagesOfPoolMotions[maxIndex].at<double>(yy + 1, xx + 1);
          localContrast = pow(localContrast, 4);
          if (localContrast < maxContrast)
          {
            viableEventsNextIteration.push_back(remainingEvents[indexEvent]);
          }
          else
          {
            usedEventsThisIteration.push_back(remainingEvents[indexEvent]);
          }
        }
        else
        {
          viableEventsNextIteration.push_back(remainingEvents[indexEvent]);
        }
      }
      if (opts_.verbose_ > 2)
      {

        cv::Mat temp_image_of_motion;
        computeImageOfWarpedEvents(velocities[indexOfMotion], usedEventsThisIteration,
                                   img_size_, &temp_image_of_motion,
                                   opts_.opts_warp_, indexOfMotion, &probabilityVectors);
        cv::imshow("Used events", temp_image_of_motion);
        cv::waitKey();

        LOG(INFO) << "Number of associated events: "
                  << remainingEvents.size() -
                         viableEventsNextIteration.size();
      }
      remainingEvents = viableEventsNextIteration;
    }

    if (opts_.verbose_)
    {
      for (int i = 0; i < opts_.num_motions_; i++)
      {
        LOG(INFO) << "velocity " << i << ": " << velocities[i].x << " "
                  << velocities[i].y;
      }

      LOG(INFO) << "finding Initial Flow end ---------------------";
    }
    // Initialize for the first iteration the old velocities
    old_velocities = velocities;
  }

  // Maximize the contrast with respect to the global flow
  void GlobalFlowEstimator::maximizeContrast()
  {
    double finalCost;

    const gsl_multimin_fdfminimizer_type *solverType;
    solverType =
        gsl_multimin_fdfminimizer_conjugate_fr; // Fletcher-Reeves conjugate
                                                // gradient algorithm
    // Auxiliary data for the cost function
    AuxdataBestFlow oAuxdata;

    oAuxdata.poEvents_subset = &events_subset_;
    oAuxdata.img_size = &img_size_;
    oAuxdata.opts = &opts_;
    oAuxdata.probabilityVectors = &probabilityVectors;
    gsl_multimin_function_fdf solverInfo;

    const int numParams = 2;                 // Size of global flow
    solverInfo.n = numParams;                // Size of the parameter vector
    solverInfo.f = contrast_f_numerical;     // Cost function
    solverInfo.df = contrast_df_numerical;   // Gradient of cost function
    solverInfo.fdf = contrast_fdf_numerical; // Cost and gradient functions

    gsl_multimin_fdfminimizer *solver =
        gsl_multimin_fdfminimizer_alloc(solverType, numParams);
    const double initialStepSize = 10; // Todo ggf anpassen
    double tol = 0.05;
    gsl_vector *vx = gsl_vector_alloc(numParams);
    // Parameters for optim
    const double epsabsGrad = 1e-4;
    const double tolfun = 1e-2;
    double costNew = 1e9;
    double costOld = 1e9;
    const int numMaxLineSearches = 1;
    for (int num = 0; num < opts_.num_motions_; num++)
    {
      oAuxdata.motionIndex = num;
      oAuxdata.warpedImage = &images_warped_[num];

      // Routines to compute the cost function and its derivatives

      solverInfo.params = &oAuxdata; // Auxiliary data

      // Initial parameter vector
      gsl_vector *vx = gsl_vector_alloc(numParams);
      cv::Point2d oldVelocity = velocities[num];
      gsl_vector_set(vx, 0, velocities[num].x);
      gsl_vector_set(vx, 1, velocities[num].y);

      // Initialize solver

      gsl_multimin_fdfminimizer_set(solver, &solverInfo, vx, initialStepSize,
                                    tol);

      const double initialCost = solver->f;

      // ITERATE

      int status;

      size_t iter = 0;
      if (opts_.verbose_ >= 2)
      {
        LOG(INFO) << "Optimization. Solver type = " << solverType->name;
        LOG(INFO) << "iter=" << std::setw(3) << iter << "  vel=["
                  << gsl_vector_get(solver->x, 0) << " "
                  << gsl_vector_get(solver->x, 1)
                  << "]  cost=" << std::setprecision(8) << solver->f;
      }

      do
      {
        iter++;
        costOld = costNew;
        status = gsl_multimin_fdfminimizer_iterate(solver);
        // status == GLS_SUCCESS (0) means that the iteration reduced the function
        // value

        if (opts_.verbose_ >= 2)
        {
          LOG(INFO) << "iter=" << std::setw(3) << iter << "  vel=["
                    << gsl_vector_get(solver->x, 0) << " "
                    << gsl_vector_get(solver->x, 1)
                    << "]  cost=" << std::setprecision(8) << solver->f;
        }

        if (status == GSL_SUCCESS)
        {
          // Test convergence due to stagnation in the value of the function
          costNew = gsl_multimin_fdfminimizer_minimum(solver);
          if (fabs(1 - costNew / (costOld + 1e-7)) < tolfun)
          {
            if (opts_.verbose_ >= 3)
              LOG(INFO) << "progress tolerance reached.";
            break;
          }
          else
          {
            // Succesful step set back to false because we might iterate again
            status = GSL_CONTINUE;
          }
        }

        // Test convergence due to absolute norm of the gradient
        if (GSL_SUCCESS ==
            gsl_multimin_test_gradient(solver->gradient, epsabsGrad))
        {
          if (opts_.verbose_ >= 3)
            LOG(INFO) << "gradient tolerance reached.";
          break;
        }

        if (status != GSL_CONTINUE)
        {
          // The iteration was not successful (did not reduce the function value)
          if (opts_.verbose_ >= 3)
            LOG(INFO) << "stopped iteration; status = " << status;
          break;
        }
      } while (status == GSL_CONTINUE && iter < numMaxLineSearches);

      // SAVE RESULTS (best global flow velocity)

      // Convert from GSL to OpenCV format
      gsl_vector *finalX = gsl_multimin_fdfminimizer_x(solver);

      velocities[num].x = gsl_vector_get(finalX, 0);
      velocities[num].y = gsl_vector_get(finalX, 1);

      if (opts_.verbose_ >= 3)
        LOG(INFO) << "Velocities  " << velocities[num].x << ", "
                  << velocities[num].y;
      finalCost = gsl_multimin_fdfminimizer_minimum(solver);

      if (opts_.verbose_ >= 2)
      {
        LOG(INFO) << "--- Initial cost = " << std::setprecision(8)
                  << initialCost;
        LOG(INFO) << "--- Final cost   = " << std::setprecision(8) << finalCost;
      }
    }
    // Release memory used during optimization
    gsl_multimin_fdfminimizer_free(solver);
    gsl_vector_free(vx);
    return;
  }
  void GlobalFlowEstimator::maximizeContrastOneMotion(
      cv::Point2d &velocity,
      std::vector<dvs_msgs::Event> *events_subset_remaining)
  {
    double finalCost;

    const gsl_multimin_fdfminimizer_type *solverType;
    solverType =
        gsl_multimin_fdfminimizer_conjugate_fr; // Fletcher-Reeves conjugate
                                                // gradient algorithm
    // Auxiliary data for the cost function
    AuxdataBestFlow oAuxdata;

    oAuxdata.poEvents_subset = events_subset_remaining;
    oAuxdata.img_size = &img_size_;
    oAuxdata.opts = &opts_;
    oAuxdata.probabilityVectors = &probabilityVectors;
    gsl_multimin_function_fdf solverInfo;

    const int numParams = 2;                 // Size of global flow
    solverInfo.n = numParams;                // Size of the parameter vector
    solverInfo.f = contrast_f_numerical;     // Cost function
    solverInfo.df = contrast_df_numerical;   // Gradient of cost function
    solverInfo.fdf = contrast_fdf_numerical; // Cost and gradient functions

    gsl_multimin_fdfminimizer *solver =
        gsl_multimin_fdfminimizer_alloc(solverType, numParams);
    const double initialStepSize = 10; // Todo ggf anpassen
    double tol = 0.05;
    gsl_vector *vx = gsl_vector_alloc(numParams);
    // Parameters for optimization
    const double epsabsGrad = 1e-5;
    const double tolfun = 1e-3;
    double costNew = 1e9;
    double costOld = 1e9;
    const int numMaxLineSearches = 10;
    oAuxdata.motionIndex = 0;
    cv::Mat temp;
    oAuxdata.warpedImage = &temp;

    // Routines to compute the cost function and its derivatives

    solverInfo.params = &oAuxdata; // Auxiliary data

    // Initial parameter vector
    cv::Point2d oldVelocity = velocity;
    gsl_vector_set(vx, 0, velocity.x);
    gsl_vector_set(vx, 1, velocity.y);

    // Initialize solver

    gsl_multimin_fdfminimizer_set(solver, &solverInfo, vx, initialStepSize,
                                  tol);

    const double initialCost = solver->f;

    // ITERATE

    int status;

    size_t iter = 0;
    if (opts_.verbose_ >= 2)
    {
      LOG(INFO) << "Optimization. Solver type = " << solverType->name;
      LOG(INFO) << "iter=" << std::setw(3) << iter << "  vel=["
                << gsl_vector_get(solver->x, 0) << " "
                << gsl_vector_get(solver->x, 1)
                << "]  cost=" << std::setprecision(8) << solver->f;
    }

    do
    {
      iter++;
      costOld = costNew;
      status = gsl_multimin_fdfminimizer_iterate(solver);
      // status == GLS_SUCCESS (0) means that the iteration reduced the function
      // value

      if (opts_.verbose_ >= 1)
      {
        LOG(INFO) << "iter=" << std::setw(3) << iter << "  vel=["
                  << gsl_vector_get(solver->x, 0) << " "
                  << gsl_vector_get(solver->x, 1)
                  << "]  cost=" << std::setprecision(8) << solver->f;
      }

      if (status == GSL_SUCCESS)
      {
        // Test convergence due to stagnation in the value of the function
        costNew = gsl_multimin_fdfminimizer_minimum(solver);
        if (fabs(1 - costNew / (costOld + 1e-7)) < tolfun)
        {
          if (opts_.verbose_ >= 1)
            LOG(INFO) << "progress tolerance reached.";
          break;
        }
        else
        {
          // Succesful step set back to false because we might iterate again
          status = GSL_CONTINUE;
        }
      }

      // Test convergence due to absolute norm of the gradient
      if (GSL_SUCCESS ==
          gsl_multimin_test_gradient(solver->gradient, epsabsGrad))
      {
        if (opts_.verbose_ >= 1)
          LOG(INFO) << "gradient tolerance reached.";
        break;
      }

      if (status != GSL_CONTINUE)
      {
        // The iteration was not successful (did not reduce the function value)
        if (opts_.verbose_ >= 1)
          LOG(INFO) << "stopped iteration; status = " << status;
        break;
      }
    } while (status == GSL_CONTINUE && iter < numMaxLineSearches);

    // SAVE RESULTS (best global flow velocity)

    // Convert from GSL to OpenCV format
    gsl_vector *finalX = gsl_multimin_fdfminimizer_x(solver);

    velocity.x = gsl_vector_get(finalX, 0);
    velocity.y = gsl_vector_get(finalX, 1);

    if (opts_.verbose_ >= 3)
      LOG(INFO) << "Velocities  " << velocity.x << ", " << velocity.y;
    finalCost = gsl_multimin_fdfminimizer_minimum(solver);

    if (opts_.verbose_ >= 2)
    {
      LOG(INFO) << "--- Initial cost = " << std::setprecision(8) << initialCost;
      LOG(INFO) << "--- Final cost   = " << std::setprecision(8) << finalCost;
    }

    // Release memory used during optimization
    gsl_multimin_fdfminimizer_free(solver);
    gsl_vector_free(vx);
  }
} // namespace dvs_global_flow