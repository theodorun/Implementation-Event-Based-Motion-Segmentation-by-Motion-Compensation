#pragma once

#include "dvs_global_flow/util.h"

#include <glog/logging.h>

/**
 * \brief Concatenate two matrices horizontally
 * \param[in] Mat A and B
 * \param[out] Mat C = [A, B]
 */
void concatHorizontal(const cv::Mat &a, const cv::Mat &b, cv::Mat *c)
{
  CHECK_EQ(a.rows, b.rows) << "Input arguments must have same number of rows";
  CHECK_EQ(a.type(), b.type()) << "Input arguments must have the same type";
  cv::hconcat(a, b, *c);
}

void clipping(cv::Mat &src, cv::Mat &dst, float discard_percentage)
{
  float min_val;
  float max_val;
  float discard_threshold =
      min_val + (max_val - min_val) * (discard_percentage / 100.0f);
  cv::Mat clipped;
  cv::threshold(src, clipped, discard_threshold, 0, cv::THRESH_TOZERO);
}