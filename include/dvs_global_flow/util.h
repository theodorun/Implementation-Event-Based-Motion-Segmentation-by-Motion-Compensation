#pragma once

#include <opencv2/opencv.hpp>

void concatHorizontal(const cv::Mat& A, const cv::Mat& B, cv::Mat* C);

void clipping(cv::Mat& src, cv::Mat& dst, float discardPercentage = 5.0f);