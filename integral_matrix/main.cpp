#include <iostream>

#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat sourceMatrix = (cv::Mat_<uint8_t>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    cv::Mat integralMatrix;
    cv::integral(sourceMatrix, integralMatrix);

    std::cout << "source matrix: " << std::endl
              << format(sourceMatrix, cv::Formatter::FMT_NUMPY) << std::endl
              << std::endl
              << "integral_matrix: " << std::endl
              << format(integralMatrix, cv::Formatter::FMT_NUMPY) << std::endl;

    return EXIT_SUCCESS;
}