#include <iostream>

#include <opencv2/opencv.hpp>

int main()
{
    const std::vector<std::string> pathsToImages = {
        "img/mountain.jpg",
        "img/wood.jpg"
    };

    std::vector<cv::Mat> images;
    images.reserve(pathsToImages.size());

    for (const auto& pathToImage : pathsToImages) {
        const cv::Mat image = cv::imread(pathToImage);
        if (image.empty()) {
            return EXIT_FAILURE;
        }
        images.push_back(std::move(image));
    }

    cv::Mat gaussianBlur;
    cv::Mat medianBlur;
    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Size ksize(5, 5); // Gaussian kernel size
        cv::GaussianBlur(images[i], gaussianBlur, ksize, 3, 3);
        cv::medianBlur(images[i], medianBlur, 3);
        cv::imshow(pathsToImages[i] + " (original)", images[i]);
        cv::imshow(pathsToImages[i] + " (gaussian)", gaussianBlur);
        cv::imshow(pathsToImages[i] + " (median)", medianBlur);
    }
    cv::waitKey(0);
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}