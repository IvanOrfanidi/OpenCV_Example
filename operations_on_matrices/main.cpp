#include <iostream>

#ifndef CUDA_ENABLE
#define CUDA_ENABLE false
#endif

#if (CUDA_ENABLE)
#include <opencv2/cudaarithm.hpp>
#endif
#include <opencv2/opencv.hpp>

constexpr int WIDTH = 1000;
constexpr int HEIGHT = 1000;

int main()
{
    int64 start;
    int64 end;

#if (CUDA_ENABLE)
    bool cudaEnable = false;
    if (cv::cuda::getCudaEnabledDeviceCount() != 0) {
        cv::cuda::DeviceInfo deviceInfo;
        if (deviceInfo.isCompatible()) {
            cudaEnable = true;
        }
    }
#endif

    cv::Mat MA(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::randu(MA, cv::Scalar::all(-10), cv::Scalar::all(10));
    cv::Mat MB(cv::Size(WIDTH, HEIGHT), CV_32F);
    cv::randu(MB, cv::Scalar::all(-10), cv::Scalar::all(10));
    cv::Mat MC;

    start = cv::getTickCount();
    MC = MA * MB;
    end = cv::getTickCount();
    std::cout << "multiplication time:                      " << ((end - start) * (1000.0f / cv::getTickFrequency())) << " ms" << std::endl;

    start = cv::getTickCount();
    MC = MA + MB;
    end = cv::getTickCount();
    std::cout << "summation time:                           " << ((end - start) * (1000.0f / cv::getTickFrequency())) << " ms" << std::endl;

    start = cv::getTickCount();
    cv::sum(cv::sum(MC))[0];
    end = cv::getTickCount();
    std::cout << "time of summation of matrix elements:     " << ((end - start) * (1000.0f / cv::getTickFrequency())) << " ms" << std::endl;

#if (CUDA_ENABLE)
    std::cout << std::endl
              << "with CUDA:" << std::endl;

    if (cudaEnable) {
        cv::cuda::GpuMat gpuMA(MA);
        cv::cuda::GpuMat gpuMB(MB);
        cv::cuda::GpuMat gpuMD;

        start = cv::getTickCount();
        cv::cuda::gemm(gpuMA, gpuMB, 1.0, cv::cuda::GpuMat(), 0.0, gpuMD);
        end = cv::getTickCount();
        std::cout << "multiplication time:                      " << ((end - start) * (1000.0f / cv::getTickFrequency())) << " ms" << std::endl;

        start = cv::getTickCount();
        cv::cuda::add(gpuMA, gpuMB, gpuMD);
        end = cv::getTickCount();
        std::cout << "summation time:                           " << ((end - start) * (1000.0f / cv::getTickFrequency())) << " ms" << std::endl;

        start = cv::getTickCount();
        cv::cuda::sum(cv::cuda::sum(gpuMD))[0];
        end = cv::getTickCount();
        std::cout << "time of summation of matrix elements:     " << ((end - start) * (1000.0f / cv::getTickFrequency())) << " ms" << std::endl;
    } else {
        std::cout << std::endl
                  << "CUDA is not supported" << std::endl;
    }
#else
    std::cout << std::endl
              << "CUDA is disabled" << std::endl;
#endif

    return EXIT_SUCCESS;
}