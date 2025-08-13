#include <iostream>
#include <opencv2/opencv.hpp>


int main ()
{
    std::cout<<"Hello Worlllll ld\n";


    cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
    cv::putText(image, "Hello OpenCV", {50, 200}, cv::FONT_HERSHEY_SIMPLEX, 1, {255,255,255}, 2); 

    cv::imshow("Test Window", image);
    cv::waitKey(0);
    return 0;
}

