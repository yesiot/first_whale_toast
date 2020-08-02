#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


const uint32_t histogram_scale = 4;
const uint32_t nr_h_values = 180;
const uint32_t h_value_scale = 3;
const uint32_t nr_s_values = 256;
const uint32_t s_value_scale = 4;
const uint32_t threshold = 10;

int main() {

    cv::VideoCapture cap;

    cv::Mat frame;
    cv::Mat frame_hsv;

    cv::Mat hist;
    std::vector<float> ranges = {0, nr_h_values, 0, nr_s_values};
    std::vector<int> histSize = {nr_h_values / h_value_scale, nr_s_values / s_value_scale};
    std::vector<int> channels = {0, 1};

    cv::Mat hist_frame(histSize[1] * histogram_scale, histSize[0] * histogram_scale, CV_8UC3);
    cap.open(0, cv::CAP_ANY);

    while (true) {

        cap.read(frame);
        cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);

        std::vector<cv::Mat> mats = { frame_hsv };
        cv::calcHist(mats, channels, cv::noArray(), hist, histSize, ranges);
        cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

        for (auto h = 0; h < histSize[0]; h++) {
            for (auto s = 0; s < histSize[1]; s++) {
                auto hist_val = hist.at<float>(h, s);

                auto res = hist_val <=> threshold;

                if (res > 0) {
                    cv::rectangle(hist_frame,
                                  cv::Rect(h * histogram_scale, s * histogram_scale, histogram_scale, histogram_scale),
                                  cv::Scalar(0, 255, hist_val),
                                  -1);
                } else if (res == 0) {
                    cv::rectangle(hist_frame,
                                  cv::Rect(h * histogram_scale, s * histogram_scale, histogram_scale, histogram_scale),
                                  cv::Scalar(60, 255, hist_val),
                                  -1);
                }
                else {
                    cv::rectangle(hist_frame,
                            cv::Rect(h * histogram_scale, s * histogram_scale, histogram_scale, histogram_scale),
                            cv::Scalar(h * h_value_scale, s * s_value_scale, 128),
                            -1);
                }
            }
        }

        cv::cvtColor(hist_frame, hist_frame, cv::COLOR_HSV2BGR);

        imshow("frame", frame);
        imshow("histogram", hist_frame);

        auto ret = cv::waitKey(5);
        if (ret != -1 && ret != 255) {
            break;
        }
    }

    return 0;
}
