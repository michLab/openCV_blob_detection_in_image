#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main()
{
    // Load an image:
    std::string path_to_images = "";
    Mat image = imread(path_to_images + "blobs.jpg");
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("Image", image);
    imshow("Image gray", gray);

    // Create blob detector
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
    vector<KeyPoint> keypoints;

    // Detect keypoints - each blob:
    detector->detect(gray, keypoints);
    Mat image_with_keypoints;
    drawKeypoints(image, keypoints, image_with_keypoints, Scalar(0,0,255),
                  DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    int number_of_blobs = static_cast<int>(keypoints.size());
    string message = "Total number of blobs = " + to_string(number_of_blobs);
    putText(image_with_keypoints, message, Point(20,550), FONT_HERSHEY_COMPLEX, 1, Scalar(100,0,255),2);
    imshow("Blobs using default parameters", image_with_keypoints);

    // Filtering with params:
    SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 100;

    params.filterByCircularity = true;
    params.minCircularity = 0.9f;

    params.filterByInertia = true;
    params.minInertiaRatio = 0.01f;

    params.filterByConvexity = true;
    params.minConvexity = 0.2f;

    // Detect only circles:
    Ptr<SimpleBlobDetector> detector_params = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints_params;
    detector_params->detect(gray, keypoints_params);

    // Drwaing image with detected blobs:
    Mat image_with_keypoints_params;
    drawKeypoints(image, keypoints_params, image_with_keypoints_params, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    int number_of_blobs_params = static_cast<int>(keypoints_params.size());
    string message_params = "Total number of blobs = " + to_string(number_of_blobs_params);
    putText(image_with_keypoints_params, message_params, Point(20,550), FONT_HERSHEY_COMPLEX, 1, Scalar(100,0,255),2);
    imshow("Blobs using my parameters", image_with_keypoints_params);

    waitKey(0);
    return 0;
}
