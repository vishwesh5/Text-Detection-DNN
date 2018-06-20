// Import libraries
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;

// Command line arguments parser
const char* keys =
    "{ help  h     | | Print help message. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
    "{ model m     | | Path to a binary .pb file contains trained network.}"
    "{ width       | 320 | Preprocess input image by resizing to a specific width. It should be multiple by 32. }"
    "{ height      | 320 | Preprocess input image by resizing to a specific height. It should be multiple by 32. }"
    "{ thr         | 0.5 | Confidence threshold. }"
    "{ nms         | 0.4 | Non-maximum suppression threshold. }";

// Decode function
void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
            std::vector<RotatedRect>& detections, std::vector<float>& confidences);

int main(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run TensorFlow implementation (https://github.com/argman/EAST) of "
                  "EAST: An Efficient and Accurate Scene Text Detector (https://arxiv.org/abs/1704.03155v2)");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    // Confidence threshold
    float confThreshold = parser.get<float>("thr");
    // Non-maximum suppression threshold
    float nmsThreshold = parser.get<float>("nms");
    // Input width
    int inpWidth = parser.get<int>("width");
    // Input height
    int inpHeight = parser.get<int>("height");
    // Ensure that the "model" is present
    CV_Assert(parser.has("model"));
    // Get "model" name
    String model = parser.get<String>("model");

    // Load network.
    // Automatically detects configuration
    // and framework based on model name
    Net net = readNet(model);

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        // If the input argument is provided, use it
        cap.open(parser.get<String>("input"));
    else
        // Otherwise, start camera stream
        cap.open(0);
    
    // download EAST using 
    // https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
    static const std::string kWinName = "EAST: An Efficient and Accurate Scene Text Detector";
    // Create a new named window
    namedWindow(kWinName, WINDOW_NORMAL);
    
    // Output images
    std::vector<Mat> outs;
    // Names of output images
    std::vector<String> outNames(2);
    outNames[0] = "feature_fusion/Conv_7/Sigmoid";
    outNames[1] = "feature_fusion/concat_3";

    Mat frame, blob;
    // Read or take input
    while (waitKey(1) < 0)
    {
        // Load an image frame
        cap >> frame;
        // If image frame is empty, stop
        if (frame.empty())
        {
            waitKey();
            break;
        }
        
        // Pre-processing
        // Perform:
        // - Mean subtraction
        // - Scaling
        // - (optional) Channel swapping
        
        // Creates a 4-dimensional blob from image
        
        // Arguments provided:
        // frame - input image
        // blob - output Mat
        // 1.0 - scale factor (multiplier for frame values)
        // Size - spatial size for blob
        // Scalar(...) - mean values which are to be subtracted from channels
        //               order is in (R,G,B)
        // true - swap first and last channels in the image (BGR to RGB)
        // false - resize image without cropping and preserving aspect ratio
        blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
        // Pass blob as input to EAST
        net.setInput(blob);
        // Make a forward pass
        net.forward(outs, outNames);
        
        // Scores
        Mat scores = outs[0];
        // Geometry
        Mat geometry = outs[1];

        // Decode predicted bounding boxes.
        std::vector<RotatedRect> boxes;
        std::vector<float> confidences;
        decode(scores, geometry, confThreshold, boxes, confidences);

        // Apply non-maximum suppression procedure.
        std::vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        // Render detections.
        Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            RotatedRect& box = boxes[indices[i]];

            Point2f vertices[4];
            box.points(vertices);
            for (int j = 0; j < 4; ++j)
            {
                vertices[j].x *= ratio.x;
                vertices[j].y *= ratio.y;
            }
            for (int j = 0; j < 4; ++j)
                line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);
        }

        // Put efficiency information.
        std::vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = format("Inference time: %.2f ms", t);
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

        imshow(kWinName, frame);
    }
    return 0;
}

void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
            std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    // Check dimensions
    CV_Assert(scores.dims == 4, geometry.dims == 4, scores.size[0] == 1,
              geometry.size[0] == 1, scores.size[1] == 1, geometry.size[1] == 5,
              scores.size[2] == geometry.size[2], scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scoresData = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* anglesData = geometry.ptr<float>(0, 4, y);
        for (int x = 0; x < width; ++x)
        {
            float score = scoresData[x];
            if (score < scoreThresh)
                continue;

            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            float offsetX = x * 4.0f, offsetY = y * 4.0f;
            float angle = anglesData[x];
            float cosA = std::cos(angle);
            float sinA = std::sin(angle);
            float h = x0_data[x] + x2_data[x];
            float w = x1_data[x] + x3_data[x];

            Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                           offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
            Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
            Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
            // Draw a rotated rectangle
            // 0.5 * (p1 + p3) = center of rectangle
            // Size2f(w,h) = width and height of rectangle
            // -angle*180/pi = Rotation angle in clockwise directions (in degrees)
            RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}
