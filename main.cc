#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void showImage(std::string path)
{
    cv::Mat img=cv::imread(path);
    imshow ("Image",img);
    cv::waitKey(0);

}
void showVideo(std::string path)
{
    cv::VideoCapture capV(path);
    // Check if the video was opened successfully
    if (!capV.isOpened())  
    {
        std::cerr << "Error: Could not open video file " << path << std::endl;
        return;
    }

    cv::Mat img;
    // This will be false when the video ends
    while (capV.read(img))  
    {
        imshow("Video", img);
        //break if ESC pressed
        if (cv::waitKey(10) == 27)  
            break;
    }
    // Release the video capture object
    capV.release();
    // Close the video window  
    cv::destroyWindow("Video");  
}

void startWebCam()
{
    cv::VideoCapture capV(0);
    cv::Mat img;
    while (true)
    {
        capV.read(img);
        imshow("WebCam",img);
        // press ESC to close the window
        if (cv::waitKey(10)==27) break;


    }
}
void basicImageProcessingOperations(std::string path)
{
    // Read the image from path
    cv::Mat img = cv::imread(path);

    // Declare matrices for processed images
    cv::Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

    // Convert the image to grayscale
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(imgGray, imgBlur, cv::Size(7, 7), 5, 0);

    // Detect edges using the Canny edge detector
    cv::Canny(imgBlur, imgCanny, 25, 75);

    // Create a 3x3 rectangular structuring element (kernel) for morphological ops
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // Dilate edges (thicken them)
    cv::dilate(imgCanny, imgDil, kernel);

    // Erode the dilated image (thin edges again, useful for noise removal)
    cv::erode(imgDil, imgErode, kernel);

    // Show all stages of image processing in separate windows
    cv::imshow("Original Image", img);
    cv::imshow("Image Gray", imgGray);
    cv::imshow("Image Blur", imgBlur);
    cv::imshow("Image Canny", imgCanny);
    cv::imshow("Image Dilation", imgDil);
    cv::imshow("Image Erode", imgErode);

    // Wait indefinitely until a key is pressed before closing windows
    cv::waitKey(0);

}
cv::Mat resizingImage(std::string path, int width, int height)  // Corrected "hight" → "height"
{
    cv::Mat img = cv::imread(path);
    if (img.empty())  
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        // Return empty Mat on failure
        return cv::Mat();  
    }
    
    cv::Mat resizedImage;
    cv::resize(img, resizedImage, cv::Size(width, height));  // Fixed spelling
    return resizedImage;
}

cv::Mat scalingImage(std::string path, float widthScale, float heightScale)  
{
    cv::Mat img = cv::imread(path);
    if (img.empty())  
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        // Return empty Mat on failure
        return cv::Mat();  
    }
    
    cv::Mat resizedImage;
    cv::resize(img, resizedImage, cv::Size(), widthScale, heightScale);  
    return resizedImage;
}
cv::Mat cropImage(std::string path,int x, int y, int width,int height)
{
    cv::Mat img=cv::imread(path);
    cv::Rect roi (x,y,width,height);
    cv::Mat imgCrop = img(roi);
    return imgCrop;
}
/**
 * @brief Creates an image with custom drawings (circle, rectangle, line, and text)
 * @param width Width of the output image
 * @param height Height of the output image
 * @param bgColor Background color (BGR format)
 * @return Mat object containing the drawn image
 */
cv::Mat createCustomDrawing(int width = 512, int height = 512, cv::Scalar bgColor = cv::Scalar(255, 255, 255)) 
{
    // Create blank image with specified background
    cv::Mat image(height, width, CV_8UC3, bgColor);

    // Draw filled orange circle at center
    cv::Point center(width/2, height/2);
    int circleRadius = 155;
    cv::Scalar circleColor(0, 69, 255);
    cv::circle(image, center, circleRadius, circleColor, cv::FILLED);

    // Draw white rectangle (acts as text background)
    cv::Point rectTopLeft(130, 226);
    cv::Point rectBottomRight(382, 286);
    cv::rectangle(image, rectTopLeft, rectBottomRight, cv::Scalar(255, 255, 255), cv::FILLED);

    // Draw white horizontal line below rectangle
    cv::Point lineStart(130, 296);
    cv::Point lineEnd(382, 296);
    cv::line(image, lineStart, lineEnd, cv::Scalar(255, 255, 255), 2);

    // Add orange text
    std::string text = "Learning OpenCV";
    cv::Point textPos(137, 262);
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    double fontScale = 0.75;
    int thickness = 2;
    putText(image, text, textPos, fontFace, fontScale, circleColor, thickness);

    return image;
}


/**
 * @brief Applies perspective transformation to extract a rectangular region from an image
 * @param inputImage Source image to be processed
 * @param srcPoints Array of 4 source points (corners of the object to extract)
 * @param width Width of the output warped image
 * @param height Height of the output warped image
 * @param debugMode If true, draws circles on source points
 * @return Warped image (Mat) containing the extracted rectangle
 */
cv::Mat extractPerspectiveRect(std::string inputImagePath, cv::Point2f srcPoints[4], float width, float height, bool debugMode = false) 
{
    // Validate input
    cv::Mat inputImage=cv::imread(inputImagePath);
    if (inputImage.empty()) 
    {
        std::cout << "Error: Input image is empty!" << std::endl;
        return cv::Mat();
    }

    cv::Mat outputImage;
    
    // Define destination points (rectangle)
    cv::Point2f dstPoints[4] = 
    {
        {0.0f, 0.0f},    // Top-left
        {width, 0.0f},   // Top-right
        {0.0f, height},  // Bottom-left
        {width, height}  // Bottom-right
    };

    // Get perspective transform matrix
    cv::Mat transformMatrix = getPerspectiveTransform(srcPoints, dstPoints);
    
    // Apply perspective warp
    cv::warpPerspective(inputImage, outputImage, transformMatrix, cv::Point(width, height));

    // Debug mode: draw circles on source points
    if (debugMode) 
    {
        cv::Mat debugImage = inputImage.clone();
        for (int i = 0; i < 4; i++) 
        {
            cv::circle(debugImage, srcPoints[i], 10, cv::Scalar(0, 0, 255), cv::FILLED);
        }
        cv::imshow("Debug Points", debugImage);
    }

    return outputImage;
}

/**
 * @brief Detects and isolates colors in an image using HSV color space with adjustable trackbars
 * @param path Path to the input image file
 */
void colorDetection(const std::string& path) 
{
    // ===== 1. Load and Verify Image =====
    cv::Mat img = cv::imread(path);
    if (img.empty()) 
    {
        std::cerr << "Error: Could not load image at " << path << std::endl;
        return;
    }

    // ===== 2. Initialize HSV Image and Mask =====
    cv::Mat imgHSV, imgMask,imgResult;;
    cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

    // ===== 3. Set Default HSV Threshold Values =====
    // These values can detect red objects by default
    int Hmin = 0, Smin = 0, Vmin = 0;
    int Hmax = 179, Smax = 255, Vmax = 255;

    // ===== 4. Create Trackbar Window =====
    cv::namedWindow("Trackbars", cv::WINDOW_NORMAL);
    cv::resizeWindow("Trackbars", 640, 200);

    // Create trackbars for HSV threshold adjustment
    cv::createTrackbar("Hue Min", "Trackbars", &Hmin, 179);
    cv::createTrackbar("Hue Max", "Trackbars", &Hmax, 179);
    cv::createTrackbar("Sat Min", "Trackbars", &Smin, 255);
    cv::createTrackbar("Sat Max", "Trackbars", &Smax, 255);
    cv::createTrackbar("Val Min", "Trackbars", &Vmin, 255);
    cv::createTrackbar("Val Max", "Trackbars", &Vmax, 255);

    // ===== 5. Main Processing Loop =====
    while (true) 
    {
        // Set lower and upper bounds for color detection
        cv::Scalar lower(Hmin, Smin, Vmin);
        cv::Scalar upper(Hmax, Smax, Vmax);

        // Create binary mask where white pixels represent detected color
        cv::inRange(imgHSV, lower, upper, imgMask);

        // Apply morphological operations to clean up the mask(Optional)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(imgMask, imgMask, cv::MORPH_OPEN, kernel);

        // Display images
        cv::imshow("Original Image", img);
        cv::imshow("HSV Image", imgHSV);
        cv::imshow("Color Mask", imgMask);

        // Exit loop when ESC key is pressed
        if (cv::waitKey(1) == 27) 
        {
            break;
        }
    }

    // ===== 6. Cleanup =====
    cv::destroyAllWindows();
}

void colorDetectionWebCam() 
{


    cv::VideoCapture capV(0);
    cv::Mat img;
    // ===== Set Default HSV Threshold Values =====
    // These values can detect red objects by default
    int Hmin = 0, Smin = 0, Vmin = 0;
    int Hmax = 179, Smax = 255, Vmax = 255;

    // =====  Create Trackbar Window =====
    cv::namedWindow("Trackbars", cv::WINDOW_NORMAL);
    cv::resizeWindow("Trackbars", 640, 200);

    // Create trackbars for HSV threshold adjustment
    cv::createTrackbar("Hue Min", "Trackbars", &Hmin, 179);
    cv::createTrackbar("Hue Max", "Trackbars", &Hmax, 179);
    cv::createTrackbar("Sat Min", "Trackbars", &Smin, 255);
    cv::createTrackbar("Sat Max", "Trackbars", &Smax, 255);
    cv::createTrackbar("Val Min", "Trackbars", &Vmin, 255);
    cv::createTrackbar("Val Max", "Trackbars", &Vmax, 255);
    while (capV.read(img))
    {
        // ===== Initialize HSV Image and Mask =====
        cv::Mat imgHSV, imgMask,imgResult;;
        cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

        // Set lower and upper bounds for color detection
        cv::Scalar lower(Hmin, Smin, Vmin);
        cv::Scalar upper(Hmax, Smax, Vmax);

        // Create binary mask where white pixels represent detected color
        cv::inRange(imgHSV, lower, upper, imgMask);

        // Apply morphological operations to clean up the mask(Optional)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(imgMask, imgMask, cv::MORPH_OPEN, kernel);

        // Display images
        cv::imshow("Original Image", img);
        //cv::imshow("HSV Image", imgHSV);
        cv::imshow("Color Mask", imgMask);

        // Exit loop when ESC key is pressed
        if (cv::waitKey(1) == 27) 
        {
            break;
        }
        
    }

    // ===== Cleanup =====
    cv::destroyAllWindows();
}



void getContors(cv::Mat imgDil,cv::Mat imgContours )
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(imgDil, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //cv::drawContours(imgContours, contours, -1, cv::Scalar(0, 255, 0), 2);
    std::string objectType;

    std::vector<std::vector<cv::Point>> contourPolygon(contours.size());
    std::vector<cv::Rect> boundingBoxPoints(contours.size());
    for ( int i=0; i<contours.size();i++ )
    {
        int area = contourArea(contours[i]);
        if (area>1000)
        {
            float perimeter = cv::arcLength(contours[i], true);
            cv::approxPolyDP(contours[i], contourPolygon[i], 0.02 * perimeter, true);
            
            boundingBoxPoints[i]=cv::boundingRect(contourPolygon[i]);       
            int objCorners = (int)contourPolygon[i].size();
            if (objCorners==3) objectType="Triangle";
            else if(objCorners==4)
            {
                float aspRatio = (float)boundingBoxPoints[i].width / (float)boundingBoxPoints[i].height;
                if (aspRatio> 0.95 && aspRatio< 1.05){ objectType = "Square"; }
                else { objectType = "Rect";}

            }
            else if (objCorners > 4) { objectType = "Circle"; }

            cv::drawContours(imgContours, contourPolygon, i, cv::Scalar(255, 0, 255), 2);
            cv::rectangle(imgContours, boundingBoxPoints[i].tl(), boundingBoxPoints[i].br(), cv::Scalar(0, 255, 0), 5);
            putText(imgContours, objectType, { boundingBoxPoints[i].x,boundingBoxPoints[i].y - 5 }, cv::FONT_HERSHEY_PLAIN,1, cv::Scalar(0, 69, 255), 2);
        }

    }


}
void contoursDetection(std::string & path)
{
    cv::Mat img=cv::imread(path);
    cv::Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;
    // Preprocessing
    cv::cvtColor (img,imgGray,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(imgGray,imgBlur,cv::Size(3,3),3,0);
    cv::Canny(imgBlur,imgCanny,25,75);// edge detector
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(imgCanny, imgDil, kernel);

    cv::Mat imgContours = img.clone();
    getContors(imgDil,imgContours);
    //imshow("Image", img);
	//imshow("Image Gray", imgGray);
	//imshow("Image Blur", imgBlur);
	//imshow("Image Canny", imgCanny);
	//imshow("Image Dil", imgDil);
    //imshow("Image Contours", imgContours);
    cv::imshow("Image Dil", imgDil);
    cv::imshow("Image Contours", imgContours);

}
cv::Mat detectAndDrawFaces(std::string inputImagePath, std::string cascadePath) 
{
    cv::Mat inputImage=cv::imread(inputImagePath);
    // Load the pre-trained Haar Cascade classifier
    cv::CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) 
    {
        std::cout << "Error: Could not load face cascade classifier from: " << cascadePath << std::endl;
        return inputImage; // Return original image if classifier fails to load
    }

    // Vector to store detected face rectangles
    std::vector<cv::Rect> faces;
    
    // Detect faces in the image
    // Parameters:
    // 1.1 = scale factor (how much image is reduced at each scale)
    // 10 = minNeighbors (how many detections are needed to confirm a face)
    faceCascade.detectMultiScale(inputImage, faces, 1.1, 10);

    // Draw rectangles around detected faces
    for (const auto& face : faces) 
    {
        // Draw rectangle with:
        // - Magenta color (BGR format: 255,0,255)
        // - Thickness of 3 pixels
        rectangle(inputImage, face.tl(), face.br(), cv::Scalar(255, 0, 255), 3);
    }

    return inputImage;
}

int main ()
{
    std::string imgPath="resources/test.png";
    std::string videoPath="resources/test_video.mp4";
    std::string cardImgPath="resources/cards.jpg";
    std::string shapeImgPath="resources/shapes.png";
    std::string lamboImgPath="resources/lambo.png";
    std::string preTrainedHaarCascadeModel="resources/haarcascade_frontalface_default.xml";

    //showImage (imgPath);
    //showVideo(videoPath);
    //startWebCam();
    //basicImageProcessingOperations(imgPath);
    //cv::imshow("Resized mage",resizingImage(imgPath,122,122));
    //cv::imshow("Scalled mage",scalingImage(imgPath,0.5,0.5));
    //cv::imshow("Croped mage",cropImage(imgPath,10,10,100,100));
    //cv::imshow("Custom mage",createCustomDrawing());
    //cv::waitKey(0);  
    
    
    /*
    //Example usecase for extractPerspectiveRect()
    // Define source points (corners of the card)
    cv::Point2f srcPoints[4] = 
    {
        {529, 142},  // Top-left
        {771, 190},  // Top-right
        {405, 395},  // Bottom-left
        {674, 457}   // Bottom-right
    };

    // Set output dimensions
    float cardWidth = 250;
    float cardHeight = 350;

    // Extract the card using perspective transformation
    cv::Mat warpedCard = extractPerspectiveRect(cardImgPath, srcPoints, cardWidth, cardHeight, true);

    // Display results
    cv::imshow("warpedCard",warpedCard);
    cv::waitKey(0); 
    */
    //colorDetection(shapeImgPath);
    //contoursDetection(shapeImgPath);
    //cv::imshow("Image",detectAndDrawFaces(imgPath,preTrainedHaarCascadeModel));
    //cv::waitKey(0); 
    colorDetectionWebCam();
    return 0;
}

