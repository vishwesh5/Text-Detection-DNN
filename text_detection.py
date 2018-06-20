import cv2
import math
import numpy as np
from math import *
from util import *

import sys
import matplotlib.pyplot as plt
# const char* keys =
#     "{ help  h     | | Print help message. }"
#     "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
#     "{ model m     | | Path to a binary .pb file contains trained network.}"
#     "{ width       | 320 | Preprocess input image by resizing to a specific width. It should be multiple by 32. }"
#     "{ height      | 320 | Preprocess input image by resizing to a specific height. It should be multiple by 32. }"
#     "{ thr         | 0.5 | Confidence threshold. }"
#     "{ nms         | 0.4 | Non-maximum suppression threshold. }";

# path of image for now
i = sys.argv[1]
# model path - .pb format (for tensorflow)
m = "frozen_east_text_detection.pb"
# width and height - default values
width = 320
height = 320

# thresholds
thr = 0.5
nms = 0.4

def createBlankCanvas(color=(0,0,0),height=300,width=300):
    """
    Creates a blank canvas
    Arguments:
        color = (B,G,R): background color of canvas
        height = intger: height of canvas
        width = integer: width of canvas
    """
    # Separate colors
    blue,green,red = color
    # Create a temp canvas
    img = np.ones((height,width,3),dtype='uint8')
    # Change colors of canvas
    img[:,:,0] = blue
    img[:,:,1] = green
    img[:,:,2] = red
    return img

def rotatedRect(center,size,image=None,angle=0,color=(255,255,255),lineThickness=3):
    """
    Creates a rotated rectangle on an image
    Arguments:
        image = 3D Numpy array; image on which the border has to be added
        center = (x,y); center of the rotated rectangle
        size = (h,w); height and width of the rectangle
        angle = angle (in degrees) by which the rectangle is rotated in clockwise direction
        color = (B,G,R); color of the border
        lineThickness = integer; thickness of the border
    """
    # If image argument is not a numpy.ndarray
    if type(image) != type(np.ones((5,5,3))):
        # Create a black 300x300 px image
        image = createBlankCanvas()
    else:
        image = image.copy()
    # Convert angle to radians
    angle = angle*pi/180
    # Center coordinates
    centerX,centerY = center
    # Height and width
    h,w = size
    # Original vertices of the rectangle
    # top left, top right, bottom right, bottom left
    verticesOriginal = [(centerX-w/2, centerY-h/2),(centerX+w/2,centerY-h/2),
                        (centerX+w/2,centerY+h/2),(centerX-w/2,centerY+h/2)]
    newVertices = [((pt[0]-centerX)*cos(angle)-(pt[1]-centerY)*sin(angle)+centerX,(pt[0]-centerX)*sin(angle)+(pt[1]-centerY)*cos(angle)+centerY) for pt in verticesOriginal]
    # Convert vertices to integers
    newVertices = [(int(pt[0]),int(pt[1])) for pt in newVertices]
    for i in range(len(newVertices)):
        cv2.line(image,newVertices[i],newVertices[(i+1)%len(newVertices)],color,lineThickness)
    # Bounding box
    min_X = min([pt[0] for pt in newVertices])
    min_Y = min([pt[1] for pt in newVertices])
    max_X = max([pt[0] for pt in newVertices])
    max_Y = max([pt[1] for pt in newVertices])
    bbox = [min_X,min_Y,max_X-min_X,max_Y-min_Y]
    return bbox

# void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
            # std::vector<RotatedRect>& detections, std::vector<float>& confidences);
def decode(frame, scores, geometry, scoreThresh):
    # CV_ASSERT(scores.dims == 4, geometry.dims == 4, scores.size[0] == 1,
    #          geometry.size[0] == 1, scores.size[1] == 1, geometry.size[1] == 5,
    #          scores.size[2] == geometry.size[2], scores.size[3] == geometry.size[3])
    detections = []
    confidences = []
    height = scores.shape[2]
    width = scores.shape[3]

    for y in range(0, height):
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]

        for x in range(0, width):
            score = scoresData[x]
            if(score < scoreThresh):
                continue
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            # print(h, w)
            offset = np.array([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])
            print(sinA, h, cosA, w, offset[0], offset[1])
            p1 = [-sinA * h, -cosA * h] + offset
            p3 = [-cosA * w, sinA * w] + offset
            print("p1, p3, w, h : ", p1, p3, w, h)
            # print(0.5 * (p1 ))
            # r = rotatedRect((0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1])), (w, h), frame, -angle)
            # r = [0.5 * (p1 + p3), (w, h), -angle]
            r = [p1[0], p1[1], p3[0], p3[1]]
            detections.append(r)
            confidences.append(float(score))
    return [detections, confidences]
    
if __name__ == "__main__":
    confThreshold = thr
    nmsThreshold = nms
    inpWidth = width
    inpHeight = height

    model = m

    # Load network.
    net = cv2.dnn.readNet(model)
    

    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)

    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")


    frame = cv2.imread(sys.argv[1], 1)
    frame = cv2.resize(frame, (320, 320), interpolation = cv2.INTER_CUBIC)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    outs = net.forward(outNames)

    scores = outs[0]
    geometry = outs[1]

    [boxes, confidences] = decode(frame, scores, geometry, confThreshold)
    print(len(boxes), len(confidences))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # indices = [131, 241, 97, 26, 300, 411, 311, 105, 238, 406, 33, 94, 292, 30, 217, 256, 82, 263, 88, 393, 267, 276, 271, 381, 69, 122, 230, 330, 174, 259, 385, 39, 154, 147, 48, 354, 307, 321, 43, 287, 314, 102, 181, 341, 373, 77, 150, 166, 388, 135, 170, 127, 202, 358, 220, 140, 21, 177, 303, 214, 367, 73, 325, 328, 145, 163, 365, 212, 371, 283, 152, 401, 143, 318, 16, 183, 297, 332, 223, 274, 278, 397, 184, 402]
    ratio = (frame.shape[1]/inpWidth, frame.shape[0]/inpHeight)
    # print(len(indices))
    for i in range(0, len(indices)):
        box = boxes[indices[i][0]]
        # print(box)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
        # vertices = [list(pt) for pt in getPointsFromBoundingBox(box)]
        # for j in range(4):
        #     vertices[j][0] *= ratio[0]
        #     vertices[j][1] *= ratio[1]
        # vertices = [list(np.asarray(pt,dtype='uint8')) for pt in vertices]
        # vertices = [tuple(pt) for pt in vertices]
        # frame = plotRectFromPoints(vertices,image=frame)

        
    cv2.imshow("Frame",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
