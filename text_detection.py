# Import required modules

import cv2
import numpy as np
import argparse
from math import sin,cos, PI
import sys

############ Add argument parser for command line arguments ############

parser = argparse.ArgumentParser(description='Use this script to run text detection deep learning networks using OpenCV.')
# Input argument
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
# Model argument
parser.add_argument('--model', required=True,
                    help='Path to a binary .pb file of model contains trained weights.'
                    )
# Width argument
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.'
                   )
# Height argument
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.'
                   )
# Confidence threshold
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.'
                   )
# Non-maximum suppression threshold
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.'
                   )

args = parser.parse_args()

############ Utility functions ############

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

def decode(frame, scores, geometry, scoreThresh):
    
    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    
    # Initialize detections and confidences
    detections = []
    confidences = []
    
    height = scores.shape[2]
    width = scores.shape[3]

    for y in range(0, height):
        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]

        for x in range(0, width):
            score = scoresData[x]
            
            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue
                
            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            
            angle = anglesData[x]
            
            # Calculate cos and sin of angle
            cosA = cos(angle)
            sinA = sin(angle)
            
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]
            
            # Calculate offset
            offset = np.array([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])
            # Find points for rectangle
            p1 = [-sinA * h, -cosA * h] + offset
            p3 = [-cosA * w, sinA * w] + offset
            # r should be a rotated rectangle, but it is giving poor results
            # TO BE FIXED
            
            # r = rotatedRect((0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1])), (w, h), frame, -angle)
            
            # Append the bounding box rectangle
            r = [p1[0], p1[1], p3[0], p3[1]]
            detections.append(r)
            # Append score
            confidences.append(float(score))
    # Return detections and confidences
    return [detections, confidences]
    
    
if __name__ == "__main__":
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model
    
    # Load network
    net = cv2.dnn.readNet(model)
    # Create a new named window
    kWinName = "EAST: An Efficient and Accurate Scene Text Detector"
    cv2.namedWindow(kWinName, cv2.WINDOW_NORMAL)

    outNames = []
    outNames.append("feature_fusion/Conv_7/Sigmoid")
    outNames.append("feature_fusion/concat_3")
    
    # Open a video file or an image file or a camera stream
    cap = cv2.VideoCapture(args.input if args.input else 0)
    
    while cv2.waitKey(1) < 0:
        # Read frame
        hasFrame, frame = cap.read()
        # If frame not found
        if not hasFrame:
            cv2.waitKey()
            break
        # Get frame height
        frameHeight = frame.shape[0]
        # Get frame width
        frameWidth = frame.shape[1]
        
        # Resize the frame to input width and height
        frame = cv2.resize(frame, (inpWidth, inpHeight), interpolation = cv2.INTER_CUBIC)
        
        # Create a 4D blob from frame.
        blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        
        # Run the model
        net.setInput(blob)
        outs = net.forward(outNames)
        
        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]

        [boxes, confidences] = decode(frame, scores, geometry, confThreshold)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold,nmsThreshold)

        # ratio = (frame.shape[1]/inpWidth, frame.shape[0]/inpHeight)
        for i in range(0, len(indices)):
            # Get the bounding box
            box = boxes[indices[i][0]]
            # Draw the bounding box in green
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0))
            
            # The following code is not necessary since the frame has already been resized
            
            # Get the vertices of the bounding box
            # vertices = [list(pt) for pt in getPointsFromBoundingBox(box)] 
            # for j in range(4):
            #    vertices[j][0] *= ratio[0]
            #    vertices[j][1] *= ratio[1]
            # vertices = [list(np.asarray(pt,dtype='uint8')) for pt in vertices]
            # vertices = [tuple(pt) for pt in vertices]
            # Plot the rectangle
            # frame = plotRectFromPoints(vertices,image=frame)

        # Put efficiency information
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        
        # Display the frame
        cv2.imshow(kWinName,frame)
