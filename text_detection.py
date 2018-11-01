# Import required modules

import cv2
import numpy as np
import argparse
from math import sin,cos

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

class point:
    """
    Class similar to Poin2f in C++
    Arguments:
        coords = (x,y)
    """
    def __init__(self,coords):
        self.x = coords[0]
        self.y = coords[1]
    def coords(self):
        return (self.x,self.y)

class RotatedRect(point):
    """
    Class for rotated rectangle
    Arguments:
        image = 3D Numpy array; image on which the border has to be added
        center = (x,y); center of the rotated rectangle
        size = (h,w); height and width of the rectangle
        angle = angle (in degrees) by which the rectangle is rotated in clockwise direction
        color = (B,G,R); color of the border
        lineThickness = integer; thickness of the border
    """
    def __init__(self,center,size,image=None,angle=0,color=(0,255,0),lineThickness=1):
        # Center of rectangle
        self.center = center
        # Size of rectangle
        self.size = size
        # If image argument is not a numpy.ndarray
        if type(image) != type(np.ones((5,5,3))):
            image = createBlankCanvas()
        else:
            image = image.copy()
        self.image = image
        self.color = color
        self.lineThickness=lineThickness
        self.angle = angle 
        self.center = point(center)
        self.w,self.h = size
        # Find coordinates of rectangle before rotation
        verticesOriginal = [(self.center.x-self.w/2, self.center.y-self.h/2),(self.center.x+self.w/2,self.center.y-self.h/2),
                        (self.center.x+self.w/2,self.center.y+self.h/2),(self.center.x-self.w/2,self.center.y+self.h/2)]
        # Get coordinates after rotation
        self.points = [point(((pt[0]-self.center.x)*cos(self.angle)-(pt[1]-self.center.y)*sin(self.angle)+self.center.x,
                              (pt[0]-self.center.x)*sin(self.angle)+(pt[1]-self.center.y)*cos(self.angle)+self.center.y)) for pt in verticesOriginal] 
        # Convert vertices to integers
        self.points = [point((int(pt.x),int(pt.y))) for pt in self.points]
        # Bounding box
        min_X = min([pt.x for pt in self.points])
        min_Y = min([pt.y for pt in self.points])
        # Get bounding box
        self.bbox = [min_X,min_Y,int(self.w), int(self.h)]
    def points(self):
        return self.points

def drawRotatedRect(r,image):
    """
    Function to draw a rotated rectangle on image
    Arguments:
        r = rotated rectangle object
        image = 3D Numpy array; image on which the rotated rectangle has to be drawn
    """
    # If image argument is not a numpy.ndarray
    if type(image) != type(np.ones((5,5,3))):
        # Create a black 300x300 px image
        image = createBlankCanvas()
    else:
        image = image.copy()
    # Draw rotated rectangle
    for i in range(len(r.points)):
        cv2.line(image,r.points[i].coords(),r.points[(i+1)%len(r.points)].coords(),r.color,r.lineThickness)
    return image

# Resizes a rotated rectangle object size to frame size
def resizeToFrame(r,frame):
    """
    Function to resize rotated rectangle object according to frame size
    Arguments:
        r = rotated rectangle object
        frame = frame according to the size of which r will be resized
    """
    ratio = (frame.shape[1] / args.width , frame.shape[0] / args.height)
    ratio = point(coords=ratio)
    print(ratio.x, ratio.y)
    for i in range(4):
        pt = r.points[i]
        pt.x *= ratio.x
        pt.y *= ratio.y
        pt.x = int(pt.x)
        pt.y = int(pt.y)
        r.points[i] = pt
    return r	

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
	
	# Initialize list of RotatedRect objects
    rotatedRectBoxes = []
    
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
            
            r = RotatedRect(center=(0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1])), size=(w, h), angle=-angle,image=frame)
            # print(r.points,r.bbox)
            
            # Append the rotated rectangle
            rotatedRectBoxes.append(r)
            
            # Append the bounding box rectangle
            detections.append(r.bbox)
            # Append score
            confidences.append(float(score))
    # Return detections and confidences
    return [rotatedRectBoxes,detections, confidences]
    
    
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
        
        if not hasFrame:
            cv2.waitKey()
            break
        
        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        
        # Create a 4D blob from frame.
        blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
        
        # Run the model
	t1 = cv2.getTickCount()
        net.setInput(blob)
        outs = net.forward(outNames)
	label = 'Inference time: %.2f ms' % ((cv2.getTickCount()-t1)*1000.0/cv2.getTickFrequency())
        
        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
    
        [rotatedRectBoxes, boxes, confidences] = decode(frame, scores, geometry, confThreshold)
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold,nmsThreshold)
    
        for i in range(0, len(indices)):
            # Get the bounding box
            box = boxes[indices[i][0]]
            # Get the rotated rectangle
            r = rotatedRectBoxes[indices[i][0]]
            # Resize to frame
            r = resizeToFrame(r,frame)

            # Draw the bounding box in green
            frame = drawRotatedRect(r,frame)
        
        # Put efficiency information
        #t, _ = net.getPerfProfile()
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    
        # Display the frame
        cv2.imshow(kWinName,frame)
