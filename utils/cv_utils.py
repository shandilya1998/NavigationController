import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from typing import Dict, List, NamedTuple, Optional, Tuple, Type


def circle_detect_v2(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,100)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    return markers

def test_circle_detector_v2(image_name, format = 'png'):
    image = cv2.imread(os.path.join('assets', 'plots', 'tests', '{}.{}'.format(image_name, format)))
    markers = circle_detect_v2(image)
    image[markers == -1] = [255,0,0]
    cv2.imwrite(os.path.join('assets', 'plots', 'tests', '{}_out.{}'.format(image_name, format)), image)

def supress(x, fs):
    for f in fs:
        distx = f.pt[0] - x.pt[0]
        disty = f.pt[1] - x.pt[1]
        dist = math.sqrt(distx*distx + disty*disty)
        if (f.size > x.size) and (dist<f.size/2):
            return True

def circle_detect(image, detector):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.blur(image, (7, 7))
    fs = detector.detect(image)
    fs.sort(key = lambda x: -x.size)
    sfs = [x for x in fs if not supress(x, fs)]
    return sfs

def test_circle_detector(image_name, format = 'png'):
    image = cv2.imread(os.path.join(
        'assets', 'plots', 'tests', '{}.{}'.format(image_name, format)
    ))
    detector = cv2.MSER_create()
    circles = circle_detect(image, detector)
    color = (255, 0, 0)
    thickness = 2
    if len(circles) > 0:
        for circle in circles:
            x = circle.pt[0]
            y = circle.pt[1]
            r = circle.size / 2
            center_coordinates = (int(x), int(y))
            image = cv2.circle(image, center_coordinates, int(r), color, thickness)
    cv2.imwrite(os.path.join('assets', 'plots', 'tests', '{}_out.{}'.format(image_name, format)), image)



def circle_detect_v1(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = cv2.blur(image, (3, 3))
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=95,
        param2=25,
        minRadius=0,
        maxRadius=0
    )
    return circles

def test_circle_detector_v1(image_name, format = 'png'):
    image = cv2.imread(os.path.join(
        'assets', 'plots', 'tests', '{}.{}'.format(image_name, format)
    ))
    circles = circle_detect_v1(image)
    if circles is not None:
        for pt in circles[0]:
            a, b, r = int(pt[0]), int(pt[1]), int(pt[2])
            image = cv2.circle(image, (a, b), r, (0, 255, 0), 2)
    cv2.imwrite(os.path.join('assets', 'plots', 'tests', '{}_out.{}'.format(image_name, format)), image)

#---------- Blob detecting function: returns keypoints and mask
#-- return keypoints, reversemask
def blob_detect(image,                  #-- The frame (cv standard)
                hsv_min,                #-- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,                #-- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,                 #-- blur value (default 0)
                blob_params=None,       #-- blob parameters (default None)
                search_window=None,     #-- window where to search as [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
                imshow=False
               ):


    #- Blur image to remove noise
    if blur > 0: 
        image    = cv2.blur(image, (blur, blur))

    if imshow:
        cv2.imshow('blur', image)

    #- Search window
    if search_window is None: search_window = [0.0, 0.0, 1.0, 1.0]
    
    #- Convert image from BGR to HSV
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #- Apply HSV threshold
    mask    = cv2.inRange(hsv,hsv_min, hsv_max)

    #- Show HSV Mask
    if imshow:
        cv2.imshow("HSV Mask", mask)
    
    #- dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=2)
    #- Show HSV Mask
    if imshow:
        cv2.imshow("Dilate Mask", mask)   
        
    mask = cv2.erode(mask, None, iterations=2)
    
    #- Show dilate/erode mask
    if imshow:
        cv2.imshow("Erode Mask", mask)
    
    #- Cut the image using the search mask
    mask = apply_search_window(mask, search_window)
    
    if imshow:
        cv2.imshow("Searching Mask", mask)
        #cv2.waitKey(0)

    #- build default blob detection parameters, if none have been provided
    if blob_params is None:
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 100;
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 20000
         
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
         
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
         
        # Filter by Inertia
        params.filterByInertia =True
        params.minInertiaRatio = 0.5
         
    else:
        params = blob_params     

    #- Apply blob detection
    detector = cv2.SimpleBlobDetector_create(params)

    # Reverse the mask: blobs are black on white
    reversemask = 255-mask
    
    if imshow:
        cv2.imshow("Reverse Mask", reversemask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    keypoints = detector.detect(reversemask)

    return keypoints, mask

#---------- Draw detected blobs: returns the image
#-- return(im_with_keypoints)
def draw_keypoints(image,                   #-- Input image
                   keypoints,               #-- CV keypoints
                   line_color=(0,0,255),    #-- line's color (b,g,r)
                   imshow=False             #-- show the result
                  ):
    
    #-- Draw detected blobs as red circles.
    #-- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), line_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        
    return(im_with_keypoints)

#---------- Draw search window: returns the image
#-- return(image)
def draw_window(image,              #- Input image
                window_adim,        #- window in adimensional units
                color=(255,0,0),    #- line's color
                line=5,             #- line's thickness
                imshow=False        #- show the image
               ):
    
    rows = image.shape[0]
    cols = image.shape[1]
    
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])  
    
    #-- Draw a rectangle from top left to bottom right corner
    image = cv2.rectangle(image,(x_min_px,y_min_px),(x_max_px,y_max_px),color,line)
    
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", image)

    return(image)

#---------- Draw X Y frame
#-- return(image)
def draw_frame(image,
               dimension=0.3,      #- dimension relative to frame size
               line=2              #- line's thickness
    ):
    
    rows = image.shape[0]
    cols = image.shape[1]
    size = min([rows, cols])
    center_x = int(cols/2.0)
    center_y = int(rows/2.0)
    
    line_length = int(size*dimension)
    
    #-- X
    image = cv2.line(image, (center_x, center_y), (center_x+line_length, center_y), (0,0,255), line)
    #-- Y
    image = cv2.line(image, (center_x, center_y), (center_x, center_y+line_length), (0,255,0), line)
    
    return (image)

#---------- Apply search window: returns the image
#-- return(image)
def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask = np.zeros(image.shape,np.uint8)
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    #--- return the mask
    return(mask)
    
#---------- Apply a blur to the outside search region
#-- return(image)
def blur_outside(image, blur=5, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask    = cv2.blur(image, (blur, blur))
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    
    
    #--- return the mask
    return(mask)
    
#---------- Obtain the camera relative frame coordinate of one single keypoint
#-- return(x,y)
def get_blob_relative_position(image, keyPoint):
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    # print(rows, cols)
    center_x    = 0.5*cols
    center_y    = 0.5*rows
    # print(center_x)
    x = (keyPoint.pt[0] - center_x)/(center_x)
    y = (keyPoint.pt[1] - center_y)/(center_y)
    return(x,y)

def test_blob_detector(image_name, format = 'png'):
    blue_min = (0, 25, 0)
    blue_max = (15, 255, 255)

    #--- Define area limit [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
    window = [0.0, 0.0, 1.0, 1.0]
    image = cv2.imread(os.path.join(
        'assets', 'plots', 'tests',
        '{}.{}'.format(image_name, format)
    ))

    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    image[0] = clahe.apply(image[0])
    image = cv2.merge(image)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    """

    #-- Detect keypoints
    keypoints, _ = blob_detect(
        image,
        blue_min,
        blue_max,
        blur = 5,
        blob_params = None,
        search_window = window,
        imshow = False
    )

    image    = blur_outside(image, blur=15, window_adim=window)
    #cv2.imshow("Outside Blur", image)
    cv2.waitKey(0)

    image     = draw_window(image, window, imshow=False)
    #-- enter to proceed
    cv2.waitKey(0)

    #-- click ENTER on the image window to proceed
    image     = draw_keypoints(image, keypoints, imshow=False)
    cv2.waitKey(0)
    #-- Draw search window

    image    = draw_frame(image)
    cv2.imwrite(
            os.path.join(
                    'assets', 
                    'plots', 
                    'tests', 
                    '{}_out.{}'.format(
                            image_name, 
                            format)), 
                    image)

"""
Object Detection Prototype
"""


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"

RED = Rgb(0.7, 0.1, 0.1)
rgb = RED
def get_hsv_ranges(rgb):
    if rgb == RED:
        return (0, 25, 0), (15, 255, 255)
    elif rgb == GREEN:
        return (36,0,0), (86,255,255)
    elif rgb == BLUE:
        return (94, 80, 2), (126, 255, 255)
min_range, max_range = get_hsv_ranges(rgb)
#---------- Draw detected blobs: returns the image
#-- return(im_with_keypoints)
def draw_keypoints(image,                   #-- Input image
                   keypoints,               #-- CV keypoints
                   line_color=(0,0,255),    #-- line's color (b,g,r)
                   imshow=False             #-- show the result
                  ):  
    
    #-- Draw detected blobs as red circles.
    #-- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), line_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
    
    return(im_with_keypoints)

#---------- Apply search window: returns the image
#-- return(image)
def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask = np.zeros(image.shape,np.uint8)
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    #--- return the mask
    return(mask)

#---------- Draw search window: returns the image
#-- return(image)
def draw_window(image,              #- Input image
                window_adim,        #- window in adimensional units
                color=(255,0,0),    #- line's color
                line=5,             #- line's thickness
                imshow=False        #- show the image
               ):

    rows = image.shape[0]
    cols = image.shape[1]

    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])

    #-- Draw a rectangle from top left to bottom right corner
    image = cv2.rectangle(image,(x_min_px,y_min_px),(x_max_px,y_max_px),color,line)

    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", image)

    return(image)

#---------- Draw X Y frame
#-- return(image)
def draw_frame(image,
               dimension=0.3,      #- dimension relative to frame size
               line=2              #- line's thickness
    ):  
    
    rows = image.shape[0]
    cols = image.shape[1]
    size = min([rows, cols])
    center_x = int(cols/2.0)
    center_y = int(rows/2.0)
    
    line_length = int(size*dimension)
    
    #-- X
    image = cv2.line(image, (center_x, center_y), (center_x+line_length, center_y), (0,0,255), line)
    #-- Y
    image = cv2.line(image, (center_x, center_y), (center_x, center_y+line_length), (0,255,0), line)
    
    return (image)

#---------- Apply a blur to the outside search region
#-- return(image)
def blur_outside(image, blur=5, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask    = cv2.blur(image, (blur, blur))
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    
    
    #--- return the mask
    return(mask)
    
#---------- Obtain the camera relative frame coordinate of one single keypoint
#-- return(x,y)
def get_blob_relative_position(image, keyPoint):
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    # print(rows, cols)
    center_x    = 0.5*cols
    center_y    = 0.5*rows
    # print(center_x)
    x = (keyPoint.pt[0] - center_x)/(center_x)
    y = (keyPoint.pt[1] - center_y)/(center_y)
    return(x,y)

def blob_detect(image,                  #-- The frame (cv standard)
                hsv_min,                #-- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,                #-- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,                 #-- blur value (default 0)
                blob_params=None,       #-- blob parameters (default None)
                search_window=None,     #-- window where to search as [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
                imshow=False
               ):


    #- Blur image to remove noise
    if blur > 0:
        image    = cv2.blur(image, (blur, blur))

    if imshow:
        cv2.imshow('blur', image)

    #- Search window
    if search_window is None: search_window = [0.0, 0.0, 1.0, 1.0]

    #- Convert image from BGR to HSV
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #- Apply HSV threshold
    mask    = cv2.inRange(hsv,hsv_min, hsv_max)

    #- Show HSV Mask
    if imshow:
        cv2.imshow("HSV Mask", mask)

    #- dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=2)
    #- Show HSV Mask
    if imshow:
        cv2.imshow("Dilate Mask", mask)

    mask = cv2.erode(mask, None, iterations=2)

    #- Show dilate/erode mask
    if imshow:
        cv2.imshow("Erode Mask", mask)

    #- Cut the image using the search mask
    mask = apply_search_window(mask, search_window)

    if imshow:
        cv2.imshow("Searching Mask", mask)

    #- build default blob detection parameters, if none have been provided
    if blob_params is None:
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()
    
        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 100;
    
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 10
        params.maxArea = 20000
    
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1 
    
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5 
    
        # Filter by Inertia
        params.filterByInertia =True
        params.minInertiaRatio = 0.5 
    
    else:
        params = blob_params    

    #- Apply blob detection
    detector = cv2.SimpleBlobDetector_create(params)

    # Reverse the mask: blobs are black on white
    reversemask = 255-mask
    
    if imshow:
        cv2.imshow("Reverse Mask", reversemask)
    
    keypoints = detector.detect(reversemask)

    return keypoints, reversemask
