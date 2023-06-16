# Housie game 

# Import modules

import cv2
import numpy as np
import imutils
import easyocr
import os
import time


# Script time
start_time = time.time()

# Configrations
classes = np.arange(0, 10)
input_size = 48

# Common functions

def getPerspectiveTransform(location, width, height):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix
    
def getPerspective(img, location, height = 900, width = 900):
    return cv2.warpPerspective(
        img, getPerspectiveTransform(location, width, height), (width, height))

def getInvPerspective(img, masked_num, location, height = 900, width = 900):
    return cv2.warpPerspective(
        masked_num, getPerspectiveTransform(location, width, height), (img.shape[1], img.shape[0]))

# Starting

def findBoard(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours  = imutils.grab_contours(keypoints)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = getPerspective(img, location)
    return result, location

def splitBoxes(board):
    rows = np.vsplit(board,3)
    count = 0
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            cv2.imwrite(os.path.join('../temp' , f'housie-{count}.jpg'), box)
            count+=1
    cv2.destroyAllWindows()
    return 


# Easy ocr
def easy_formate(text):
    if text == []:
        return text
    else:
        for i in text:
            for j in i:
                if type(j) == str:
                    return j
                else:
                    continue

def toCreate():
    dict = {}
    count = 1
    reader = easyocr.Reader(['en'], gpu=False)
    
    list_dir = os.listdir('../temp')
    list_dir.sort(key=lambda x: int(x.partition('-')[2].partition('.')[0]))
    print(list_dir)
    for file in list_dir:
        dict[count] = reader.readtext(f'../temp/{file}')
        # os.remove(f'../temp/{file}')
        count+=1
    count = 1
   
    for value in dict.values():
        dict[count] = easy_formate(value)
        count+=1
    return dict

# Read image

img = cv2.imread('images/housie6.png')

board, location = findBoard(img)

gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
splitBoxes(gray) # Cut box into small parts.
print(toCreate()) # Read text inside folder.

print("--- %s seconds ---" % (time.time() - start_time))
