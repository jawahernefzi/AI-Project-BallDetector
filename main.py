import math
import cv2
from cvzone.ColorModule import ColorFinder
import cvzone
import numpy as np
# Initialize the Video
cap = cv2.VideoCapture('Videos/vid (4).mp4')
# the color Finder object
myColorFinder = ColorFinder(False)
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
while True:
    success, img = cap.read()
    img = img[0:900, :]  # crop the image

    # Find the color Ball
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Find location of the ball
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)
    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Polynomial regression: y = Ax*2 + Bx + C
        # Find the coefficients
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            pos = (posX, posY)
            cv2.circle(imgContours, pos, 10, (0, 255, 0), cv2.FILLED)

        # Draw the connecting lines
        curve = np.array([(x, int(A * x ** 2 + B * x + C)) for x in xList])
        cv2.polylines(imgContours, [curve], isClosed=False, color=(255, 0, 255), thickness=5)

        #Prediction with regression
        # X values 330 to 430 Y 590
        a = A
        b = B
        c = C - 590

        x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))

        prediction = 330 < x < 430
        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 150),
                               scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 150),
                               scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Display
    imgContours = cv2.resize(imgContours, (0, 0), None, 0.7, 0.7)
    # cv2.imshow("Image", img)
    cv2.imshow("ImageColor", imgContours)
    cv2.waitKey(100)
