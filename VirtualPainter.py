import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

# import overlay headers
folderPath = "VirtualPainterResources"
headerList = os.listdir(folderPath)
overlayList = []

for imagePath in headerList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)

header = overlayList[0]

# drawing variables
brushThickness = 15
drawColor = (255, 0, 255)

# run webcam
cap = cv2.VideoCapture(0)

detector = htm.handDetector(minDetectionConfidence=0.85)
xPrev, yPrev = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    # import image and flip image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if success:
        # find hand landmarks
        img = detector.findHands(img)
        landmarks = detector.findPosition(img, draw=False)

        if detector.results.multi_hand_landmarks:
            # tip of index finger and middle finger
            indexX, indexY = landmarks[8][1:]
            middleX, middleY = landmarks[12][1:]

            # check which fingers are up
            fingers = detector.fingersUp()

            # selection mode (two fingers)
            if fingers[1] and fingers[2]:
                # check for selection
                if indexY < 100:
                    if 150 < indexX < 252:
                        # magenta
                        header = overlayList[4]
                        drawColor = (255, 0, 255)
                    elif 352 < indexX < 452:
                        # red
                        header = overlayList[3]
                        drawColor = (0, 0, 255)
                    elif 572 < indexX < 672:
                        # aqua
                        header = overlayList[1]
                        drawColor = (255, 191, 0)
                    elif 802 < indexX < 902:
                        # white
                        header = overlayList[2]
                        drawColor = (255, 255, 255)
                    elif 1022 < indexX < 1122:
                        # eraser
                        header = overlayList[0]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (indexX, indexY - 25), (middleX, middleY + 25), drawColor, cv2.FILLED)
                xPrev, yPrev = indexX, indexY
            # Drawing mode (index fingers)
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (indexX, indexY), 15, drawColor, cv2.FILLED)
                # first frame edge case
                if xPrev == 0 and yPrev == 0:
                    xPrev, yPrev = indexX, indexY
                # draw line
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xPrev, yPrev), (indexX, indexY), drawColor, brushThickness + 30)
                    cv2.line(imgCanvas, (xPrev, yPrev), (indexX, indexY), drawColor, brushThickness + 30)
                else:
                    cv2.line(img, (xPrev, yPrev), (indexX, indexY), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xPrev, yPrev), (indexX, indexY), drawColor, brushThickness)
                xPrev, yPrev = indexX, indexY

        # layer canvas on top of img
        # add image inverse and image => black drawings on canvas
        # or operation with image and imgCanvas to regain color on line
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # slice header into video and combine canvas
        img[0:100, 0:1280] = header

        # display video
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Video capture error")
        exit(1)

cap.release()
cv2.destroyAllWindows()
