import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Obtaining hand points with media pipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 0.8, 0.5)
mpDraw = mp.solutions.drawing_utils

# Time variables for FPS
pTime = 0
cTime = 0

# Capturing images through webcam
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting color to RGB
    results = hands.process(imgRGB)  # will process the given image for hands

    # check for multiple hands and extract
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                # find x and y coordinates of each landmark
                h, w, c = img.shape
                centerX, centerY = int(lm.x*w), int(lm.y*h)
                # draw circles on main landmarks
                cv2.circle(img, (centerX,centerY), 25, (255, 0, 255), cv2.FILLED)
            # draw hand landmarks on each hand
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    # create and display fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    # display video
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
