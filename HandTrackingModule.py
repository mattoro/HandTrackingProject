import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, minDetectionConfidence=0.6, minTrackConfidence=0.6):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackConfidence = minTrackConfidence

        # Obtaining hand points with media pipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.minDetectionConfidence, self.minTrackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting color to RGB
        self.results = self.hands.process(imgRGB)  # will process the given image for hands
        # check for multiple hands and extract
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    # draw hand landmarks on each hand
                    self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNumber=0, draw=True):
        # adds position of all 20 landmarks to lmList
        lmList = []

        # check if hand detected
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumber]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                centerX, centerY = int(lm.x * w), int(lm.y * h)
                lmList.append([id, centerX, centerY])

                if draw:
                    cv2.circle(img, (centerX, centerY), 7, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    # Time variables for FPS
    pTime = 0
    cTime = 0

    # Capturing images through webcam
    while True:
        success, img = cap.read()

        # hand detection
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # create and display fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        # display video
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
