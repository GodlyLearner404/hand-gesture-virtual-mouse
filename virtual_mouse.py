import cv2
import pyautogui
import numpy as np
from gesture_detection import HandGestureDetector

detector = HandGestureDetector(detection_confidence=0.8)

# Initialize camera and detector
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Frame reduction for smoother edges
frameR = 100
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:] # Middle finger tip

        # Fingers up
        fingers = detector.fingersUp()

        # Moving Mode (only index finger up)
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, 640 - frameR), (0, screen_width))
            y3 = np.interp(y1, (frameR, 480 - frameR), (0, screen_height))

            # Smooth values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move mouse
            pyautogui.moveTo(screen_width - clocX, clocY)  # invert x for mirror

            plocX, plocY = clocX, clocY

        # Clicking Mode (both index and middle finger up)
        if fingers[1] == 1 and fingers[2] == 1:
            length, _, _ = detector.find_distance(8, 12, img)
            if length < 40:
                pyautogui.click()
                cv2.circle(img, ((x1 + x2)//2, (y1 + y2)//2), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break
