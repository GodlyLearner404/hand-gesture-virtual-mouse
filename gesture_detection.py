import cv2
import mediapipe as mp

class HandGestureDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []

        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                handLms = self.results.multi_hand_landmarks[hand_no]
                for id, lm in enumerate(handLms.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        self.lmList = lm_list
        
        return lm_list
    
    def fingersUp(self, hand_no=0):
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]
            lm = hand.landmark

            # Thumb (compare x not y for thumb)
            if lm[tip_ids[0]].x < lm[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers
            for id in range(1, 5):
                if lm[tip_ids[id]].y < lm[tip_ids[id] - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers
    
    def find_distance(self, p1, p2, img=None, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
        return length, (x1, y1), (x2, y2)

    