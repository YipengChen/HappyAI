import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.hand.hand_detection import HandDetection
from base_ai.hand.hand_gesture_recognition import HandGestureRecognition

cap = cv2.VideoCapture(0)
hand_detection = HandDetection()
hand_gesture_recognition = HandGestureRecognition()

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    results = hand_detection.inference(image)
    multi_hand_landmarks = hand_detection.get_all_landmarks(results, image.shape)
    for hand_landmarks in multi_hand_landmarks:
        min_x, min_y = int(hand_landmarks[:, 0].min()), int(hand_landmarks[:, 1].min())
        gesture_recognitio_result = hand_gesture_recognition.inference(hand_landmarks)
        image = hand_gesture_recognition.draw(image, gesture_recognitio_result, (min_x-50, min_y-50, 100, 100))
    
    cv2.imshow('Hand Detection', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
