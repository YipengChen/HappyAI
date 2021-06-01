import cv2
import sys
from os import path   
sys.path.append(path.dirname(path.dirname(__file__)))
from base_ai.style_transfer_ai import StyleTransfer

cap = cv2.VideoCapture(0)
# mosaic, candy, rain-princess, udnie, pointilism, cartoon
style_transfer = StyleTransfer(style='cartoon')

while cap.isOpened():
  
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image = cv2.resize(image, (320, 180))
    results = style_transfer.inference(image)
    
    cv2.imshow('Face Detection', results)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
