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
    draw = cv2.resize(image.copy(), (320, 240))
    draw = style_transfer.inference(draw)
    draw = cv2.resize(draw, (image.shape[1], image.shape[0]))
    cv2.imshow('Face Detection', draw)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
